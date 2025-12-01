from typing import Optional, Tuple
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax

from torch_geometric.nn.models import GIN
from torch_geometric.nn.glob import global_mean_pool

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.agent import Agent
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper, GinAgentObsTensor
from scheduler.rl_model.agents.gin_agent.shared_debug import set_last_action_probs


# Base Gin Network
# ----------------------------------------------------------------------------------------------------------------------


class BaseGinNetwork(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device, num_layers: int = 2) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.task_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.vm_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.graph_network = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=embedding_dim,
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        # Normalizers
        max_vm_cores = max(obs.vm_cpu_cores.max().item(), 1.0)

        task_features = [
            obs.task_state_scheduled,
            obs.task_state_ready,
            obs.task_length,
            obs.task_completion_time,
            obs.task_memory_req_mb / 1000.0,
            obs.task_cpu_req_cores / max_vm_cores,
        ]
        vm_features = [
            obs.vm_completion_time,
            1 / (obs.vm_speed + 1e-8),
            obs.vm_energy_rate,
            obs.vm_memory_mb / 1000.0,
            obs.vm_available_memory_mb / 1000.0,
            obs.vm_used_memory_fraction,
            obs.vm_active_tasks_count,
            # obs.vm_next_release_time,
            obs.vm_cpu_cores / max_vm_cores,
            obs.vm_available_cpu_cores / max_vm_cores,
            obs.vm_used_cpu_fraction_cores,
            # obs.vm_next_core_release_time,
        ]

        # Encode tasks
        task_x = torch.stack(task_features, dim=-1)
        task_h: torch.Tensor = self.task_encoder(task_x)

        # Encode VMs
        vm_x = torch.stack(vm_features, dim=-1)
        vm_h: torch.Tensor = self.vm_encoder(vm_x)

        # Structuring nodes as [0, 1, ..., T-1] [T, T+1, ..., T+VM-1], edges are between Tasks -> Compatible VMs
        task_vm_edges = obs.compatibilities.clone()
        task_vm_edges[1] = task_vm_edges[1] + num_tasks  # Reindex VMs

        # Get features
        node_x = torch.cat([task_h, vm_h])
        edge_index = torch.cat([task_vm_edges, obs.task_dependencies], dim=-1)

        # Get embeddings
        batch = torch.zeros(num_tasks + num_vms, dtype=torch.long, device=self.device)
        node_embeddings = self.graph_network(node_x, edge_index=edge_index)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


# Gin Actor
# ----------------------------------------------------------------------------------------------------------------------


class GinActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.network = BaseGinNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=num_layers,
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        num_tasks = obs.task_completion_time.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        _, edge_embeddings, graph_embedding = self.network(obs)

        # Get edge embedding scores
        rep_graph_embedding = graph_embedding.expand(edge_embeddings.shape[0], self.embedding_dim)
        edge_embeddings = torch.cat([edge_embeddings, rep_graph_embedding], dim=1)
        edge_embedding_scores: torch.Tensor = self.edge_scorer(edge_embeddings)

        # Extract the exact edges
        task_vm_edge_scores = edge_embedding_scores.flatten()
        task_vm_edge_scores = task_vm_edge_scores[: obs.compatibilities.shape[1]]

        # Actions scores should be the value in edge embedding, but -inf on invalid actions
        action_scores = torch.ones((num_tasks, num_vms), dtype=torch.float32).to(self.device) * -1e8
        action_scores[obs.compatibilities[0], obs.compatibilities[1]] = task_vm_edge_scores
        # Mask out not-ready tasks
        action_scores[obs.task_state_ready == 0, :] = -1e8
        # Mask out already-scheduled tasks (double safety)
        action_scores[obs.task_state_scheduled == 1, :] = -1e8

        return action_scores


# Gin Critic
# ----------------------------------------------------------------------------------------------------------------------


class GinCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device, num_layers: int = 2) -> None:
        super().__init__()
        self.device = device

        self.network = BaseGinNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
            num_layers=num_layers,
        )
        self.graph_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        # Critic value is derived from global graph state
        _, _, graph_embedding = self.network(obs)
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# Gin Agent
# ----------------------------------------------------------------------------------------------------------------------


class GinAgent(Agent, nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.mapper = GinAgentMapper(MAX_OBS_SIZE)
        # Configuration: 16-d embeddings and 3 GIN layers
        self.actor = GinActor(hidden_dim=32, embedding_dim=16, device=device, num_layers=3)
        self.critic = GinCritic(hidden_dim=32, embedding_dim=16, device=device, num_layers=3)
        import os
        self.debug: bool = os.environ.get("GIN_DEBUG", "0") == "1"

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            value = self.critic(decoded_obs)
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            action_scores = self.actor(decoded_obs)
            action_scores = action_scores.flatten()
            action_probabilities = softmax(action_scores, dim=0)

            # Action mask: only allow choices from READY and NOT-SCHEDULED tasks (across all VMs),
            # AND restrict to COMPATIBLE (task, vm) pairs to avoid invalid EnvAction selections.
            num_tasks = decoded_obs.task_state_ready.shape[0]
            num_vms = decoded_obs.vm_completion_time.shape[0]
            ready_mask_tasks: torch.Tensor = decoded_obs.task_state_ready.bool()
            not_scheduled_mask_tasks: torch.Tensor = (decoded_obs.task_state_scheduled == 0)
            valid_task_mask = ready_mask_tasks & not_scheduled_mask_tasks
            ready_mask_flat = valid_task_mask.repeat_interleave(num_vms)
            # Build compatibility mask over flat actions
            compat_mask_flat = torch.zeros(num_tasks * num_vms, dtype=torch.bool, device=self.device)
            # raise Exception ('compat_mask_flat')
            # decoded_obs.compatibilities is shape (2, E) with edges (task, vm)
            if decoded_obs.compatibilities.numel() > 0:
                comp_t = decoded_obs.compatibilities[0].to(torch.long)
                comp_v = decoded_obs.compatibilities[1].to(torch.long)
                comp_flat_idx = comp_t * num_vms + comp_v
                # Ensure indices are within bounds
                valid_mask = (comp_flat_idx >= 0) & (comp_flat_idx < num_tasks * num_vms)
                compat_mask_flat[comp_flat_idx[valid_mask]] = True
            # Combined mask
            combined_mask_flat = ready_mask_flat & compat_mask_flat

            # Zero-out probabilities for invalid actions, then renormalize
            mask_float = combined_mask_flat.to(action_probabilities.dtype)
            action_probabilities = action_probabilities * mask_float
            action_scores = action_scores * mask_float
            total_prob = action_probabilities.sum()
            if total_prob.item() <= 0:
                # Fallbacks to avoid invalid sampling
                if combined_mask_flat.any():
                    # Uniform over actions involving READY ∧ NOT-SCHEDULED ∧ COMPATIBLE
                    mask = combined_mask_flat.to(action_probabilities.dtype)
                    action_probabilities = mask / mask.sum()
                    action_scores = mask
                else:
                    # No ready tasks detected (should be rare) -> uniform over all actions
                    print(f'No ready tasks detected (should be rare) -> uniform over all actions')
                    action_probabilities = torch.ones_like(action_probabilities) / action_probabilities.numel()
                    action_scores = torch.ones_like(action_scores) / action_scores.numel()

            probs = Categorical(action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()
            # Export last probs for wrapper-side debug (flat vector)
            try:
                set_last_action_probs(action_probabilities.detach().cpu().numpy())
            except Exception:
                pass
            if self.debug:
                ca = int(chosen_action.item())
                if 0 <= ca < num_tasks * num_vms:
                    t_id = ca // num_vms
                    v_id = ca % num_vms
                    is_ready = bool(valid_task_mask[t_id].item()) if 0 <= t_id < num_tasks else False
                    is_compat = bool(compat_mask_flat[ca].item())
                    is_valid = bool((valid_task_mask.repeat_interleave(num_vms))[ca].item() and compat_mask_flat[ca].item())
                    if not is_valid:
                        print(f"[GIN_DEBUG][agent] INVALID sample idx={ca} -> (t={t_id}, v={v_id}) ready={is_ready} compat={is_compat}")
            value = self.critic(decoded_obs)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(probs.log_prob(chosen_action))
            all_entropies.append(probs.entropy())
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values
