"""
Gradient-Based Domain Discovery with Trained Model

1. Train the model for 20,000 steps on MIXED data (both QF and BN)
2. Then collect gradients from batches of each domain
3. Cluster gradients to discover domains

This should work better because the trained model has learned
domain-specific patterns, so gradients will be more meaningful.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

from scheduler.rl_model import ablation_gnn as AG
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.dataset_generator.gen_dataset import DatasetArgs

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)


def create_env(domain: str, seed: int):
    if domain == "qf":
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=8, gnp_max_n=12, gnp_p=0.2,
            workflow_count=3, host_count=2, vm_count=4, style="wide",
        )
    else:
        ds = DatasetArgs(
            dag_method="gnp", gnp_min_n=12, gnp_max_n=20, gnp_p=0.8,
            workflow_count=5, host_count=2, vm_count=4, style="long_cp",
        )
    ds.seed = seed
    env = CloudSchedulingGymEnvironment(dataset_args=ds)
    env = GinAgentWrapper(env)
    return env


def train_mixed_ppo(agent, optimizer, envs_qf, envs_bn, device, total_steps=20000, num_steps=128):
    """
    Train agent on mixed data from both domains using PPO.
    Similar to ablation_gnn_traj_main.py training loop.
    """
    print(f"\nTraining for {total_steps:,} steps on mixed data...")
    
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    vf_coef = 0.5
    ent_coef = 0.0
    max_grad_norm = 0.5
    update_epochs = 4
    
    all_envs = envs_qf + envs_bn
    num_envs = len(all_envs)
    
    # Initialize observations
    obs_list = []
    for env in all_envs:
        obs, _ = env.reset()
        obs_list.append(obs)
    
    global_step = 0
    iteration = 0
    
    while global_step < total_steps:
        iteration += 1
        
        # Collect rollout
        obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        
        for step in range(num_steps):
            obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs_t)
            
            obs_buf.append(obs_t)
            actions_buf.append(actions)
            logprobs_buf.append(logprobs)
            values_buf.append(values.flatten())
            
            # Step all environments
            new_obs_list = []
            rewards = []
            dones = []
            
            for i, env in enumerate(all_envs):
                action = actions[i].cpu().numpy().item()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if done:
                    next_obs, _ = env.reset()
                
                new_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(float(done))
            
            obs_list = new_obs_list
            rewards_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            dones_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            
            global_step += num_envs
        
        # Stack buffers
        obs_buf = torch.stack(obs_buf)
        actions_buf = torch.stack(actions_buf)
        logprobs_buf = torch.stack(logprobs_buf)
        rewards_buf = torch.stack(rewards_buf)
        dones_buf = torch.stack(dones_buf)
        values_buf = torch.stack(values_buf)
        
        # Compute advantages (GAE)
        with torch.no_grad():
            next_obs_t = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
            next_value = agent.get_value(next_obs_t).flatten()
            
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - torch.zeros(num_envs, device=device)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + values_buf
        
        # Flatten
        b_obs = obs_buf.reshape((-1,) + obs_buf.shape[2:])
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        
        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // 4
        
        # PPO update
        for epoch in range(update_epochs):
            b_inds = np.arange(batch_size)
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds].long()
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                ).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                
                # Total loss
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        
        if iteration % 5 == 0:
            print(f"  Step {global_step:,}/{total_steps:,} - Loss: {loss.item():.4f}")
    
    print(f"Training complete! Total steps: {global_step:,}")
    return agent


def collect_batch_gradient(env, agent, num_steps, device):
    """Collect gradient from a batch."""
    states = []
    actions = []
    rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        states.append(obs_t)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)
        actions.append(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().item())
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    # Compute gradient
    agent.zero_grad()
    
    states_t = torch.cat(states)
    actions_t = torch.cat(actions)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    # Normalize rewards
    if len(rewards_t) > 1:
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    
    # Forward pass
    _, logprobs, _, _ = agent.get_action_and_value(states_t, actions_t.long())
    
    # Policy gradient
    loss = -(logprobs * rewards_t).mean()
    loss.backward()
    
    # Extract actor gradients
    actor_grads = []
    for name, p in agent.named_parameters():
        if 'actor' in name and p.grad is not None:
            actor_grads.append(p.grad.flatten())
    
    if actor_grads:
        return torch.cat(actor_grads).detach().cpu().numpy()
    return None


def main():
    print("="*70)
    print("GRADIENT CLUSTERING WITH TRAINED MODEL")
    print("="*70)
    
    device = torch.device("cpu")
    
    # Create architecture
    variant = AG.AblationVariant(
        name='hetero', graph_type='hetero', hetero_base='sage',
        gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True,
    )
    
    agent = AG.AblationGinAgent(device, variant, hidden_dim=64, embedding_dim=32)
    
    # Create training environments
    num_train_envs = 3
    print(f"\nCreating {num_train_envs} training environments per domain...")
    train_envs_qf = [create_env("qf", seed=42+i) for i in range(num_train_envs)]
    train_envs_bn = [create_env("bn", seed=100+i) for i in range(num_train_envs)]
    
    # Initialize agent
    obs_init, _ = train_envs_qf[0].reset()
    obs_init_t = torch.tensor(obs_init, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _ = agent.get_action_and_value(obs_init_t)
    
    actor_params = sum(p.numel() for name, p in agent.named_parameters() if 'actor' in name)
    print(f"Actor parameters: {actor_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4)
    
    # Train on mixed data
    agent = train_mixed_ppo(
        agent, optimizer, 
        train_envs_qf, train_envs_bn, 
        device, 
        total_steps=20000,
        num_steps=64
    )
    
    # Close training envs
    for env in train_envs_qf + train_envs_bn:
        env.close()
    
    # Create fresh environments for gradient collection
    print("\n" + "="*70)
    print("COLLECTING GRADIENTS FROM TRAINED MODEL")
    print("="*70)
    
    num_collect_envs = 5
    collect_envs_qf = [create_env("qf", seed=200+i) for i in range(num_collect_envs)]
    collect_envs_bn = [create_env("bn", seed=300+i) for i in range(num_collect_envs)]
    
    num_batches_per_env = 8
    num_steps_per_batch = 50
    
    print(f"\nCollecting {num_collect_envs * num_batches_per_env * 2} batches...")
    
    gradients = []
    true_labels = []
    
    # Collect from QF
    for env in collect_envs_qf:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient(env, agent, num_steps_per_batch, device)
                if grad is not None:
                    gradients.append(grad)
                    true_labels.append(0)
            except:
                continue
    
    # Collect from BN
    for env in collect_envs_bn:
        for _ in range(num_batches_per_env):
            try:
                grad = collect_batch_gradient(env, agent, num_steps_per_batch, device)
                if grad is not None:
                    gradients.append(grad)
                    true_labels.append(1)
            except:
                continue
    
    for env in collect_envs_qf + collect_envs_bn:
        env.close()
    
    gradients = np.array(gradients)
    true_labels = np.array(true_labels)
    
    print(f"Collected {len(gradients)} gradient vectors")
    print(f"  QF: {sum(true_labels == 0)}, BN: {sum(true_labels == 1)}")
    
    # Normalize
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients_norm = gradients / (grad_norms + 1e-8)
    
    # PCA
    pca = PCA(n_components=min(30, len(gradients)-1))
    gradients_pca = pca.fit_transform(gradients_norm)
    print(f"PCA explained variance (top 10): {pca.explained_variance_ratio_[:10].sum():.1%}")
    
    # Cosine similarity
    cos_sim = cosine_similarity(gradients_norm)
    
    # Clustering
    print("\nClustering...")
    
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(gradients_pca)
    
    # Spectral on cosine
    affinity = (cos_sim + 1) / 2
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
    labels_sp = spectral.fit_predict(affinity)
    
    # Evaluate
    for name, labels in [("K-Means", labels_km), ("Spectral", labels_sp)]:
        sil = silhouette_score(gradients_pca, labels)
        ari = adjusted_rand_score(true_labels, labels)
        
        c0_qf = sum((labels == 0) & (true_labels == 0))
        c0_bn = sum((labels == 0) & (true_labels == 1))
        c1_qf = sum((labels == 1) & (true_labels == 0))
        c1_bn = sum((labels == 1) & (true_labels == 1))
        purity = max(c0_qf + c1_bn, c0_bn + c1_qf) / len(labels)
        
        print(f"  {name:10s}: Silhouette={sil:.3f}, ARI={ari:.3f}, Purity={purity:.1%}")
    
    # Use best
    best_labels = labels_sp if adjusted_rand_score(true_labels, labels_sp) > adjusted_rand_score(true_labels, labels_km) else labels_km
    best_name = "Spectral" if adjusted_rand_score(true_labels, labels_sp) > adjusted_rand_score(true_labels, labels_km) else "K-Means"
    
    # Final metrics
    c0_qf = sum((best_labels == 0) & (true_labels == 0))
    c0_bn = sum((best_labels == 0) & (true_labels == 1))
    c1_qf = sum((best_labels == 1) & (true_labels == 0))
    c1_bn = sum((best_labels == 1) & (true_labels == 1))
    
    if c0_qf + c1_bn > c0_bn + c1_qf:
        cluster_map = {0: 'QF', 1: 'BN'}
        purity_0 = c0_qf / (c0_qf + c0_bn) if (c0_qf + c0_bn) > 0 else 0
        purity_1 = c1_bn / (c1_qf + c1_bn) if (c1_qf + c1_bn) > 0 else 0
    else:
        cluster_map = {0: 'BN', 1: 'QF'}
        purity_0 = c0_bn / (c0_qf + c0_bn) if (c0_qf + c0_bn) > 0 else 0
        purity_1 = c1_qf / (c1_qf + c1_bn) if (c1_qf + c1_bn) > 0 else 0
    
    avg_purity = (purity_0 + purity_1) / 2
    final_ari = adjusted_rand_score(true_labels, best_labels)
    final_sil = silhouette_score(gradients_pca, best_labels)
    
    print(f"\nBest: {best_name}")
    print(f"  Purity: {avg_purity:.1%}")
    print(f"  ARI: {final_ari:.3f}")
    
    # t-SNE
    print("\nApplying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gradients)-1))
    gradients_tsne = tsne.fit_transform(gradients_pca)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # True labels
    ax = axes[0, 0]
    colors_true = ['#0077BB' if l == 0 else '#EE7733' for l in true_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_true, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('(a) Gradient Space (True Labels)\nBlue=QF, Orange=BN', fontweight='bold')
    
    # Discovered clusters
    ax = axes[0, 1]
    colors_cluster = ['#CC3311' if l == 0 else '#009988' for l in best_labels]
    ax.scatter(gradients_tsne[:, 0], gradients_tsne[:, 1], c=colors_cluster, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'(b) Discovered Clusters ({best_name})\nCluster 0={cluster_map[0]}, Cluster 1={cluster_map[1]}', fontweight='bold')
    
    # Cosine similarity heatmap
    ax = axes[1, 0]
    sort_idx = np.argsort(true_labels)
    sorted_sim = cos_sim[sort_idx][:, sort_idx]
    im = ax.imshow(sorted_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    n_qf = sum(true_labels == 0)
    ax.axhline(y=n_qf-0.5, color='black', linewidth=2)
    ax.axvline(x=n_qf-0.5, color='black', linewidth=2)
    ax.set_xlabel('Batch Index (sorted)')
    ax.set_ylabel('Batch Index (sorted)')
    ax.set_title('(c) Cosine Similarity Matrix\n(QF | BN)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
GRADIENT CLUSTERING (TRAINED MODEL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training:
  • 20,000 steps on mixed QF+BN data
  • PPO with your exact architecture

Gradient Collection:
  • {sum(true_labels == 0)} QF batches
  • {sum(true_labels == 1)} BN batches
  • {num_steps_per_batch} steps per batch

Clustering ({best_name}):
  • Silhouette: {final_sil:.3f}
  • Adjusted Rand Index: {final_ari:.3f}
  • Average Purity: {avg_purity:.1%}

Confusion Matrix:
                 True QF    True BN
  Cluster 0       {c0_qf:3d}        {c0_bn:3d}
  Cluster 1       {c1_qf:3d}        {c1_bn:3d}

CONCLUSION:
  {"✓ DOMAINS DISCOVERED!" if avg_purity > 0.75 else "○ Partial separation" if avg_purity > 0.6 else "✗ Weak separation"}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradient_clustering_trained.pdf', dpi=300, facecolor='white')
    plt.savefig(output_dir / 'gradient_clustering_trained.png', dpi=300, facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_dir / 'gradient_clustering_trained.png'}")


if __name__ == "__main__":
    main()
