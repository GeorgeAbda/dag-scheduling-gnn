"""
Neural Network-Based Adversarial MDP Discovery

Strategy:
1. Train a neural network to predict gradient conflict from MDP parameters
2. Use gradient ascent on the NN input to find MDPs that maximize conflict
3. Much faster than evaluating actual gradients repeatedly

Architecture:
    MDP1 params (5) + MDP2 params (5) → NN → conflict score
    Then: ∇_{params} conflict_score to optimize params
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from scheduler.rl_model.ablation_gnn import AblationGinAgent, AblationVariant
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
from scheduler.dataset_generator.gen_dataset import DatasetArgs


class ConflictPredictor(nn.Module):
    """Neural network to predict gradient conflict from MDP parameters."""
    
    def __init__(self, input_dim=10, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1] like cosine similarity
        )
    
    def forward(self, x):
        """
        x: (batch, 10) tensor of MDP params
           [edge_prob1, min_tasks1, max_tasks1, min_len1, max_len1,
            edge_prob2, min_tasks2, max_tasks2, min_len2, max_len2]
        Returns: (batch, 1) predicted conflict (cosine similarity)
        """
        return self.net(x)


def normalize_params(params):
    """
    Normalize MDP parameters to [0, 1] for neural network input.
    
    params: dict or array [edge_prob, min_tasks, max_tasks, min_len, max_len]
    """
    if isinstance(params, dict):
        p = np.array([
            params['edge_prob'],
            params['min_tasks'],
            params['max_tasks'],
            params['min_length'],
            params['max_length']
        ])
    else:
        p = np.array(params)
    
    # Normalize each dimension
    normalized = np.zeros(5)
    normalized[0] = (p[0] - 0.01) / (0.99 - 0.01)  # edge_prob
    normalized[1] = (p[1] - 5) / (50 - 5)  # min_tasks
    normalized[2] = (p[2] - 5) / (50 - 5)  # max_tasks
    normalized[3] = (p[3] - 100) / (10000 - 100)  # min_length
    normalized[4] = (p[4] - 1000) / (200000 - 1000)  # max_length
    
    return normalized


def denormalize_params(normalized):
    """
    Denormalize from [0, 1] back to original ranges.
    """
    params = np.zeros(5)
    params[0] = normalized[0] * (0.99 - 0.01) + 0.01  # edge_prob
    params[1] = normalized[1] * (50 - 5) + 5  # min_tasks
    params[2] = normalized[2] * (50 - 5) + 5  # max_tasks
    params[3] = normalized[3] * (10000 - 100) + 100  # min_length
    params[4] = normalized[4] * (200000 - 1000) + 1000  # max_length
    
    # Ensure constraints
    params[1] = max(5, min(50, params[1]))
    params[2] = max(params[1], min(50, params[2]))
    params[3] = max(100, min(10000, params[3]))
    params[4] = max(params[3], min(200000, params[4]))
    
    return params


def create_mdp_from_params(params, host_specs_file):
    """Create MDP environment from parameters."""
    if isinstance(params, dict):
        edge_prob = params['edge_prob']
        min_tasks = int(params['min_tasks'])
        max_tasks = int(params['max_tasks'])
        min_length = int(params['min_length'])
        max_length = int(params['max_length'])
    else:
        edge_prob = float(params[0])
        min_tasks = int(params[1])
        max_tasks = int(params[2])
        min_length = int(params[3])
        max_length = int(params[4])
    
    style = "long_cp" if edge_prob > 0.5 else "wide"
    
    args = DatasetArgs(
        host_count=10,
        vm_count=10,
        workflow_count=1,
        style=style,
        gnp_p=edge_prob,
        gnp_min_n=min_tasks,
        gnp_max_n=max_tasks,
        min_task_length=min_length,
        max_task_length=max_length,
    )
    
    os.environ['HOST_SPECS_PATH'] = host_specs_file
    
    env = GinAgentWrapper(CloudSchedulingGymEnvironment(
        dataset_args=args, collect_timelines=False, compute_metrics=False
    ))
    
    return env, args


def compute_gradient(agent, env, num_steps=256, device=torch.device("cpu"), seed=None):
    """Compute policy gradient for given environment."""
    obs_list = []
    action_list = []
    rewards = []
    
    if seed is not None:
        obs, _ = env.reset(seed=int(seed))
    else:
        obs, _ = env.reset()
    
    for _ in range(num_steps):
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32).reshape(1, -1)).to(device)
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs_list.append(obs_tensor)
        action_list.append(action)
        
        obs, reward, terminated, truncated, info = env.step(int(action.item()))
        rewards.append(reward)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    # REINFORCE gradient
    agent.zero_grad()
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    loss = 0.0
    for obs_t, action_t, ret in zip(obs_list, action_list, returns):
        _, log_prob, _, _ = agent.get_action_and_value(obs_t, action_t)
        loss = loss - log_prob * ret
    
    loss = loss / len(obs_list)
    loss.backward()
    
    # Extract gradients
    grad_parts = []
    for p in agent.actor.parameters():
        if p.grad is not None:
            grad_parts.append(p.grad.view(-1).detach().clone())
    
    agent.zero_grad()
    
    if not grad_parts:
        return None
    
    return torch.cat(grad_parts).cpu().numpy()


def compute_true_conflict(agent, params1, params2, host_specs_file, device):
    """Compute actual gradient conflict (ground truth)."""
    try:
        env1, _ = create_mdp_from_params(params1, host_specs_file)
        env2, _ = create_mdp_from_params(params2, host_specs_file)
        
        g1 = compute_gradient(agent, env1, num_steps=128, device=device, seed=12345)
        g2 = compute_gradient(agent, env2, num_steps=128, device=device, seed=67890)
        
        env1.close()
        env2.close()
        
        if g1 is None or g2 is None:
            return 0.0
        
        cos_sim = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-9)
        return cos_sim
    
    except Exception as e:
        print(f"Error: {e}")
        return 0.0


def generate_training_data(agent, host_specs_file, device, num_samples=100):
    """
    Generate training data: random MDP pairs → conflict scores.
    """
    print(f"Generating {num_samples} training samples...")
    
    X = []  # Input: normalized MDP params
    y = []  # Output: conflict scores
    
    for i in tqdm(range(num_samples)):
        # Sample random MDP parameters
        params1 = {
            'edge_prob': np.random.uniform(0.01, 0.99),
            'min_tasks': np.random.randint(5, 30),
            'max_tasks': np.random.randint(10, 50),
            'min_length': np.random.randint(100, 5000),
            'max_length': np.random.randint(10000, 200000)
        }
        params1['max_tasks'] = max(params1['min_tasks'], params1['max_tasks'])
        params1['max_length'] = max(params1['min_length'], params1['max_length'])
        
        params2 = {
            'edge_prob': np.random.uniform(0.01, 0.99),
            'min_tasks': np.random.randint(5, 30),
            'max_tasks': np.random.randint(10, 50),
            'min_length': np.random.randint(100, 5000),
            'max_length': np.random.randint(10000, 200000)
        }
        params2['max_tasks'] = max(params2['min_tasks'], params2['max_tasks'])
        params2['max_length'] = max(params2['min_length'], params2['max_length'])
        
        # Compute true conflict
        conflict = compute_true_conflict(agent, params1, params2, host_specs_file, device)
        
        # Normalize and concatenate
        norm1 = normalize_params(params1)
        norm2 = normalize_params(params2)
        x = np.concatenate([norm1, norm2])
        
        X.append(x)
        y.append(conflict)
    
    return np.array(X), np.array(y)


def train_conflict_predictor(X, y, epochs=200, lr=0.001):
    """Train the conflict predictor network with regularization."""
    print(f"\nTraining conflict predictor on {len(X)} samples...")
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    model = ConflictPredictor(input_dim=10, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
    criterion = nn.MSELoss()
    
    model.train()
    best_loss = float('inf')
    patience = 0
    max_patience = 20
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Early stopping to prevent overfitting
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    print("  ✓ Training complete")
    return model


def optimize_mdp_pair(model, X_train, num_steps=500, lr=0.05, num_restarts=5):
    """
    Use gradient descent to find MDP pair that maximizes conflict.
    Multiple random restarts to avoid local minima.
    Constrain search to convex hull of training data.
    
    Optimize: min_{params} model(params)  (minimize cosine similarity = maximize conflict)
    Target: cosine similarity → -1 (opposing gradients)
    """
    print(f"\nOptimizing MDP pair to maximize conflict ({num_steps} steps, {num_restarts} restarts)...")
    print(f"  Constraining search to convex hull of {len(X_train)} training samples...")
    
    # Compute bounds from training data (convex hull approximation)
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    # Tighter bounds: mean ± 2*std, clipped to [min, max]
    lower_bound = np.maximum(X_min, X_mean - 2 * X_std)
    upper_bound = np.minimum(X_max, X_mean + 2 * X_std)
    
    print(f"  Edge prob bounds: MDP1=[{lower_bound[0]:.3f}, {upper_bound[0]:.3f}], "
          f"MDP2=[{lower_bound[5]:.3f}, {upper_bound[5]:.3f}]")
    
    global_best_conflict = 1.0
    global_best_params = None
    
    for restart in range(num_restarts):
        # Initialize within training data bounds
        params = torch.zeros(10, requires_grad=True)
        with torch.no_grad():
            # Sample uniformly within bounds
            for i in range(10):
                params[i] = lower_bound[i] + (upper_bound[i] - lower_bound[i]) * torch.rand(1)
        
        optimizer = optim.Adam([params], lr=lr)
        
        model.eval()
        best_conflict = 1.0
        best_params = None
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Predict conflict (cosine similarity)
            pred_conflict = model(params.unsqueeze(0))
            
            # MINIMIZE cosine similarity (make it more negative = more conflict)
            loss = pred_conflict.mean()
            loss.backward()
            
            optimizer.step()
            
            # Project back to training data bounds (convex hull constraint)
            with torch.no_grad():
                for i in range(10):
                    params[i] = torch.clamp(params[i], lower_bound[i], upper_bound[i])
            
            conflict_val = pred_conflict.item()
            # Track best (most negative = most conflict)
            if conflict_val < best_conflict:
                best_conflict = conflict_val
                best_params = params.detach().clone()
        
        print(f"  Restart {restart+1}/{num_restarts}: Best conflict = {best_conflict:.4f}")
        
        if best_conflict < global_best_conflict:
            global_best_conflict = best_conflict
            global_best_params = best_params
    
    print(f"  ✓ Optimization complete, best predicted conflict: {global_best_conflict:.4f}")
    print(f"    (More negative = more conflict)")
    
    # Denormalize
    params_np = global_best_params.cpu().numpy()
    params1 = denormalize_params(params_np[:5])
    params2 = denormalize_params(params_np[5:])
    
    return params1, params2, global_best_conflict


def discover_adversarial_mdp_pair_neural(
    host_specs_file: str,
    output_dir: str = "logs/mdp_discovery",
    num_training_samples: int = 50,
    training_epochs: int = 200,
    optimization_steps: int = 500
):
    """
    Discover adversarial MDP pair using neural network + gradient ascent.
    """
    print("="*70)
    print("ADVERSARIAL MDP DISCOVERY (Neural Network)")
    print("="*70)
    print("Strategy:")
    print("  1. Generate training data: random MDP pairs → conflict scores")
    print("  2. Train neural network to predict conflict from MDP params")
    print("  3. Use gradient ascent to find params that maximize conflict")
    print()
    
    device = torch.device("cpu")
    
    # Global seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Initialize agent
    print("[1/5] Initializing agent...")
    variant = AblationVariant(
        name="hetero", graph_type="hetero", gin_num_layers=2,
        use_batchnorm=True, use_task_dependencies=True,
        use_actor_global_embedding=True, mlp_only=False, gat_heads=4,
    )
    agent = AblationGinAgent(device=device, variant=variant, hidden_dim=128, embedding_dim=64)
    print("  ✓ Agent initialized")
    
    # Generate training data
    print(f"\n[2/5] Generating training data ({num_training_samples} samples)...")
    X, y = generate_training_data(agent, host_specs_file, device, num_samples=num_training_samples)
    print(f"  ✓ Generated {len(X)} samples")
    print(f"  Conflict range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Train predictor
    print(f"\n[3/5] Training conflict predictor...")
    model = train_conflict_predictor(X, y, epochs=training_epochs)
    
    # Optimize MDP pair (constrained to training data hull)
    print(f"\n[4/5] Optimizing MDP pair...")
    params1, params2, predicted_conflict = optimize_mdp_pair(model, X, num_steps=optimization_steps)
    
    # Verify with true conflict
    print(f"\n[5/5] Verifying with true gradient conflict...")
    true_conflict = compute_true_conflict(agent, params1, params2, host_specs_file, device)
    print(f"  Predicted conflict: {predicted_conflict:.4f}")
    print(f"  True conflict:      {true_conflict:.4f}")
    print(f"  Error:              {abs(predicted_conflict - true_conflict):.4f}")
    
    # Create configs
    _, args1 = create_mdp_from_params(params1, host_specs_file)
    _, args2 = create_mdp_from_params(params2, host_specs_file)
    
    config1 = {
        "training_seeds": [101001],
        "all_seeds": [101001],
        "dataset": {
            "style": args1.style,
            "edge_probability": float(args1.gnp_p),
            "min_tasks": int(args1.gnp_min_n),
            "max_tasks": int(args1.gnp_max_n),
            "hosts": 10,
            "vms": 10,
            "workflow_count": 1,
            "task_length": {
                "distribution": "normal",
                "min": int(args1.min_task_length),
                "max": int(args1.max_task_length)
            }
        },
        "comment": f"Discovered via neural optimization (true_conflict={true_conflict:.4f})"
    }
    
    config2 = {
        "training_seeds": [202002],
        "all_seeds": [202002],
        "dataset": {
            "style": args2.style,
            "edge_probability": float(args2.gnp_p),
            "min_tasks": int(args2.gnp_min_n),
            "max_tasks": int(args2.gnp_max_n),
            "hosts": 10,
            "vms": 10,
            "workflow_count": 1,
            "task_length": {
                "distribution": "normal",
                "min": int(args2.min_task_length),
                "max": int(args2.max_task_length)
            }
        },
        "comment": f"Discovered via neural optimization (true_conflict={true_conflict:.4f})"
    }
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config1_file = output_path / "discovered_mdp1.json"
    config2_file = output_path / "discovered_mdp2.json"
    
    with open(config1_file, 'w') as f:
        json.dump(config1, f, indent=2)
    
    with open(config2_file, 'w') as f:
        json.dump(config2, f, indent=2)
    
    print(f"\n  ✓ Saved configs to {output_dir}")
    
    print("\n" + "="*70)
    print("DISCOVERED MDP CONFIGURATIONS")
    print("="*70)
    
    print(f"\nMDP 1 ({args1.style}):")
    print(f"  Edge Probability: {args1.gnp_p:.3f}")
    print(f"  Tasks: {args1.gnp_min_n}-{args1.gnp_max_n}")
    print(f"  Task Length: {args1.min_task_length}-{args1.max_task_length}")
    
    print(f"\nMDP 2 ({args2.style}):")
    print(f"  Edge Probability: {args2.gnp_p:.3f}")
    print(f"  Tasks: {args2.gnp_min_n}-{args2.gnp_max_n}")
    print(f"  Task Length: {args2.min_task_length}-{args2.max_task_length}")
    
    print(f"\nGradient Conflict: {true_conflict:.4f}")
    
    if true_conflict < -0.8:
        print("  → SEVERE CONFLICT: Separate policies required")
    elif true_conflict < -0.3:
        print("  → MODERATE CONFLICT: Task-conditioned policy recommended")
    else:
        print("  → LOW CONFLICT: Single policy may work")
    
    return config1, config2, true_conflict


if __name__ == "__main__":
    import sys
    
    host_specs = sys.argv[1] if len(sys.argv) > 1 else "data/host_specs.json"
    
    config1, config2, conflict = discover_adversarial_mdp_pair_neural(
        host_specs_file=host_specs,
        output_dir="logs/mdp_discovery",
        num_training_samples=10000,  # 1000 samples for better coverage
        training_epochs=200,         # Train predictor
        optimization_steps=500       # Gradient ascent steps
    )
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Use discovered configs:")
    print("   python discover_domains_improved.py \\")
    print("     logs/mdp_discovery/discovered_mdp1.json \\")
    print("     logs/mdp_discovery/discovered_mdp2.json \\")
    print(f"     {host_specs}")
