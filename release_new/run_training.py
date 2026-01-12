#!/usr/bin/env python3
"""
Training Launcher - Run with YAML config or CLI arguments.

Usage:
    python run_training.py --config configs/train_longcp_specialist.yaml
    python run_training.py --config configs/train_longcp_specialist.yaml --total_timesteps 500000
"""
import sys, os, argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def load_config(path):
    import yaml
    with open(path) as f: return yaml.safe_load(f)

def set_host_specs_from_config(cfg):
    """Set HOST_SPECS_PATH env var BEFORE importing lib modules."""
    d = cfg.get('domain', {})
    host_specs = d.get('host_specs_file')
    if host_specs:
        os.environ['HOST_SPECS_PATH'] = os.path.abspath(host_specs)

def config_to_args(cfg):
    from cogito.gnn_deeprl_model.ablation_gnn_traj_main import Args
    a = Args()
    e = cfg.get('experiment', {})
    a.exp_name = e.get('name', a.exp_name)
    a.seed = e.get('seed', a.seed)
    a.output_dir = e.get('output_dir', a.output_dir)
    a.device = e.get('device', a.device)
    t = cfg.get('training', {})
    a.total_timesteps = t.get('total_timesteps', a.total_timesteps)
    a.learning_rate = t.get('learning_rate', a.learning_rate)
    a.num_envs = t.get('num_envs', a.num_envs)
    a.num_steps = t.get('num_steps', a.num_steps)
    a.gamma = t.get('gamma', a.gamma)
    a.gae_lambda = t.get('gae_lambda', a.gae_lambda)
    a.num_minibatches = t.get('num_minibatches', a.num_minibatches)
    a.update_epochs = t.get('update_epochs', a.update_epochs)
    a.clip_coef = t.get('clip_coef', a.clip_coef)
    a.ent_coef = t.get('ent_coef', a.ent_coef)
    a.vf_coef = t.get('vf_coef', a.vf_coef)
    a.max_grad_norm = t.get('max_grad_norm', a.max_grad_norm)
    a.anneal_lr = t.get('anneal_lr', a.anneal_lr)
    a.norm_adv = t.get('norm_adv', a.norm_adv)
    a.clip_vloss = t.get('clip_vloss', a.clip_vloss)
    ev = cfg.get('evaluation', {})
    a.test_every_iters = ev.get('test_every_iters', a.test_every_iters)
    a.robust_eval_alpha = ev.get('robust_eval_alpha', a.robust_eval_alpha)
    d = cfg.get('domain', {})
    a.longcp_config = d.get('longcp_config')
    a.wide_config = d.get('wide_config')
    s = cfg.get('seed_control', {})
    a.training_seed_mode = s.get('mode', a.training_seed_mode)
    a.train_seeds_file = s.get('seeds_file')
    v = cfg.get('variant', {})
    a.train_only_variant = v.get('name', a.train_only_variant)
    tr = cfg.get('trajectory', {})
    if hasattr(a, 'trajectory_enabled'):
        a.trajectory_enabled = tr.get('enabled', a.trajectory_enabled)
    if hasattr(a, 'trajectory_collect_every'):
        a.trajectory_collect_every = tr.get('collect_every', a.trajectory_collect_every)
    if hasattr(a, 'trajectory_method'):
        a.trajectory_method = tr.get('method', a.trajectory_method)
    l = cfg.get('logging', {})
    if hasattr(a, 'no_tensorboard'):
        a.no_tensorboard = not l.get('tensorboard', True)
    if hasattr(a, 'log_every'):
        a.log_every = l.get('log_every', a.log_every)
    return a

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config', '-c', type=str)
    p.add_argument('--help', '-h', action='store_true')
    known, rest = p.parse_known_args()
    if known.config:
        print(f"Loading: {known.config}")
        cfg = load_config(known.config)
        # Set host specs BEFORE importing lib modules
        set_host_specs_from_config(cfg)
        args = config_to_args(cfg)
        if rest:
            op = argparse.ArgumentParser()
            op.add_argument('--exp_name', type=str)
            op.add_argument('--seed', type=int)
            op.add_argument('--device', type=str)
            op.add_argument('--total_timesteps', type=int)
            op.add_argument('--num_envs', type=int)
            op.add_argument('--trajectory_enabled', action='store_true')
            ov, _ = op.parse_known_args(rest)
            for k,v in vars(ov).items():
                if v is not None: setattr(args, k, v)
        from cogito.gnn_deeprl_model.ablation_gnn_traj_main import main as train
        train(args)
    elif known.help:
        print(__doc__)
        print("\nConfigs:", *[f"  - {c}" for c in sorted(__import__('glob').glob("configs/*.yaml"))], sep='\n')
    else:
        import tyro
        from cogito.gnn_deeprl_model.ablation_gnn_traj_main import main as train, Args
        train(tyro.cli(Args))

if __name__ == "__main__": main()
