"""
Hetero-GNN Gradient Clustering on Scheduler MDPs
Generates domain-variant MDPs, collects real gradients with the hetero agent,
clusters with SW distance, and saves metrics + a figure.
"""
import os, sys, json, warnings
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

proj = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if proj not in sys.path: sys.path.insert(0, proj)

from scheduler.dataset_generator.gen_dataset import DatasetArgs
from scheduler.rl_model.core.env.gym_env import CloudSchedulingGymEnvironment
from scheduler.rl_model.agents.gin_agent.wrapper import GinAgentWrapper
import scheduler.rl_model.ablation_gnn as AG

plt.rcParams['figure.facecolor'] = 'white'

@dataclass
class DomainConfig:
    name: str; style: str; gnp_p_range: Tuple[float,float]; task_length_range: Tuple[int,int]
    workflow_count: int; gnp_min_n: int; gnp_max_n: int; label: int; description: str

def get_domain_configs() -> Dict[str, DomainConfig]:
    return {
        'wide_parallel': DomainConfig('wide_parallel','wide',(0.02,0.10),(500,50000),10,15,30,0,'Wide DAGs'),
        'long_sequential': DomainConfig('long_sequential','long_cp',(0.75,0.90),(500,50000),10,15,30,1,'Long CP'),
        'short_tasks': DomainConfig('short_tasks','generic',(0.20,0.40),(100,5000),10,20,40,2,'Short tasks'),
        'long_tasks': DomainConfig('long_tasks','generic',(0.20,0.40),(50000,200000),10,8,15,3,'Long tasks'),
    }

def make_ds_args(cfg: DomainConfig, seed: int) -> DatasetArgs:
    return DatasetArgs(seed=seed, style=cfg.style, host_count=10, vm_count=10, max_memory_gb=128,
        min_cpu_speed=500, max_cpu_speed=5000, workflow_count=cfg.workflow_count, dag_method='gnp',
        gnp_min_n=cfg.gnp_min_n, gnp_max_n=cfg.gnp_max_n, task_length_dist='normal',
        min_task_length=cfg.task_length_range[0], max_task_length=cfg.task_length_range[1],
        task_arrival='static', arrival_rate=3.0, gnp_p=float(np.random.uniform(*cfg.gnp_p_range)), req_divisor=None)

def xavier_reset(m: nn.Module):
    for l in m.modules():
        if isinstance(l, nn.Linear):
            nn.init.xavier_uniform_(l.weight); 
            if l.bias is not None: nn.init.zeros_(l.bias)

def collect_episode_grad(agent: AG.AblationGinAgent, env: GinAgentWrapper, max_steps: int = 256, gamma: float = 0.99) -> Optional[np.ndarray]:
    obs, _ = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logps: list[torch.Tensor] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        act, logp, _, _ = agent.get_action_and_value(obs_t)
        ob2, r, term, trunc, _ = env.step(int(act.item()))
        logps.append(logp.squeeze())
        rewards.append(float(r))
        obs_t = torch.tensor(ob2, dtype=torch.float32).unsqueeze(0)
        if term or trunc:
            break
    # Compute returns-to-go and standardize for baseline
    if not rewards:
        return None
    G = []
    ret = 0.0
    for r in reversed(rewards):
        ret = float(r) + gamma * ret
        G.append(ret)
    G = list(reversed(G))
    G_arr = np.array(G, dtype=np.float32)
    adv = (G_arr - G_arr.mean()) / (G_arr.std() + 1e-8)
    # Policy loss
    pol_loss = 0.0
    for lp, a in zip(logps, adv):
        pol_loss = pol_loss - lp * float(a)
    agent.zero_grad(); pol_loss.backward()
    g = [p.grad.detach().cpu().numpy().ravel() for p in agent.actor.parameters() if p.grad is not None]
    return None if not g else np.concatenate(g)

def sw_dist(X: np.ndarray, Y: np.ndarray, n_proj: int = 100) -> float:
    if X.ndim == 1:
        X = X[None, :]
    if Y.ndim == 1:
        Y = Y[None, :]
    d = X.shape[1]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    P = np.random.randn(n_proj, d)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return float(np.mean([wasserstein_distance(Xn @ p, Yn @ p) for p in P]))

def run():
    out_dir = 'results/scheduler_domains_hetero'; os.makedirs(out_dir, exist_ok=True)
    # Emphasize makespan structure and reduce noisy scoring
    os.environ['GIN_ENERGY_WEIGHT'] = os.environ.get('GIN_ENERGY_WEIGHT', '0.0')
    os.environ['GIN_MAKESPAN_WEIGHT'] = os.environ.get('GIN_MAKESPAN_WEIGHT', '1.0')
    os.environ['GIN_STEP_MAKESPAN'] = os.environ.get('GIN_STEP_MAKESPAN', '1')
    os.environ['VALID_ONLY_SCORING'] = os.environ.get('VALID_ONLY_SCORING', '1')
    seed = 7; torch.manual_seed(seed); np.random.seed(seed)
    n_inst = 6; n_grad = 20
    cfgs = get_domain_configs(); K = len(cfgs)
    var = AG.AblationVariant(name='hetero', graph_type='hetero', hetero_base='sage', gin_num_layers=2, use_batchnorm=True, use_task_dependencies=True, use_actor_global_embedding=True)
    agent = AG.AblationGinAgent(torch.device('cpu'), var, hidden_dim=64, embedding_dim=32)

    names: List[str] = []; labels: List[int] = []; grads: Dict[str,np.ndarray] = {}
    print('='*60,'\nCollecting gradients (hetero)\n','='*60)
    for nm, cfg in cfgs.items():
        print(f"Domain: {nm}")
        for i in range(n_inst):
            inst = f"{nm}_{i}"; names.append(inst); labels.append(cfg.label)
            ds_args = make_ds_args(cfg, seed + hash(inst)%1_000_000)
            env = GinAgentWrapper(CloudSchedulingGymEnvironment(dataset_args=ds_args, fixed_env_seed=True, dataset_episode_mode='single'))
            # Fixed actor initialization per instance to capture environment-specific gradient geometry
            torch.manual_seed(ds_args.seed); np.random.seed(ds_args.seed % 2**31)
            xavier_reset(agent)
            G = []
            for k in range(n_grad):
                g = collect_episode_grad(agent, env)
                if g is not None: G.append(g)
            grads[inst] = np.array(G); print(f"  {inst}: {len(G)} grads")

    labels = np.array(labels); n = len(names)
    D = np.zeros((n,n), float); 
    total_pairs = n*(n-1)//2
    with tqdm(total=total_pairs, desc='SW pairs') as pbar:
        for i in range(n):
            for j in range(i+1,n):
                d = sw_dist(grads[names[i]], grads[names[j]], n_proj=256); D[i,j]=D[j,i]=d; pbar.update(1)

    S = 1 - D/(D.max()+1e-8); pred = SpectralClustering(n_clusters=K, affinity='precomputed', random_state=seed).fit_predict(S)
    ari = adjusted_rand_score(labels, pred); nmi = normalized_mutual_info_score(labels, pred)
    try: sil = silhouette_score(D, pred, metric='precomputed')
    except: sil = 0.0
    purity = 100.0*sum(np.bincount(labels[pred==c]).max() for c in np.unique(pred))/len(labels)
    w, c = [], []
    for i in range(n):
        for j in range(i+1,n):
            (w if labels[i]==labels[j] else c).append(D[i,j])
    sep = float(np.mean(c)/(np.mean(w)+1e-8))
    print(f"Metrics: ARI={ari:.4f} NMI={nmi:.4f} Purity={purity:.1f}% Sil={sil:.3f} Sep={sep:.2f}")

    # Figure
    fig = plt.figure(figsize=(16,10))
    order = np.argsort(labels)
    ax1 = fig.add_subplot(2,3,1); sns.heatmap(D[np.ix_(order,order)], cmap='viridis_r', ax=ax1, cbar_kws={'label':'SW'})
    ax1.set_title('(a) SW Distance (sorted)')
    bnds = np.cumsum([n_inst]*K)[:-1]
    for b in bnds: ax1.axhline(b, color='w'); ax1.axvline(b, color='w')
    ax2 = fig.add_subplot(2,3,2)
    mean_g = np.array([grads[nm].mean(axis=0) for nm in names])
    pc = PCA(n_components=2, random_state=42).fit_transform(mean_g)
    cols = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']; keys=list(cfgs.keys())
    for lb in sorted(set(labels)):
        m = labels==lb; ax2.scatter(pc[m,0], pc[m,1], c=cols[lb], s=70, edgecolors='k', label=keys[lb])
    ax2.set_title('(b) True labels'); ax2.legend(fontsize=7)
    ax3 = fig.add_subplot(2,3,3)
    for c_id in np.unique(pred):
        m = pred==c_id; ax3.scatter(pc[m,0], pc[m,1], c=[plt.cm.tab10(c_id)], s=70, edgecolors='k', label=f'C{c_id}')
    ax3.set_title(f'(c) Pred clusters (Purity {purity:.1f}%)'); ax3.legend(fontsize=7)
    ax4 = fig.add_subplot(2,3,4)
    vals=[ari,nmi,purity/100,sep/5]; bars=ax4.bar(['ARI','NMI','Pur/100','Sep/5'], vals, color=plt.cm.viridis(np.linspace(0.2,0.8,4)))
    for b,v in zip(bars,vals): ax4.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=8)
    ax4.set_ylim(0,1.2); ax4.set_title('(d) Metrics')
    ax5 = fig.add_subplot(2,3,5); ax5.axis('off')
    ax5.text(0.0,1.0, f'Hetero-GNN domain clustering\nInst/domain {n_inst}  |  Grad/inst {n_grad}\nARI {ari:.3f}  NMI {nmi:.3f}  Pur {purity:.1f}%  Sep {sep:.2f}', va='top', family='monospace')
    plt.tight_layout(); fig_path=f'{out_dir}/hetero_domain_clustering.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white'); print('Saved:', fig_path)
    os.makedirs('paper/figures', exist_ok=True); plt.savefig('paper/figures/scheduler_domains_hetero.png', dpi=300, bbox_inches='tight', facecolor='white')

    with open(f'{out_dir}/results.json','w') as f:
        json.dump({'ari':ari,'nmi':nmi,'purity':purity,'silhouette':sil,'separability':sep,'n_instances':n,'n_clusters':K,
                   'domains':{k:asdict(v) for k,v in cfgs.items()}}, f, indent=2)

if __name__=='__main__':
    run()
