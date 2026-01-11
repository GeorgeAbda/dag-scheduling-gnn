# Related Work Search Results: Gradient-Based Domain Detection in MORL

## Summary of Search

This document summarizes the literature search for work related to:
- Gradient conflict in multi-task/multi-objective learning
- Domain/task clustering using gradient similarity
- Optimal transport for comparing distributions
- Pareto front identification in MORL

---

## Key Papers Found

### 1. Gradient Conflict in Multi-Task Learning

#### **PCGrad: Gradient Surgery for Multi-Task Learning** (Yu et al., NeurIPS 2020)
- **URL**: https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
- **Key Idea**: When gradients from different tasks conflict (negative cosine similarity), project each gradient onto the normal plane of the other
- **Relevance**: Directly addresses gradient conflict, but focuses on *resolving* conflict rather than *using* it for clustering
- **Gap**: Does not use conflict patterns as task signatures

#### **CAGrad: Conflict-Averse Gradient Descent** (Liu et al., NeurIPS 2021)
- **URL**: https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf
- **Key Idea**: Find update direction that minimizes worst-case conflict with any task
- **Relevance**: Formalizes gradient conflict mathematically
- **Gap**: Focuses on optimization, not domain identification

#### **Nash-MTL: Multi-Task Learning as a Bargaining Game** (Navon et al., ICML 2022)
- **URL**: https://arxiv.org/abs/2202.01017
- **Key Idea**: View gradient combination as Nash bargaining problem; tasks "negotiate" for update direction
- **Relevance**: Game-theoretic view of gradient conflict; Pareto optimality in gradient space
- **Gap**: Does not cluster tasks by gradient patterns

---

### 2. Task Clustering Using Gradients

#### **Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity** (KDD 2024)
- **URL**: https://arxiv.org/html/2409.06091v1
- **Key Idea**: 
  - Use gradients as features to estimate task affinity
  - Linearize fine-tuned models around pre-trained weights
  - Task affinity ≈ gradient inner product
  - Cluster tasks using affinity matrix
- **Relevance**: **MOST DIRECTLY RELATED** - uses gradients to cluster tasks!
- **Method**: 
  1. Compute gradients g_i for each task
  2. Estimate affinity as ⟨g_i, g_j⟩
  3. Cluster using semidefinite programming
- **Gap**: Supervised learning only; does not consider distribution of gradients (just mean)

#### **Similarity of Neural Networks with Gradients** (2020)
- **URL**: https://arxiv.org/abs/2003.11498
- **Key Idea**: Define neural network similarity using gradient information
- **Relevance**: Theoretical foundation for gradient-based similarity

---

### 3. Multi-Objective RL and Pareto Fronts

#### **Pareto Set Learning for Multi-Objective RL** (2025)
- **URL**: https://arxiv.org/html/2501.06773v2
- **Key Idea**: Learn entire Pareto front as a continuous manifold
- **Relevance**: Addresses Pareto front structure in MORL

#### **Prediction-Guided MORL (PGMORL)**
- **URL**: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
- **Key Idea**: Use prediction to guide Pareto front discovery
- **Relevance**: Efficient Pareto front estimation

#### **C-MORL: Efficient Discovery of Pareto Front** (2024)
- **URL**: https://arxiv.org/abs/2410.02236
- **Key Idea**: Constraint-based approach to MORL
- **Relevance**: Addresses preference-conditioned policies

---

### 4. Domain Shift and Environment Detection

#### **Domain Shifts in Reinforcement Learning: Identifying Disturbances**
- **URL**: https://ceur-ws.org/Vol-2916/paper_11.pdf
- **Key Idea**: Detect when environment dynamics change
- **Relevance**: Domain detection in RL, but not gradient-based

#### **Hidden Parameter MDPs: A Semiparametric Approach**
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5466173/
- **Key Idea**: MDPs with hidden parameters that affect dynamics
- **Relevance**: Formalizes the "same reward, different dynamics" setting

#### **Block Contextual MDPs for Continual Learning**
- **URL**: https://www.researchgate.net/publication/355235631
- **Key Idea**: Non-stationary MDPs with context-dependent dynamics
- **Relevance**: Theoretical framework for MDP distributions

---

### 5. Continual Learning and Gradient Memory

#### **Gradient Episodic Memory (GEM)** (Lopez-Paz & Ranzato, NeurIPS 2017)
- **URL**: https://arxiv.org/pdf/1706.08840
- **Key Idea**: Store gradients from past tasks; constrain updates to not conflict
- **Relevance**: Uses gradient directions to identify task interference
- **Gap**: Focuses on preventing forgetting, not clustering

---

### 6. Optimal Transport for Distribution Comparison

#### **Optimal Transport and Wasserstein Distance** (CMU Tutorial)
- **URL**: https://www.stat.cmu.edu/~larry/=sml/Opt.pdf
- **Key Idea**: Wasserstein distance as principled way to compare distributions
- **Relevance**: Theoretical foundation for using OT on gradient distributions

#### **POT: Python Optimal Transport**
- **URL**: https://pythonot.github.io/quickstart.html
- **Key Idea**: Practical implementation of OT distances
- **Relevance**: Tool for computing Sliced Wasserstein distance

---

## Identified Research Gaps

### Gap 1: Gradient Distributions, Not Just Means
- Existing work (PCGrad, CAGrad, Nash-MTL) uses **mean gradients**
- Our approach: Compare **distributions** of gradients using OT
- Why it matters: High variance in MORL means mean is unreliable

### Gap 2: Domain Detection via Gradient Signatures
- Existing work resolves gradient conflict
- Our approach: Use conflict patterns as **domain fingerprints**
- Why it matters: Different Pareto fronts → different gradient distributions

### Gap 3: MORL-Specific Gradient Conflict
- Existing MTL work assumes different tasks
- Our setting: **Same reward function**, different dynamics
- Why it matters: Gradient conflict arises from Pareto structure, not task difference

### Gap 4: Unsupervised Domain Discovery
- Existing work assumes known task/domain labels
- Our approach: **Discover** domains from gradient patterns
- Why it matters: In practice, domain labels may be unknown

### Gap 5: OT for Gradient Comparison in RL
- OT used for distribution comparison in ML
- Not applied to policy gradient distributions
- Our contribution: First use of Wasserstein distance for domain clustering in MORL

---

## Proposed Formalization

### Problem Setting
- Distribution of MDPs: {M_1, ..., M_K} with same reward R but different dynamics P_k
- Single policy π_θ trained on mixture
- Preference weight α fixed

### Key Observation
For trajectory τ from MDP M_k:
```
g(τ; M_k) = ∇_θ J_α(θ; τ)
```
Different MDPs produce different gradient distributions:
```
μ_k = Distribution of g(τ; M_k) for τ ~ π_θ in M_k
```

### Domain Detection Problem
Given: Unlabeled trajectories {τ_1, ..., τ_N} from unknown MDPs
Goal: Cluster trajectories by source MDP using gradient signatures

### Proposed Method
1. Compute gradient g(τ_i) for each trajectory
2. Estimate pairwise distances using Sliced Wasserstein:
   ```
   d(i, j) = SW_2(μ_i, μ_j)
   ```
3. Cluster using spectral clustering on similarity matrix

### Theoretical Justification
- Different Pareto fronts → different optimal policies
- Different optimal policies → different gradient directions
- Gradient distribution encodes Pareto structure

---

## Key References to Cite

1. **Yu et al. (2020)** - PCGrad: Gradient Surgery for Multi-Task Learning
2. **Liu et al. (2021)** - CAGrad: Conflict-Averse Gradient Descent
3. **Navon et al. (2022)** - Nash-MTL: Multi-Task Learning as Bargaining
4. **Sener & Koltun (2018)** - MGDA: Multi-Task Learning as Multi-Objective Optimization
5. **Lopez-Paz & Ranzato (2017)** - Gradient Episodic Memory
6. **KDD 2024** - Gradient-based Task Affinity Estimation
7. **Hayes et al. (2022)** - MORL Survey
8. **Hallak et al. (2015)** - Contextual MDPs

---

## Novelty Statement

**Our contribution**: We propose using Optimal Transport (specifically Sliced Wasserstein distance) to compare gradient distributions across domains in multi-objective RL. Unlike existing work that:
- Resolves gradient conflict (PCGrad, CAGrad)
- Uses mean gradients for task affinity (KDD 2024)
- Assumes known task labels (Nash-MTL)

We:
- Use gradient conflict as a **domain signature**
- Compare full **distributions** via OT
- Perform **unsupervised** domain discovery
- Handle the MORL-specific case of same reward, different dynamics
