import torch
import numpy as np
try:
    import matplotlib.pyplot as plt  # optional, not required for plotly-based figures
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
from cogito.gnn_deeprl_model.ablation_gnn import AblationGinAgent
try:
    import seaborn as sns  # optional
except Exception:
    sns = None
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except (ImportError, AttributeError):
    UMAP_AVAILABLE = False
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import plotly.express as px  # optional; imports pandas/pyarrow
    PX_AVAILABLE = True
except Exception:
    px = None
    PX_AVAILABLE = False
from typing import Dict, List, Tuple, Any
import pandas as pd
from pathlib import Path

class GINInterpreter:
    """
    Comprehensive interpretability framework for GIN agent in cloud scheduling.
    """
    
    def __init__(self, gin_agent, device='cpu'):
        self.gin_agent = gin_agent
        self.device = device
        self.embedding_history = []
        self.feature_history = []
        self.action_history = []
        
    def extract_embeddings_and_features(self, obs_tensor):
        """Extract all embeddings and features from a single observation."""
        self.gin_agent.eval()
        
        with torch.no_grad():
            # Decode observation
            decoded_obs = self.gin_agent.mapper.unmap(obs_tensor.squeeze())
            
            # Get embeddings from base network
            node_embeddings, edge_embeddings, graph_embedding = self.gin_agent.actor.network(decoded_obs)
            
            # Extract features
            num_tasks = decoded_obs.task_state_scheduled.shape[0]
            num_vms = decoded_obs.vm_completion_time.shape[0]
            
            task_embeddings = node_embeddings[:num_tasks]
            vm_embeddings = node_embeddings[num_tasks:]
            
            # Get raw features
            task_features = self._extract_task_features(decoded_obs)
            vm_features = self._extract_vm_features(decoded_obs)
            
            # Get action scores
            action_scores = self.gin_agent.actor(decoded_obs)
            
            return {
                'task_embeddings': task_embeddings.cpu().numpy(),
                'vm_embeddings': vm_embeddings.cpu().numpy(),
                'edge_embeddings': edge_embeddings.cpu().numpy(),
                'graph_embedding': graph_embedding.cpu().numpy(),
                'task_features': task_features,
                'vm_features': vm_features,
                'action_scores': action_scores.cpu().numpy(),
                'compatibilities': decoded_obs.compatibilities.cpu().numpy(),
                'num_tasks': num_tasks,
                'num_vms': num_vms
            }
    
    def _extract_task_features(self, decoded_obs):
        """Extract interpretable task features."""
        return {
            'scheduled': decoded_obs.task_state_scheduled.cpu().numpy(),
            'ready': decoded_obs.task_state_ready.cpu().numpy(),
            'length': decoded_obs.task_length.cpu().numpy(),
            'completion_time': decoded_obs.task_completion_time.cpu().numpy(),
            'memory_req': decoded_obs.task_memory_req_mb.cpu().numpy(),
            'cpu_req': decoded_obs.task_cpu_req_cores.cpu().numpy()
        }
    
    def _extract_vm_features(self, decoded_obs):
        """Extract interpretable VM features."""
        return {
            'completion_time': decoded_obs.vm_completion_time.cpu().numpy(),
            'speed': decoded_obs.vm_speed.cpu().numpy(),
            'energy_rate': decoded_obs.vm_energy_rate.cpu().numpy(),
            'memory_total': decoded_obs.vm_memory_mb.cpu().numpy(),
            'memory_available': decoded_obs.vm_available_memory_mb.cpu().numpy(),
            'memory_used_fraction': decoded_obs.vm_used_memory_fraction.cpu().numpy(),
            'active_tasks': decoded_obs.vm_active_tasks_count.cpu().numpy(),
            'cpu_cores': decoded_obs.vm_cpu_cores.cpu().numpy(),
            'cpu_available': decoded_obs.vm_available_cpu_cores.cpu().numpy(),
            'cpu_used_fraction': decoded_obs.vm_used_cpu_fraction_cores.cpu().numpy()
        }

    def create_task_vm_labels(self, task_features, vm_features, method='characteristics'):
        """
        Create labels for task-VM pairs based on their characteristics.
        
        Args:
            method: 'characteristics', 'kmeans', 'performance', 'resource_match'
        """
        if method == 'characteristics':
            return self._label_by_characteristics(task_features, vm_features)
        elif method == 'kmeans':
            return self._label_by_kmeans(task_features, vm_features)
        elif method == 'performance':
            return self._label_by_performance(task_features, vm_features)
        elif method == 'resource_match':
            return self._label_by_resource_match(task_features, vm_features)
    
    def _label_by_characteristics(self, task_features, vm_features):
        """Label based on task and VM characteristics."""
        labels = {}
        
        # Task categories
        task_length_percentiles = np.percentile(task_features['length'], [33, 67])
        task_memory_percentiles = np.percentile(task_features['memory_req'], [33, 67])
        
        task_labels = []
        for i in range(len(task_features['length'])):
            length_cat = 'short' if task_features['length'][i] < task_length_percentiles[0] else \
                        'medium' if task_features['length'][i] < task_length_percentiles[1] else 'long'
            memory_cat = 'low_mem' if task_features['memory_req'][i] < task_memory_percentiles[0] else \
                        'med_mem' if task_features['memory_req'][i] < task_memory_percentiles[1] else 'high_mem'
            task_labels.append(f"{length_cat}_{memory_cat}")
        
        # VM categories
        vm_speed_percentiles = np.percentile(vm_features['speed'], [33, 67])
        vm_memory_percentiles = np.percentile(vm_features['memory_total'], [33, 67])
        
        vm_labels = []
        for i in range(len(vm_features['speed'])):
            speed_cat = 'slow' if vm_features['speed'][i] < vm_speed_percentiles[0] else \
                       'medium' if vm_features['speed'][i] < vm_speed_percentiles[1] else 'fast'
            memory_cat = 'small' if vm_features['memory_total'][i] < vm_memory_percentiles[0] else \
                        'medium' if vm_features['memory_total'][i] < vm_memory_percentiles[1] else 'large'
            vm_labels.append(f"{speed_cat}_{memory_cat}")
        
        return {
            'task_labels': task_labels,
            'vm_labels': vm_labels,
            'task_categories': list(set(task_labels)),
            'vm_categories': list(set(vm_labels))
        }
    
    def _label_by_kmeans(self, task_features, vm_features, n_clusters=5):
        """Label using K-means clustering on features."""
        # Task clustering
        task_feature_matrix = np.column_stack([
            task_features['length'],
            task_features['memory_req'],
            task_features['cpu_req']
        ])
        task_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        task_cluster_labels = task_kmeans.fit_predict(task_feature_matrix)
        
        # VM clustering
        vm_feature_matrix = np.column_stack([
            vm_features['speed'],
            vm_features['memory_total'],
            vm_features['cpu_cores'],
            vm_features['energy_rate']
        ])
        vm_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        vm_cluster_labels = vm_kmeans.fit_predict(vm_feature_matrix)
        
        return {
            'task_labels': [f"task_cluster_{i}" for i in task_cluster_labels],
            'vm_labels': [f"vm_cluster_{i}" for i in vm_cluster_labels],
            'task_categories': [f"task_cluster_{i}" for i in range(n_clusters)],
            'vm_categories': [f"vm_cluster_{i}" for i in range(n_clusters)]
        }
    
    def _label_by_resource_match(self, task_features, vm_features):
        """Label based on resource requirement vs capacity match."""
        labels = {}
        
        task_labels = []
        for i in range(len(task_features['memory_req'])):
            # Categorize by resource intensity
            mem_intensity = task_features['memory_req'][i] / np.mean(task_features['memory_req'])
            cpu_intensity = task_features['cpu_req'][i] / np.mean(task_features['cpu_req'])
            
            if mem_intensity > 1.5 and cpu_intensity > 1.5:
                task_labels.append('resource_intensive')
            elif mem_intensity > 1.5:
                task_labels.append('memory_intensive')
            elif cpu_intensity > 1.5:
                task_labels.append('cpu_intensive')
            else:
                task_labels.append('lightweight')
        
        vm_labels = []
        for i in range(len(vm_features['memory_total'])):
            # Categorize by capacity and efficiency
            mem_capacity = vm_features['memory_total'][i] / np.mean(vm_features['memory_total'])
            cpu_capacity = vm_features['cpu_cores'][i] / np.mean(vm_features['cpu_cores'])
            efficiency = vm_features['speed'][i] / vm_features['energy_rate'][i]
            
            if mem_capacity > 1.5 and cpu_capacity > 1.5:
                vm_labels.append('high_capacity')
            elif efficiency > np.mean([vm_features['speed'][j] / vm_features['energy_rate'][j] 
                                     for j in range(len(vm_features['speed']))]):
                vm_labels.append('efficient')
            else:
                vm_labels.append('standard')
        
        return {
            'task_labels': task_labels,
            'vm_labels': vm_labels,
            'task_categories': list(set(task_labels)),
            'vm_categories': list(set(vm_labels))
        }

    def visualize_embeddings_tsne(self, embeddings_data, labels_data, title="GIN Embeddings t-SNE"):
        """Create t-SNE visualization of embeddings with labels.
        Accepts either string or numeric labels and maps them to numeric codes for coloring.
        """
        
        # Combine task and VM embeddings
        all_embeddings = np.vstack([
            embeddings_data['task_embeddings'],
            embeddings_data['vm_embeddings']
        ])
        
        # Node types
        node_types = (['Task'] * len(embeddings_data['task_embeddings']) + 
                     ['VM'] * len(embeddings_data['vm_embeddings']))
        
        # Normalize labels into numeric codes for coloring
        def _to_numeric_codes(seq):
            # If already numeric, return as list
            if isinstance(seq, (list, tuple, np.ndarray)) and len(seq) > 0 and np.issubdtype(np.array(seq).dtype, np.number):
                return np.array(seq).astype(float).tolist()
            # Otherwise, treat as categorical strings
            cat = pd.Categorical(seq)
            return cat.codes.astype(float).tolist()
        
        task_label_codes = _to_numeric_codes(labels_data['task_labels'])
        vm_label_codes = _to_numeric_codes(labels_data['vm_labels'])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1, len(all_embeddings)-1)))
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Create interactive plot
        fig = go.Figure()
        
        # Plot tasks
        task_count = len(embeddings_data['task_embeddings'])
        fig.add_trace(go.Scatter(
            x=embeddings_2d[:task_count, 0],
            y=embeddings_2d[:task_count, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=task_label_codes,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"Task {i}: {labels_data['task_labels'][i]}" for i in range(task_count)],
            name='Tasks',
            hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
        
        # Plot VMs
        vm_count = len(embeddings_data['vm_embeddings'])
        fig.add_trace(go.Scatter(
            x=embeddings_2d[task_count:, 0],
            y=embeddings_2d[task_count:, 1],
            mode='markers',
            marker=dict(
                size=10,
                color=vm_label_codes,
                colorscale='Plasma',
                symbol='square',
                showscale=True
            ),
            text=[f"VM {i}: {labels_data['vm_labels'][i]}" for i in range(vm_count)],
            name='VMs',
            hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hovermode='closest'
        )
        
        return fig

    def visualize_embeddings_umap(self, embeddings_data, labels_data, title="GIN Embeddings UMAP"):
        """Create UMAP visualization of embeddings."""
        
        # Combine embeddings
        all_embeddings = np.vstack([
            embeddings_data['task_embeddings'],
            embeddings_data['vm_embeddings']
        ])
        
        # Apply UMAP
        if UMAP_AVAILABLE:
            reducer = UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(all_embeddings)
        else:
            print("Warning: UMAP not available, falling back to t-SNE")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1, len(all_embeddings)-1)))
            embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'type': (['Task'] * len(embeddings_data['task_embeddings']) + 
                    ['VM'] * len(embeddings_data['vm_embeddings'])),
            'label': (labels_data['task_labels'] + 
                     [f"VM_{label}" for label in labels_data['vm_labels']]),
            'id': (list(range(len(embeddings_data['task_embeddings']))) + 
                  list(range(len(embeddings_data['vm_embeddings']))))
        })
        
        if PX_AVAILABLE:
            fig = px.scatter(df, x='x', y='y', color='label', symbol='type',
                             title=title,
                             hover_data=['id'],
                             labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'})
        else:
            # Fallback: basic GO scatter without px/pandas integrations
            fig = go.Figure()
            # Tasks
            mask_task = df['type'] == 'Task'
            fig.add_trace(go.Scatter(
                x=df.loc[mask_task, 'x'], y=df.loc[mask_task, 'y'], mode='markers', name='Task',
                marker=dict(size=7, color='rgba(31,119,180,0.7)'),
                text=[f"Task {i}: {lab}" for i, lab in zip(df.loc[mask_task, 'id'], df.loc[mask_task, 'label'])],
                hovertemplate='%{text}<br>x:%{x:.2f}<br>y:%{y:.2f}<extra></extra>'
            ))
            # VMs
            mask_vm = df['type'] == 'VM'
            fig.add_trace(go.Scatter(
                x=df.loc[mask_vm, 'x'], y=df.loc[mask_vm, 'y'], mode='markers', name='VM',
                marker=dict(size=8, color='rgba(255,127,14,0.7)', symbol='square'),
                text=[f"VM {i}: {lab}" for i, lab in zip(df.loc[mask_vm, 'id'], df.loc[mask_vm, 'label'])],
                hovertemplate='%{text}<br>x:%{x:.2f}<br>y:%{y:.2f}<extra></extra>'
            ))
            fig.update_layout(title=title, xaxis_title='UMAP Dimension 1', yaxis_title='UMAP Dimension 2')
 
        return fig

    def analyze_embedding_clusters(self, embeddings_data, labels_data):
        """Analyze how well embeddings separate different categories."""
        
        # Task embedding analysis
        task_embeddings = embeddings_data['task_embeddings']
        task_labels = labels_data['task_labels']
        
        # Calculate within-cluster and between-cluster distances
        results = {}
        
        for category in labels_data['task_categories']:
            category_mask = np.array(task_labels) == category
            category_embeddings = task_embeddings[category_mask]
            
            if len(category_embeddings) > 1:
                # Within-cluster distances
                within_distances = []
                for i in range(len(category_embeddings)):
                    for j in range(i+1, len(category_embeddings)):
                        dist = np.linalg.norm(category_embeddings[i] - category_embeddings[j])
                        within_distances.append(dist)
                
                results[f'task_{category}_within_dist'] = np.mean(within_distances) if within_distances else 0
        
        # Between-cluster distances
        between_distances = []
        categories = list(set(task_labels))
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                mask1 = np.array(task_labels) == cat1
                mask2 = np.array(task_labels) == cat2
                
                emb1 = task_embeddings[mask1]
                emb2 = task_embeddings[mask2]
                
                for e1 in emb1:
                    for e2 in emb2:
                        dist = np.linalg.norm(e1 - e2)
                        between_distances.append(dist)
        
        results['task_between_cluster_dist'] = np.mean(between_distances) if between_distances else 0
        
        # Silhouette-like score
        if results['task_between_cluster_dist'] > 0:
            avg_within = np.mean([v for k, v in results.items() if 'within_dist' in k])
            results['task_separation_score'] = results['task_between_cluster_dist'] / (avg_within + 1e-8)
        
        return results

    def visualize_action_scores_heatmap(self, embeddings_data, labels_data):
        """Visualize action scores as heatmap with task/VM labels."""
        
        action_scores = embeddings_data['action_scores']
        
        fig = go.Figure(data=go.Heatmap(
            z=action_scores,
            x=[f"VM_{i}_{label}" for i, label in enumerate(labels_data['vm_labels'])],
            y=[f"Task_{i}_{label}" for i, label in enumerate(labels_data['task_labels'])],
            colorscale='RdYlBu_r',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Action Scores Heatmap (Task-VM Assignments)",
            xaxis_title="VMs",
            yaxis_title="Tasks"
        )
        
        return fig

    def create_comprehensive_dashboard(self, obs_tensor, labeling_method='characteristics'):
        """Create a comprehensive dashboard for GIN interpretability."""
        
        # Extract embeddings and features
        embeddings_data = self.extract_embeddings_and_features(obs_tensor)
        
        # Create labels
        labels_data = self.create_task_vm_labels(
            embeddings_data['task_features'],
            embeddings_data['vm_features'],
            method=labeling_method
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('t-SNE Embeddings', 'UMAP Embeddings', 
                           'Action Scores Heatmap', 'Embedding Statistics'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # t-SNE plot
        tsne_fig = self.visualize_embeddings_tsne(embeddings_data, labels_data)
        for trace in tsne_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # UMAP plot
        umap_fig = self.visualize_embeddings_umap(embeddings_data, labels_data)
        for trace in umap_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Action scores heatmap
        action_heatmap = self.visualize_action_scores_heatmap(embeddings_data, labels_data)
        fig.add_trace(action_heatmap.data[0], row=2, col=1)
        
        # Embedding statistics
        cluster_analysis = self.analyze_embedding_clusters(embeddings_data, labels_data)
        stats_names = list(cluster_analysis.keys())
        stats_values = list(cluster_analysis.values())
        
        fig.add_trace(go.Bar(x=stats_names, y=stats_values, name="Embedding Stats"), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text=f"GIN Interpretability Dashboard - {labeling_method.title()} Labeling")
        
        return fig

    # ========================= Pairwise (task, VM) analysis =========================
    def _build_task_vm_pairs(self, embeddings_data, use_all_pairs=False):
        """
        Construct (task, VM) pairs and assemble:
          - raw pair features from task/vm features
          - learned pair embeddings as concat(task_emb, vm_emb)
          - optional action scores aligned per pair (NaN if not available)
        If use_all_pairs=False, restrict to compatibilities; else use full cartesian.
        """
        num_tasks = embeddings_data['num_tasks']
        num_vms = embeddings_data['num_vms']
        task_feats = embeddings_data['task_features']
        vm_feats = embeddings_data['vm_features']
        task_emb = embeddings_data['task_embeddings']
        vm_emb = embeddings_data['vm_embeddings']
        action_scores_mat = embeddings_data.get('action_scores', None)

        # Choose indices
        if use_all_pairs:
            pair_indices = [(i, j) for i in range(num_tasks) for j in range(num_vms)]
        else:
            compat = embeddings_data['compatibilities']  # shape [2, E]
            pair_indices = list(zip(compat[0].astype(int).tolist(), compat[1].astype(int).tolist()))

        # Feature slices
        t_len = task_feats['length']
        t_mem = task_feats['memory_req']
        t_cpu = task_feats['cpu_req']
        t_ready = task_feats['ready']
        v_speed = vm_feats['speed']
        v_mem = vm_feats['memory_total']
        v_cpu = vm_feats['cpu_cores']
        v_energy = vm_feats['energy_rate']
        v_ct = vm_feats['completion_time']

        raw_list, learned_list, act_list = [], [], []
        for (ti, vj) in pair_indices:
            raw_vec = [
                float(t_len[ti]), float(t_mem[ti]), float(t_cpu[ti]), float(t_ready[ti]),
                float(v_speed[vj]), float(v_mem[vj]), float(v_cpu[vj]), float(v_energy[vj]), float(v_ct[vj]),
                float(t_len[ti] / (v_speed[vj] + 1e-8)),
            ]
            raw_list.append(raw_vec)
            learned_list.append(np.concatenate([task_emb[ti], vm_emb[vj]], axis=0))
            if action_scores_mat is not None and action_scores_mat.ndim == 2 \
               and ti < action_scores_mat.shape[0] and vj < action_scores_mat.shape[1]:
                act_list.append(float(action_scores_mat[ti, vj]))
            else:
                act_list.append(np.nan)

        return {
            'pair_indices': pair_indices,
            'raw_features': np.asarray(raw_list, dtype=float),
            'learned_embeddings': np.asarray(learned_list, dtype=float),
            'action_scores': np.asarray(act_list, dtype=float),
        }

    def _pair_labels(self, labels_data, pair_indices, method='characteristics', raw_features=None, n_clusters=6):
        """Create labels for (task, VM) pairs via characteristics or KMeans on raw_features."""
        if method == 'kmeans':
            assert raw_features is not None, "raw_features required for kmeans labeling"
            km = KMeans(n_clusters=n_clusters, random_state=42)
            y = km.fit_predict(raw_features)
            return [f"pair_cluster_{i}" for i in y], [f"pair_cluster_{i}" for i in range(n_clusters)]

        # default: characteristics (combine per-node labels)
        task_labels = labels_data['task_labels']
        vm_labels = labels_data['vm_labels']
        pair_labels = [f"{task_labels[ti]}|{vm_labels[vj]}" for (ti, vj) in pair_indices]
        categories = sorted(list(set(pair_labels)))
        return pair_labels, categories

    def _reduce_2d(self, X, method='tsne', random_state=42):
        if X.shape[0] < 2:
            return np.zeros((X.shape[0], 2))
        if method.lower() == 'umap' and UMAP_AVAILABLE:
            reducer = UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(X)
        elif method.lower() == 'umap' and not UMAP_AVAILABLE:
            print("UMAP reduction requested but not available - using t-SNE instead")
            method = 'tsne'
        # TSNE
        perp = min(30, max(2, X.shape[0] - 1))
        return TSNE(n_components=2, random_state=random_state, perplexity=perp).fit_transform(X)

    def visualize_task_vm_pairs(self, obs_tensor, labeling_method='characteristics', reducer='tsne', use_all_pairs=False, n_clusters=6, title_prefix='Task-VM Pairs', highlight_pairs=None, top_k_highlight=None):
        """
        Compare raw pair features vs learned pair embeddings with t-SNE/UMAP and labels.
        Returns (plotly_fig, metrics_dict).
        """
        embeddings_data = self.extract_embeddings_and_features(obs_tensor)
        # Base per-node labels via characteristics; pair labeling can still be kmeans
        labels_data = self.create_task_vm_labels(
            embeddings_data['task_features'], embeddings_data['vm_features'], method='characteristics'
        )

        # Extract ONLY task-VM compatibility edges for visualization
        compat_indices = embeddings_data['compatibilities']
        task_vm_pairs = list(zip(compat_indices[0].astype(int).tolist(), compat_indices[1].astype(int).tolist()))
        
        # Build pair data using only task-VM edges
        pair_data = {
            'pair_indices': task_vm_pairs,
            'raw_features': [],
            'learned_embeddings': [],
            'action_scores': []
        }
        
        # Fill pair data for task-VM edges only
        task_emb = embeddings_data['task_embeddings']
        vm_emb = embeddings_data['vm_embeddings']
        action_scores_mat = embeddings_data.get('action_scores', None)
        
        for ti, vj in task_vm_pairs:
            # Raw features
            raw_vec = [
                float(embeddings_data['task_features']['length'][ti]),
                float(embeddings_data['task_features']['memory_req'][ti]),
                float(embeddings_data['vm_features']['speed'][vj]),
                float(embeddings_data['vm_features']['memory_total'][vj]),
                float(embeddings_data['task_features']['length'][ti] / (embeddings_data['vm_features']['speed'][vj] + 1e-8))
            ]
            pair_data['raw_features'].append(raw_vec)
            
            # Learned embeddings
            pair_data['learned_embeddings'].append(np.concatenate([task_emb[ti], vm_emb[vj]]))
            
            # Action scores
            if action_scores_mat is not None and action_scores_mat.ndim == 2 \
               and ti < action_scores_mat.shape[0] and vj < action_scores_mat.shape[1]:
                pair_data['action_scores'].append(float(action_scores_mat[ti, vj]))
            else:
                pair_data['action_scores'].append(np.nan)
        
        # Convert to numpy arrays
        pair_data['raw_features'] = np.asarray(pair_data['raw_features'], dtype=float)
        pair_data['learned_embeddings'] = np.asarray(pair_data['learned_embeddings'], dtype=float)
        pair_data['action_scores'] = np.asarray(pair_data['action_scores'], dtype=float)
        
        pair_labels, _ = self._pair_labels(
            labels_data, pair_data['pair_indices'], method=labeling_method, raw_features=pair_data['raw_features'], n_clusters=n_clusters
        )

        if not UMAP_AVAILABLE and reducer.lower() == 'umap':
            print("UMAP reduction requested but not available - using t-SNE instead")

        pair_codes = pd.Categorical(pair_labels).codes.astype(int)
        raw_2d = self._reduce_2d(pair_data['raw_features'], method=reducer)
        emb_2d = self._reduce_2d(pair_data['learned_embeddings'], method=reducer)

        pairs_text = [f"task {ti} — vm {vj}" for (ti, vj) in pair_data['pair_indices']]
        action_vals = pair_data['action_scores']
        df_raw = pd.DataFrame({'x': raw_2d[:, 0], 'y': raw_2d[:, 1], 'label': pair_labels, 'pair': pairs_text, 'action': action_vals})
        df_emb = pd.DataFrame({'x': emb_2d[:, 0], 'y': emb_2d[:, 1], 'label': pair_labels, 'pair': pairs_text, 'action': action_vals})

        # Build highlight mask: either explicit highlight_pairs or top-K by action scores
        highlight_mask = np.zeros(len(pair_data['pair_indices']), dtype=bool)
        if highlight_pairs is not None:
            hi = set(highlight_pairs)
            for idx, (ti, vj) in enumerate(pair_data['pair_indices']):
                if (ti, vj) in hi:
                    highlight_mask[idx] = True
        elif top_k_highlight is not None and np.isfinite(action_vals).any():
            k = max(1, int(top_k_highlight))
            scores = np.nan_to_num(action_vals, nan=-1e9)
            top_idx = np.argsort(-scores)[:k]
            highlight_mask[top_idx] = True

        # Metrics
        metrics = {}
        if len(set(pair_codes)) > 1 and len(pair_codes) > len(set(pair_codes)):
            try:
                metrics['silhouette_raw'] = float(silhouette_score(pair_data['raw_features'], pair_codes))
            except Exception:
                pass
            try:
                metrics['silhouette_emb'] = float(silhouette_score(pair_data['learned_embeddings'], pair_codes))
            except Exception:
                pass
        if labeling_method == 'kmeans':
            km_raw = KMeans(n_clusters=n_clusters, random_state=42).fit(pair_data['raw_features'])
            km_emb = KMeans(n_clusters=n_clusters, random_state=42).fit(pair_data['learned_embeddings'])
            metrics['ari_raw_vs_label'] = float(adjusted_rand_score(pair_codes, km_raw.labels_))
            metrics['ari_emb_vs_label'] = float(adjusted_rand_score(pair_codes, km_emb.labels_))
            metrics['nmi_raw_vs_label'] = float(normalized_mutual_info_score(pair_codes, km_raw.labels_))
            metrics['nmi_emb_vs_label'] = float(normalized_mutual_info_score(pair_codes, km_emb.labels_))

        # Two subplots: raw vs learned
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f"{title_prefix} — Raw ({reducer.upper()})",
            f"{title_prefix} — Learned ({reducer.upper()})"
        ))

        # Use a single uniform color for all points; highlight selected feasible pairs
        base_color = 'rgba(31, 119, 180, 1.0)'  # steelblue
        low_opacity = 0.15
        high_opacity = 0.95

        if highlight_mask.any():
            df_raw_hi = df_raw[highlight_mask]
            df_raw_lo = df_raw[~highlight_mask]
            df_emb_hi = df_emb[highlight_mask]
            df_emb_lo = df_emb[~highlight_mask]

            # Raw space
            fig.add_trace(
                go.Scatter(
                    x=df_raw_lo['x'], y=df_raw_lo['y'], mode='markers', name='Feasible pairs',
                    marker=dict(color=base_color, size=6, opacity=low_opacity),
                    text=df_raw_lo['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_raw_hi['x'], y=df_raw_hi['y'], mode='markers', name='Top-K pairs',
                    marker=dict(color=base_color, size=11, opacity=high_opacity, line=dict(width=1, color='black')),
                    text=df_raw_hi['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )

            # Embedding space
            fig.add_trace(
                go.Scatter(
                    x=df_emb_lo['x'], y=df_emb_lo['y'], mode='markers', name='Feasible pairs',
                    marker=dict(color=base_color, size=6, opacity=low_opacity),
                    text=df_emb_lo['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=df_emb_hi['x'], y=df_emb_hi['y'], mode='markers', name='Top-K pairs',
                    marker=dict(color=base_color, size=11, opacity=high_opacity, line=dict(width=1, color='black')),
                    text=df_emb_hi['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=2
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_raw['x'], y=df_raw['y'], mode='markers', name='Feasible pairs',
                    marker=dict(color=base_color, size=7, opacity=0.6),
                    text=df_raw['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_emb['x'], y=df_emb['y'], mode='markers', name='Feasible pairs',
                    marker=dict(color=base_color, size=7, opacity=0.6),
                    text=df_emb['pair'], hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                ), row=1, col=2
            )

        fig.update_layout(height=500, width=1100, title_text=f"{title_prefix} — labeling: {labeling_method}")
        fig.update_xaxes(title_text=f"{reducer.upper()} 1", row=1, col=1)
        fig.update_yaxes(title_text=f"{reducer.upper()} 2", row=1, col=1)
        fig.update_xaxes(title_text=f"{reducer.upper()} 1", row=1, col=2)
        fig.update_yaxes(title_text=f"{reducer.upper()} 2", row=1, col=2)

        return fig, metrics

    def compare_models_top_pairs(self, obs_tensor, model_paths, n_top_pairs=20, reducer='tsne'):
        """
        Compare top probability pairs across multiple models.
        Args:
            model_paths: List of 3 model paths to compare
            n_top_pairs: Number of top pairs to highlight
        """
        # Load all models
        models = []
        for path in model_paths:
            variant = AblationVariant(name="viz_gin_linear")
            model = AblationGinAgent(device=self.device, variant=variant)
            model.load_state_dict(torch.load(path, map_location=self.device))
            models.append((Path(path).parent.name, model))
        
        # Get top pairs and their probabilities from each model
        all_top_pairs = []
        for name, model in models:
            interpreter = GINInterpreter(model, device=str(self.device))
            embeddings_data = interpreter.extract_embeddings_and_features(obs_tensor)
            
            # Get top scoring pairs
            scores = embeddings_data['action_scores']
            compat = embeddings_data['compatibilities']
            flat_scores = scores[compat[0], compat[1]]
            top_indices = np.argsort(-flat_scores)[:n_top_pairs]
            top_pairs = [(compat[0][i], compat[1][i], float(flat_scores[i])) for i in top_indices]
            all_top_pairs.append((name, top_pairs))
        
        # Create combined visualization
        fig = make_subplots(rows=1, cols=len(models), 
                           subplot_titles=[name for name, _ in models])
        
        for i, (name, top_pairs) in enumerate(all_top_pairs):
            # Get embeddings for this model
            interpreter = GINInterpreter(models[i][1], device=str(self.device))
            embeddings_data = interpreter.extract_embeddings_and_features(obs_tensor)
            
            # Prepare all pairs data
            pair_data = self._build_task_vm_pairs(embeddings_data)
            
            # Reduce dimensions
            reduced = self._reduce_2d(pair_data['learned_embeddings'], method=reducer)
            
            # Create scatter plot with opacity based on score
            sizes = np.ones(len(pair_data['pair_indices'])) * 5
            opacities = np.ones(len(pair_data['pair_indices'])) * 0.2
            
            # Highlight top pairs
            for ti, vj, score in top_pairs:
                idx = pair_data['pair_indices'].index((ti, vj))
                sizes[idx] = 10
                opacities[idx] = 1.0
            
            scatter = go.Scatter(
                x=reduced[:, 0],
                y=reduced[:, 1],
                mode='markers',
                marker=dict(
                    size=sizes,
                    opacity=opacities,
                    color='blue'
                ),
                name=name
            )
            fig.add_trace(scatter, row=1, col=i+1)
        
        fig.update_layout(height=500, width=1500, title_text="Top Pairs Comparison")
        return fig

    def get_pair_embeddings_and_scores(self, obs_tensor):
        """
        Returns learned pair embeddings and action scores over FEASIBLE (task, VM) pairs.
        Output:
          pair_indices: List[(task_idx, vm_idx)] of feasible pairs
          learned_embeddings: np.ndarray [N_pairs, 2*embedding_dim]
          action_scores: np.ndarray [N_pairs] (float), unnormalized scores
          action_probs: np.ndarray [N_pairs] (float), softmax over feasible pairs
        """
        data = self.extract_embeddings_and_features(obs_tensor)
        comp = data['compatibilities']  # shape (2, E)
        task_emb = data['task_embeddings']
        vm_emb = data['vm_embeddings']

        # Build pair indices list
        t_list = comp[0].astype(int).tolist()
        v_list = comp[1].astype(int).tolist()
        pair_indices = list(zip(t_list, v_list))

        # Learned pair embeddings = concat(task_emb, vm_emb)
        learned = []
        for ti, vj in pair_indices:
            learned.append(np.concatenate([task_emb[ti], vm_emb[vj]], axis=0))
        learned = np.asarray(learned, dtype=float)

        # Action scores matrix, gather feasible entries
        action_scores_mat = data.get('action_scores', None)
        scores = []
        if action_scores_mat is not None:
            for ti, vj in pair_indices:
                if ti < action_scores_mat.shape[0] and vj < action_scores_mat.shape[1]:
                    scores.append(float(action_scores_mat[ti, vj]))
                else:
                    scores.append(float('-inf'))
        else:
            scores = [0.0 for _ in pair_indices]
        action_scores = np.asarray(scores, dtype=float)

        # Softmax over finite scores to get probs
        s = np.copy(action_scores)
        mask = np.isfinite(s)
        if mask.any():
            s[~mask] = -1e9
            s = s - np.max(s)
            p = np.exp(s)
            p = p / (np.sum(p) + 1e-12)
        else:
            p = np.ones_like(s) / max(1, len(s))

        return pair_indices, learned, action_scores, p

    # ========================= Side-by-side model comparison =========================
    def _pair_data_for_obs(self, obs_tensor) -> Dict[str, Any]:
        """Helper: extract base data once per observation."""
        emb = self.extract_embeddings_and_features(obs_tensor)
        compat = emb['compatibilities']
        pairs = list(zip(compat[0].astype(int).tolist(), compat[1].astype(int).tolist()))
        return {
            'pairs': pairs,
            'task_emb': emb['task_embeddings'],
            'vm_emb': emb['vm_embeddings'],
            # ensure 1D graph embedding for concatenation
            'graph_emb': np.asarray(emb['graph_embedding']).reshape(-1),
        }
 
    def model_learned_pair_embeddings(self, obs_tensor, include_graph: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Return (learned_pair_embeddings, action_scores_per_pair, pair_indices) for current agent.
        If include_graph=True, append graph embedding to every pair embedding.
        """
        self.gin_agent.eval()
        with torch.no_grad():
            decoded = self.gin_agent.mapper.unmap(obs_tensor.squeeze())
            # Get action scores for (ti, vj)
            action_scores_mat = self.gin_agent.actor(decoded).cpu().numpy()
            # Also access the network's edge embeddings (already graph-aware)
            node_embeddings, edge_embeddings, graph_embedding = self.gin_agent.actor.network(decoded)
            edge_embeddings = edge_embeddings.detach().cpu().numpy()

        # Build pair indices in compatibilities order
        emb_data = self.extract_embeddings_and_features(obs_tensor)
        compat = emb_data['compatibilities']
        pairs: List[Tuple[int, int]] = list(zip(compat[0].astype(int).tolist(), compat[1].astype(int).tolist()))
        E = len(pairs)

        if include_graph:
            # Use the actor's edge embeddings, concatenated with the (replicated) graph embedding
            g_np = graph_embedding.detach().cpu().numpy().reshape(1, -1)
            rep_g = np.repeat(g_np, edge_embeddings.shape[0], axis=0)
            edge_plus_g = np.concatenate([edge_embeddings, rep_g], axis=1)
            learned_pairs = edge_plus_g[:E]
        else:
            # Use concat(task_emb, vm_emb) as the edge-only representation
            task_emb = emb_data['task_embeddings']
            vm_emb = emb_data['vm_embeddings']
            learned_list = [np.concatenate([task_emb[ti], vm_emb[vj]], axis=0) for (ti, vj) in pairs]
            learned_pairs = np.asarray(learned_list, dtype=float)

        # Gather action scores per pair in the same order
        scores = np.asarray([float(action_scores_mat[ti, vj]) for (ti, vj) in pairs], dtype=float)
        return learned_pairs, scores, pairs

    def visualize_side_by_side_models(self, obs_tensor, model_specs: List[Tuple[str, 'AblationGinAgent']], top_k: int = 3, reducer: str = 'tsne', include_graph: bool = False, title: str = "Model comparison"):
        """Create a side-by-side figure comparing multiple models (2 columns typical).
        model_specs: list of (label, agent)
        include_graph: if True, append graph embedding to pair embeddings for visualization
        """
        cols = len(model_specs)
        fig = make_subplots(rows=1, cols=cols, subplot_titles=[lab for lab, _ in model_specs])
        base_color = 'rgba(31, 119, 180, 1.0)'
        low_opacity = 0.15
        high_opacity = 0.95

        # Compute per-model embeddings and highlights
        emb_2d_all, df_all = [], []
        for lab, agent in model_specs:
            interpreter = GINInterpreter(agent, device=self.device)
            learned, scores, pairs = interpreter.model_learned_pair_embeddings(obs_tensor, include_graph=include_graph)
            # Reduce to 2D
            Z = self._reduce_2d(learned, method=reducer)
            # Top-K mask
            k = max(1, min(int(top_k), len(scores)))
            s = np.nan_to_num(scores, nan=-1e9)
            top_idx = np.argsort(-s)[:k]
            mask = np.zeros(len(scores), dtype=bool)
            mask[top_idx] = True
            pairs_text = [f"task {ti} — vm {vj}" for (ti, vj) in pairs]
            df = pd.DataFrame({'x': Z[:,0], 'y': Z[:,1], 'pair': pairs_text, 'highlight': mask})
            emb_2d_all.append(Z)
            df_all.append((lab, df))

        # Render per column
        for col_idx, (lab, df) in enumerate(df_all, start=1):
            lo = df[~df['highlight']]
            hi = df[df['highlight']]
            fig.add_trace(
                go.Scatter(
                    x=lo['x'], y=lo['y'], mode='markers', name=f'{lab} feasible',
                    marker=dict(color=base_color, size=6, opacity=low_opacity),
                    text=lo['pair'], hovertemplate='%{text}<br>x:%{x:.2f}<br>y:%{y:.2f}<extra></extra>'
                ), row=1, col=col_idx
            )
            fig.add_trace(
                go.Scatter(
                    x=hi['x'], y=hi['y'], mode='markers', name=f'{lab} top-{top_k}',
                    marker=dict(color=base_color, size=11, opacity=high_opacity, line=dict(width=1, color='black')),
                    text=hi['pair'], hovertemplate='%{text}<br>x:%{x:.2f}<br>y:%{y:.2f}<extra></extra>'
                ), row=1, col=col_idx
            )

        fig.update_layout(height=480, width=1200, title_text=title, showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        for c in range(1, cols+1):
            fig.update_xaxes(title_text=f"{reducer.upper()} 1", row=1, col=c)
            fig.update_yaxes(title_text=f"{reducer.upper()} 2", row=1, col=c)
        return fig

    # Usage example and additional analysis functions
    def analyze_gin_decision_process(gin_agent, obs_tensor):
        """Analyze the decision-making process of GIN step by step."""
        
        interpreter = GINInterpreter(gin_agent)
        embeddings_data = interpreter.extract_embeddings_and_features(obs_tensor)
        
        # 1. Feature importance analysis
        def feature_importance_analysis():
            # Analyze which features contribute most to embeddings
            task_features = embeddings_data['task_features']
            vm_features = embeddings_data['vm_features']
            
            # Correlation between features and embedding dimensions
            task_feature_matrix = np.column_stack([
                task_features['length'],
                task_features['memory_req'],
                task_features['cpu_req'],
                task_features['completion_time']
            ])
            
            task_embeddings = embeddings_data['task_embeddings']
            
            # Calculate correlations
            correlations = {}
            feature_names = ['length', 'memory_req', 'cpu_req', 'completion_time']
            
            for i, feature_name in enumerate(feature_names):
                feature_corrs = []
                for dim in range(task_embeddings.shape[1]):
                    corr = np.corrcoef(task_feature_matrix[:, i], task_embeddings[:, dim])[0, 1]
                    feature_corrs.append(abs(corr))
                correlations[feature_name] = np.mean(feature_corrs)
            
            return correlations
        
        # 2. Decision boundary analysis
        def decision_boundary_analysis():
            action_scores = embeddings_data['action_scores']
            
            # Find the chosen action
            flat_scores = action_scores.flatten()
            chosen_action_idx = np.argmax(flat_scores)
            
            num_vms = embeddings_data['num_vms']
            chosen_task = chosen_action_idx // num_vms
            chosen_vm = chosen_action_idx % num_vms
            
            # Analyze why this action was chosen
            task_embedding = embeddings_data['task_embeddings'][chosen_task]
            vm_embedding = embeddings_data['vm_embeddings'][chosen_vm]
            
            # Compare with other possible actions for the same task
            task_scores = action_scores[chosen_task, :]
            alternative_scores = [(i, score) for i, score in enumerate(task_scores) if i != chosen_vm]
            alternative_scores.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'chosen_action': (chosen_task, chosen_vm),
                'chosen_score': flat_scores[chosen_action_idx],
                'task_embedding': task_embedding,
                'vm_embedding': vm_embedding,
                'top_alternatives': alternative_scores[:3]
            }
        
        feature_importance = feature_importance_analysis()
        decision_analysis = decision_boundary_analysis()
        
        return {
            'embeddings_data': embeddings_data,
            'feature_importance': feature_importance,
            'decision_analysis': decision_analysis
        }

    def compare_embedding_evolution(gin_agent, obs_sequence):
        """Compare how embeddings evolve across multiple time steps."""
        
        interpreter = GINInterpreter(gin_agent)
        evolution_data = []
        
        for step, obs_tensor in enumerate(obs_sequence):
            embeddings_data = interpreter.extract_embeddings_and_features(obs_tensor)
            evolution_data.append({
                'step': step,
                'task_embeddings': embeddings_data['task_embeddings'],
                'vm_embeddings': embeddings_data['vm_embeddings'],
                'graph_embedding': embeddings_data['graph_embedding']
            })
        
        # Analyze embedding stability
        def embedding_stability_analysis():
            if len(evolution_data) < 2:
                return {}
            
            stability_scores = {}
            
            # Task embedding stability
            task_stabilities = []
            for i in range(len(evolution_data[0]['task_embeddings'])):
                task_embeddings_over_time = [data['task_embeddings'][i] for data in evolution_data]
                
                # Calculate pairwise distances
                distances = []
                for j in range(len(task_embeddings_over_time) - 1):
                    dist = np.linalg.norm(task_embeddings_over_time[j] - task_embeddings_over_time[j+1])
                    distances.append(dist)
                
                task_stabilities.append(np.mean(distances))
            
            stability_scores['task_stability'] = np.mean(task_stabilities)
            
            # Graph embedding stability
            graph_embeddings = [data['graph_embedding'] for data in evolution_data]
            graph_distances = []
            for j in range(len(graph_embeddings) - 1):
                dist = np.linalg.norm(graph_embeddings[j] - graph_embeddings[j+1])
                graph_distances.append(dist)
            
            stability_scores['graph_stability'] = np.mean(graph_distances)
            
            return stability_scores
        
        return {
            'evolution_data': evolution_data,
            'stability_analysis': embedding_stability_analysis()
        }