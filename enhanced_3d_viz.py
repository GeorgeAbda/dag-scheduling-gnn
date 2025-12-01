#!/usr/bin/env python3
"""
Enhanced 3D Feasibility Space Visualization
Improved styling with professional aesthetics, better lighting, and modern design
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm
import os

# Add scheduler to path
THIS_DIR = Path(__file__).resolve().parent
SCHEDULER_DIR = THIS_DIR / "scheduler"
if str(SCHEDULER_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULER_DIR))

from scheduler.viz_results.decision_boundaries.solution_landscape import (
    EnumArgs, build_env, enumerate_spacefill_once
)

def create_enhanced_colormap():
    """Create a beautiful gradient colormap for 3D surfaces"""
    # Professional color scheme: deep blue -> cyan -> green -> yellow -> orange -> red
    colors = [
        '#0d1b2a',  # Deep navy
        '#1b263b',  # Dark blue
        '#415a77',  # Steel blue
        '#778da9',  # Light blue
        '#a8dadc',  # Cyan
        '#457b9d',  # Medium blue
        '#1d3557',  # Dark blue-green
        '#f1faee',  # Off white (peaks)
        '#e63946',  # Red (highest values)
    ]
    return LinearSegmentedColormap.from_list('enhanced_surface', colors, N=512)

def create_enhanced_3d_surface(grids, feas_mask, visited, side, dag_type, metric, out_dir, 
                             vmin_global=None, vmax_global=None, paper_mode=False):
    """Create enhanced 3D surface with professional styling"""
    
    G_raw = grids[metric]
    m = feas_mask & ~np.isnan(G_raw)
    
    if not np.any(m):
        print(f"No feasible data for {dag_type} {metric}")
        return
    
    # Normalize with global bounds if provided
    if vmin_global is not None and vmax_global is not None:
        vmin, vmax = vmin_global, vmax_global
    else:
        vmin = float(np.nanmin(G_raw[m]))
        vmax = float(np.nanmax(G_raw[m]))
    
    Z = np.full_like(G_raw, np.nan)
    if vmax > vmin:
        # Monotonic mapping: higher metric -> higher Z and higher color
        Z[m] = (G_raw[m] - vmin) / (vmax - vmin)
    else:
        Z[m] = 0.5
    
    # Enhanced figure setup
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Create enhanced colormap
    cmap_enhanced = create_enhanced_colormap()
    
    # Prepare coordinates and crop to feasible region
    H, W = Z.shape
    feas_idx = np.argwhere(feas_mask)
    
    if feas_idx.size > 0:
        pad = 3
        iy_min = max(0, int(feas_idx[:, 0].min()) - pad)
        iy_max = min(H - 1, int(feas_idx[:, 0].max()) + pad)
        ix_min = max(0, int(feas_idx[:, 1].min()) - pad)
        ix_max = min(W - 1, int(feas_idx[:, 1].max()) + pad)
    else:
        iy_min, iy_max, ix_min, ix_max = 0, H - 1, 0, W - 1
    
    # Create mesh for cropped region
    y_range = np.arange(iy_min, iy_max + 1)
    x_range = np.arange(ix_min, ix_max + 1)
    X, Y = np.meshgrid(x_range, y_range)
    Z_plot = Z[np.ix_(y_range, x_range)]
    feas_crop = feas_mask[np.ix_(y_range, x_range)]
    
    # Mask infeasible regions
    Z_plot[~feas_crop] = np.nan
    
    # Apply light smoothing for better visual quality (simplified approach)
    Z_smooth = Z_plot.copy()
    valid_mask = ~np.isnan(Z_plot)
    if np.any(valid_mask):
        try:
            from scipy.ndimage import gaussian_filter
            # Simple 2D smoothing on the entire array, preserving NaNs
            temp = Z_plot.copy()
            temp[np.isnan(temp)] = 0  # Temporarily fill NaNs
            smoothed = gaussian_filter(temp, sigma=0.3)
            Z_smooth = np.where(valid_mask, smoothed, np.nan)
        except ImportError:
            # Fallback: no smoothing if scipy not available
            Z_smooth = Z_plot
    
    # Create the main surface with consistent color-height mapping
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    facecols = cmap_enhanced(norm(Z_smooth))
    surf = ax.plot_surface(
        X, Y, Z_smooth,
        facecolors=facecols,
        cmap=None,
        alpha=0.95,
        linewidth=0.1,
        edgecolor='white',
        antialiased=True,
        shade=False,
        rstride=1, cstride=1
    )
    
    # Add subtle wireframe for structure
    ax.plot_wireframe(
        X, Y, Z_smooth,
        rstride=3, cstride=3,
        color='white',
        alpha=0.15,
        linewidth=0.3
    )
    
    # Enhanced contour lines at the base
    if np.any(~np.isnan(Z_smooth)):
        z_base = np.nanmin(Z_smooth) - 0.05
        levels = np.linspace(np.nanmin(Z_smooth), np.nanmax(Z_smooth), 8)
        contours = ax.contour(
            X, Y, Z_smooth,
            levels=levels,
            colors=['#2c3e50'],
            alpha=0.4,
            linewidths=0.8,
            zdir='z',
            offset=z_base
        )
    
    # Mark global optimum with enhanced styling
    if np.any(~np.isnan(Z_smooth)):
        flat_idx = np.nanargmin(Z_smooth)
        iy_opt, ix_opt = np.unravel_index(flat_idx, Z_smooth.shape)
        x_opt, y_opt, z_opt = x_range[ix_opt], y_range[iy_opt], Z_smooth[iy_opt, ix_opt]
        
        # Optimum marker with glow effect
        ax.scatter([x_opt], [y_opt], [z_opt], 
                  c='gold', s=120, alpha=1.0, 
                  edgecolors='darkred', linewidths=2,
                  marker='*', depthshade=False)
        
        # Add a subtle pillar to the optimum
        ax.plot([x_opt, x_opt], [y_opt, y_opt], [z_base, z_opt],
               color='gold', alpha=0.6, linewidth=3)
    
    # Professional axis styling
    ax.set_xlim(ix_min, ix_max)
    ax.set_ylim(iy_min, iy_max)
    ax.set_zlim(0, 1)
    
    # Clean axis appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Subtle pane colors
    ax.xaxis.pane.set_edgecolor('#e8e8e8')
    ax.yaxis.pane.set_edgecolor('#e8e8e8')
    ax.zaxis.pane.set_edgecolor('#e8e8e8')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, color='gray', linestyle='-', linewidth=0.5)
    
    # Minimal ticks for clean look
    ax.set_xticks(np.linspace(ix_min, ix_max, 4))
    ax.set_yticks(np.linspace(iy_min, iy_max, 4))
    ax.set_zticks([0, 0.5, 1.0])
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=9, pad=1, length=3, width=0.8)
    ax.tick_params(axis='z', which='major', labelsize=9, pad=3, length=3, width=0.8)
    
    # Professional viewing angle (lower perspective)
    ax.view_init(elev=29, azim=-65)
    ax.dist = 8  # Camera distance
    
    # Enhanced title and labels
    if not paper_mode:
        title = f"{dag_type.upper()} DAG - {metric.capitalize()} Landscape"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
        
        ax.set_xlabel('Solution Space X', fontsize=10, labelpad=8, color='#34495e')
        ax.set_ylabel('Solution Space Y', fontsize=10, labelpad=8, color='#34495e')
        ax.set_zlabel(f'Normalized {metric.capitalize()}', fontsize=10, labelpad=8, color='#34495e')
    
    # Enhanced colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap_enhanced)
    sm.set_array([])
    
    # Position colorbar nicely
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=25, pad=0.1, 
                       orientation='vertical', fraction=0.046)
    
    # Colorbar styling
    cbar.set_label(f'{metric.capitalize()}', fontsize=11, fontweight='bold', 
                  color='#2c3e50', labelpad=15)
    cbar.ax.tick_params(labelsize=9, colors='#34495e', width=0.8, length=3)
    
    # Add actual value labels to colorbar
    if vmin_global is not None and vmax_global is not None:
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels([f'{vmin:.2f}', f'{(vmin+vmax)/2:.2f}', f'{vmax:.2f}'])
    
    # Enhanced statistics annotation
    visited_count = int(np.sum(visited))
    feasible_count = int(np.sum(feas_mask & visited))
    feas_frac = (feasible_count / visited_count) if visited_count > 0 else 0.0
    
    if not paper_mode:
        stats_text = (
            f"Topology: {dag_type.upper()}\n"
            f"Feasible: {feasible_count:,}/{visited_count:,} ({feas_frac:.1%})\n"
            f"Range: {vmin:.3f} - {vmax:.3f}"
        )
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 fontsize=10, va='top', ha='left', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                          alpha=0.9, edgecolor='#bdc3c7', linewidth=1.2))
    
    # Save with high quality
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"enhanced_3d_{dag_type}_{metric}"
    
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    
    # Tight layout with padding
    plt.tight_layout(pad=1.5)
    
    # High-quality output
    fig.savefig(png_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               transparent=False, pad_inches=0.2)
    fig.savefig(pdf_path, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               transparent=False, pad_inches=0.2)
    
    plt.close(fig)
    
    print(f"Enhanced 3D surface saved: {png_path}")
    return png_path, pdf_path

def create_enhanced_side_by_side_3d(a, sf_limit, metric, out_dir):
    """Create enhanced side-by-side 3D comparison with professional styling"""
    
    # Build environments for both DAG types
    a_linear = EnumArgs(**{**a.__dict__})
    a_linear.dag_method = "linear"
    a_gnp = EnumArgs(**{**a.__dict__})
    a_gnp.dag_method = "gnp"
    
    env_lin = build_env(a_linear)
    env_gnp = build_env(a_gnp)
    
    print("Enumerating linear DAG solutions...")
    grids_lin, feas_lin, visited_lin, side_lin = enumerate_spacefill_once(env_lin, a_linear, sf_limit)
    print("Enumerating GNP DAG solutions...")
    grids_gnp, feas_gnp, visited_gnp, side_gnp = enumerate_spacefill_once(env_gnp, a_gnp, sf_limit)
    
    # Get global bounds for consistent scaling
    G_lin_raw = grids_lin[metric]
    G_gnp_raw = grids_gnp[metric]
    m_lin = feas_lin & ~np.isnan(G_lin_raw)
    m_gnp = feas_gnp & ~np.isnan(G_gnp_raw)
    
    vals = []
    if np.any(m_lin):
        vals.append(G_lin_raw[m_lin])
    if np.any(m_gnp):
        vals.append(G_gnp_raw[m_gnp])
    
    if len(vals) == 0:
        print("No feasible values found!")
        return
    
    allv = np.concatenate(vals)
    vmin_global = float(np.nanmin(allv))
    vmax_global = float(np.nanmax(allv))
    
    print(f"Global {metric} range: {vmin_global:.3f} - {vmax_global:.3f}")
    
    # Create individual enhanced surfaces
    create_enhanced_3d_surface(grids_lin, feas_lin, visited_lin, side_lin, 
                              "linear", metric, out_dir, vmin_global, vmax_global)
    
    create_enhanced_3d_surface(grids_gnp, feas_gnp, visited_gnp, side_gnp, 
                              "gnp", metric, out_dir, vmin_global, vmax_global)
    
    # Create enhanced side-by-side comparison
    create_enhanced_dual_panel_3d(grids_lin, feas_lin, visited_lin, side_lin,
                                 grids_gnp, feas_gnp, visited_gnp, side_gnp,
                                 metric, out_dir, vmin_global, vmax_global)
    
    print(f"Enhanced 3D visualizations completed for {metric}")

def create_enhanced_dual_panel_3d(grids_lin, feas_lin, visited_lin, side_lin,
                                 grids_gnp, feas_gnp, visited_gnp, side_gnp,
                                 metric, out_dir, vmin_global, vmax_global):
    """Create enhanced side-by-side 3D surface comparison"""
    
    # Enhanced figure setup for dual panel
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    fig = plt.figure(figsize=(16, 8), dpi=300, facecolor='white')
    
    # Create enhanced colormap
    cmap_enhanced = create_enhanced_colormap()
    
    # Enhanced lighting
    ls = LightSource(azdeg=315, altdeg=60)
    
    def plot_enhanced_surface(ax, grids, feas_mask, visited, side, title, show_colorbar=False):
        """Plot a single enhanced 3D surface"""
        G_raw = grids[metric]
        m = feas_mask & ~np.isnan(G_raw)
        
        if not np.any(m):
            return
        
        # Normalize with global bounds
        Z = np.full_like(G_raw, np.nan)
        if vmax_global > vmin_global:
            # Monotonic mapping: higher metric -> higher Z and higher color
            Z[m] = (G_raw[m] - vmin_global) / (vmax_global - vmin_global)
        else:
            Z[m] = 0.5
        
        # Crop to feasible region
        H, W = Z.shape
        feas_idx = np.argwhere(feas_mask)
        
        if feas_idx.size > 0:
            pad = 3
            iy_min = max(0, int(feas_idx[:, 0].min()) - pad)
            iy_max = min(H - 1, int(feas_idx[:, 0].max()) + pad)
            ix_min = max(0, int(feas_idx[:, 1].min()) - pad)
            ix_max = min(W - 1, int(feas_idx[:, 1].max()) + pad)
        else:
            iy_min, iy_max, ix_min, ix_max = 0, H - 1, 0, W - 1
        
        # Create mesh for cropped region
        y_range = np.arange(iy_min, iy_max + 1)
        x_range = np.arange(ix_min, ix_max + 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z_plot = Z[np.ix_(y_range, x_range)]
        feas_crop = feas_mask[np.ix_(y_range, x_range)]
        
        # Mask infeasible regions
        Z_plot[~feas_crop] = np.nan
        
        # Apply smoothing
        Z_smooth = Z_plot.copy()
        valid_mask = ~np.isnan(Z_plot)
        if np.any(valid_mask):
            try:
                from scipy.ndimage import gaussian_filter
                temp = Z_plot.copy()
                temp[np.isnan(temp)] = 0
                smoothed = gaussian_filter(temp, sigma=0.3)
                Z_smooth = np.where(valid_mask, smoothed, np.nan)
            except ImportError:
                Z_smooth = Z_plot
        
        # Create the main surface with consistent color-height mapping
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=1)
        facecols = cmap_enhanced(norm(Z_smooth))
        surf = ax.plot_surface(
            X, Y, Z_smooth,
            facecolors=facecols,
            cmap=None,
            alpha=0.95,
            linewidth=0.1,
            edgecolor='white',
            antialiased=True,
            shade=False,
            rstride=1, cstride=1
        )
        
        # Add subtle wireframe
        ax.plot_wireframe(
            X, Y, Z_smooth,
            rstride=4, cstride=4,
            color='white',
            alpha=0.12,
            linewidth=0.25
        )
        
        # Enhanced contour lines at the base
        if np.any(~np.isnan(Z_smooth)):
            z_base = np.nanmin(Z_smooth) - 0.05
            levels = np.linspace(np.nanmin(Z_smooth), np.nanmax(Z_smooth), 6)
            ax.contour(
                X, Y, Z_smooth,
                levels=levels,
                colors=['#2c3e50'],
                alpha=0.3,
                linewidths=0.6,
                zdir='z',
                offset=z_base
            )
            
            # Add yellow base surface like in reference image
            # Create a yellow base plane under the feasible regions
            Z_base = np.full_like(Z_smooth, z_base - 0.02)
            # Only show yellow base where we have feasible data
            Z_base[np.isnan(Z_smooth)] = np.nan
            
            ax.plot_surface(
                X, Y, Z_base,
                color='lightsteelblue',
                alpha=0.5,
                linewidth=0,
                antialiased=True,
                shade=False
            )
        
        # Mark global optimum
        if np.any(~np.isnan(Z_smooth)):
            flat_idx = np.nanargmin(Z_smooth)
            iy_opt, ix_opt = np.unravel_index(flat_idx, Z_smooth.shape)
            x_opt, y_opt, z_opt = x_range[ix_opt], y_range[iy_opt], Z_smooth[iy_opt, ix_opt]
            
            # Optimum marker with glow effect
            ax.scatter([x_opt], [y_opt], [z_opt], 
                      c='gold', s=100, alpha=1.0, 
                      edgecolors='darkred', linewidths=1.5,
                      marker='*', depthshade=False)
        
        # Professional axis styling
        ax.set_xlim(ix_min, ix_max)
        ax.set_ylim(iy_min, iy_max)
        ax.set_zlim(0, 1)
        
        # Clean axis appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Subtle pane colors
        ax.xaxis.pane.set_edgecolor('#e8e8e8')
        ax.yaxis.pane.set_edgecolor('#e8e8e8')
        ax.zaxis.pane.set_edgecolor('#e8e8e8')
        ax.xaxis.pane.set_alpha(0.08)
        ax.yaxis.pane.set_alpha(0.08)
        ax.zaxis.pane.set_alpha(0.08)
        
        # Enhanced grid
        ax.grid(True, alpha=0.15, color='gray', linestyle='-', linewidth=0.4)
        
        # Minimal ticks
        ax.set_xticks(np.linspace(ix_min, ix_max, 3))
        ax.set_yticks(np.linspace(iy_min, iy_max, 3))
        ax.set_zticks([0, 0.5, 1.0])
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=8, pad=1, length=2, width=0.6)
        ax.tick_params(axis='z', which='major', labelsize=8, pad=2, length=2, width=0.6)
        
        # Professional viewing angle (lower perspective) - adjusted for side-by-side
        ax.view_init(elev=29, azim=-55)
        try:
            ax.dist = 9.0  # Increased distance to prevent overlap
        except:
            pass
        
        # Enhanced title
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15, color='#2c3e50')
        
        # Statistics annotation
        visited_count = int(np.sum(visited))
        feasible_count = int(np.sum(feas_mask & visited))
        feas_frac = (feasible_count / visited_count) if visited_count > 0 else 0.0
        
        stats_text = f"Feasible: {feasible_count:,}/{visited_count:,} ({feas_frac:.1%})"
        ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes, 
                 fontsize=9, va='top', ha='left', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          alpha=0.85, edgecolor='#bdc3c7', linewidth=0.8))
        
        return surf
    
    # Create subplots with proper spacing to prevent interference
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Adjust subplot spacing with more separation and room for bottom colorbar
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12, wspace=0.3)
    
    # Plot both surfaces
    surf1 = plot_enhanced_surface(ax1, grids_lin, feas_lin, visited_lin, side_lin, "Linear DAG")
    surf2 = plot_enhanced_surface(ax2, grids_gnp, feas_gnp, visited_gnp, side_gnp, "GNP DAG")
    
    # No main title for conference paper
    
    # Shared colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap_enhanced)
    sm.set_array([])
    
    # Position colorbar at the bottom for better conference paper layout
    cbar_ax = fig.add_axes([0.15, 0.02, 0.6, 0.03])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Colorbar styling
    cbar.set_label(f'{metric.capitalize()}', fontsize=12, fontweight='bold', 
                  color='#2c3e50', labelpad=15)
    cbar.ax.tick_params(labelsize=9, colors='#34495e', width=0.8, length=3)
    
    # Add actual value labels (ascending with Z)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels([f'{vmin_global:.2f}', f'{(vmin_global+vmax_global)/2:.2f}', f'{vmax_global:.2f}'])
    
    # Save enhanced side-by-side visualization
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"enhanced_sidebyside_3d_{metric}"
    
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    
    # High-quality output
    fig.savefig(png_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               transparent=False, pad_inches=0.3)
    fig.savefig(pdf_path, bbox_inches='tight', 
               facecolor='white', edgecolor='none', 
               transparent=False, pad_inches=0.3)
    
    plt.close(fig)
    
    print(f"Enhanced side-by-side 3D saved: {png_path}")
    return png_path, pdf_path

def main():
    """Main function to generate enhanced 3D visualizations"""
    
    # Configuration for enhanced visualization
    a = EnumArgs(
        seed=42,
        host_count=2,
        vm_count=3,
        workflow_count=1,
        gnp_min_n=4,
        gnp_max_n=4,
        dag_method="linear",  # Will be overridden for comparison
        min_task_length=500,
        max_task_length=5000,
        min_cpu_speed=500,
        max_cpu_speed=3000,
        max_memory_gb=4,
        show_progress=True,
        out_dir="logs/landscape",
        dpi=300,
        paper=False
    )
    
    # Generate enhanced 3D visualizations
    sf_limit = 2000
    
    print("Generating enhanced 3D feasibility landscapes...")
    create_enhanced_side_by_side_3d(a, sf_limit, "makespan", a.out_dir)
    
    # Also generate for energy metric
    print("Generating enhanced 3D energy landscapes...")
    create_enhanced_side_by_side_3d(a, sf_limit, "energy", a.out_dir)
    
    print("Enhanced 3D visualization generation complete!")

if __name__ == "__main__":
    main()
