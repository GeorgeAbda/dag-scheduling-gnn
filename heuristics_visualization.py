"""
Visualization of concurrency-aware heuristics for makespan and energy estimation.
Creates diagrams showing how the greedy ECT policy schedules tasks with concurrent execution.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300

def visualize_concurrent_scheduling():
    """
    Visualize the difference between sequential and concurrent task scheduling
    on a multi-core VM, showing impact on makespan and energy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Concurrency-Aware Greedy Heuristics: Makespan and Energy Estimation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # VM parameters
    vm_cores = 4
    vm_mem = 8  # GB
    vm_mips = 100
    p_idle = 50  # W
    p_peak = 200  # W
    
    # Task parameters
    # Task j: already scheduled
    task_j = {'name': 'Task j', 'cores': 2, 'mem': 4, 'length': 1000, 
              'start': 0, 'end': 10, 'color': '#FF6B6B'}
    # Task i: to be scheduled
    task_i = {'name': 'Task i', 'cores': 1, 'mem': 2, 'length': 1000, 
              'color': '#4ECDC4'}
    
    # ========== Sequential Scheduling (Top Row) ==========
    ax_seq_timeline = axes[0, 0]
    ax_seq_energy = axes[0, 1]
    
    # Sequential timeline
    ax_seq_timeline.set_xlim(0, 22)
    ax_seq_timeline.set_ylim(0, 5)
    ax_seq_timeline.set_xlabel('Time (s)', fontsize=11)
    ax_seq_timeline.set_ylabel('CPU Cores Used', fontsize=11)
    ax_seq_timeline.set_title('(a) Sequential Scheduling: Timeline', fontsize=12, fontweight='bold')
    ax_seq_timeline.grid(True, alpha=0.3, linestyle='--')
    
    # Task j
    rect_j_seq = Rectangle((task_j['start'], 0), task_j['end']-task_j['start'], task_j['cores'],
                            facecolor=task_j['color'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax_seq_timeline.add_patch(rect_j_seq)
    ax_seq_timeline.text(5, 1, task_j['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Task i (sequential: starts after j)
    task_i_seq_start = 10
    task_i_seq_end = 20
    rect_i_seq = Rectangle((task_i_seq_start, 0), task_i_seq_end-task_i_seq_start, task_i['cores'],
                            facecolor=task_i['color'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax_seq_timeline.add_patch(rect_i_seq)
    ax_seq_timeline.text(15, 0.5, task_i['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Capacity line
    ax_seq_timeline.axhline(y=vm_cores, color='red', linestyle='--', linewidth=2, label=f'VM Capacity ({vm_cores} cores)')
    ax_seq_timeline.legend(loc='upper right', fontsize=9)
    
    # Makespan annotation
    ax_seq_timeline.annotate('', xy=(20, 4.5), xytext=(0, 4.5),
                            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax_seq_timeline.text(10, 4.7, f'Makespan = 20s', ha='center', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Sequential energy
    ax_seq_energy.set_xlim(0, 22)
    ax_seq_energy.set_ylim(0, 250)
    ax_seq_energy.set_xlabel('Time (s)', fontsize=11)
    ax_seq_energy.set_ylabel('Power (W)', fontsize=11)
    ax_seq_energy.set_title('(b) Sequential Scheduling: Power Profile', fontsize=12, fontweight='bold')
    ax_seq_energy.grid(True, alpha=0.3, linestyle='--')
    
    # Power segments
    # Segment 1: [0,10], U=2/4=0.5
    u1 = 2/4
    p1 = p_idle + (p_peak - p_idle) * u1
    e1 = p1 * 10
    ax_seq_energy.fill_between([0, 10], 0, p1, color=task_j['color'], alpha=0.5, label=f'Task j only: U={u1:.2f}, P={p1:.1f}W')
    ax_seq_energy.text(5, p1/2, f'E₁={e1:.0f}J', ha='center', fontsize=9, fontweight='bold')
    
    # Segment 2: [10,20], U=1/4=0.25
    u2 = 1/4
    p2 = p_idle + (p_peak - p_idle) * u2
    e2 = p2 * 10
    ax_seq_energy.fill_between([10, 20], 0, p2, color=task_i['color'], alpha=0.5, label=f'Task i only: U={u2:.2f}, P={p2:.1f}W')
    ax_seq_energy.text(15, p2/2, f'E₂={e2:.0f}J', ha='center', fontsize=9, fontweight='bold')
    
    # Idle power line
    ax_seq_energy.axhline(y=p_idle, color='gray', linestyle=':', linewidth=1.5, label=f'Idle Power ({p_idle}W)')
    ax_seq_energy.axhline(y=p_peak, color='red', linestyle='--', linewidth=1.5, label=f'Peak Power ({p_peak}W)')
    
    total_e_seq = e1 + e2
    ax_seq_energy.text(21, 220, f'Total E = {total_e_seq:.0f}J', ha='right', fontsize=11, 
                      bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7), fontweight='bold')
    ax_seq_energy.legend(loc='upper left', fontsize=8)
    
    # ========== Concurrent Scheduling (Bottom Row) ==========
    ax_conc_timeline = axes[1, 0]
    ax_conc_energy = axes[1, 1]
    
    # Concurrent timeline
    ax_conc_timeline.set_xlim(0, 22)
    ax_conc_timeline.set_ylim(0, 5)
    ax_conc_timeline.set_xlabel('Time (s)', fontsize=11)
    ax_conc_timeline.set_ylabel('CPU Cores Used', fontsize=11)
    ax_conc_timeline.set_title('(c) Concurrent Scheduling: Timeline', fontsize=12, fontweight='bold')
    ax_conc_timeline.grid(True, alpha=0.3, linestyle='--')
    
    # Task j
    rect_j_conc = Rectangle((task_j['start'], 0), task_j['end']-task_j['start'], task_j['cores'],
                            facecolor=task_j['color'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax_conc_timeline.add_patch(rect_j_conc)
    ax_conc_timeline.text(5, 1, task_j['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Task i (concurrent: starts at 0)
    task_i_conc_start = 0
    task_i_conc_end = 10
    rect_i_conc = Rectangle((task_i_conc_start, task_j['cores']), task_i_conc_end-task_i_conc_start, task_i['cores'],
                            facecolor=task_i['color'], edgecolor='black', linewidth=1.5, alpha=0.7)
    ax_conc_timeline.add_patch(rect_i_conc)
    ax_conc_timeline.text(5, 2.5, task_i['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Capacity line
    ax_conc_timeline.axhline(y=vm_cores, color='red', linestyle='--', linewidth=2, label=f'VM Capacity ({vm_cores} cores)')
    ax_conc_timeline.legend(loc='upper right', fontsize=9)
    
    # Makespan annotation
    ax_conc_timeline.annotate('', xy=(10, 4.5), xytext=(0, 4.5),
                             arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax_conc_timeline.text(5, 4.7, f'Makespan = 10s', ha='center', fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Concurrent energy
    ax_conc_energy.set_xlim(0, 22)
    ax_conc_energy.set_ylim(0, 250)
    ax_conc_energy.set_xlabel('Time (s)', fontsize=11)
    ax_conc_energy.set_ylabel('Power (W)', fontsize=11)
    ax_conc_energy.set_title('(d) Concurrent Scheduling: Power Profile', fontsize=12, fontweight='bold')
    ax_conc_energy.grid(True, alpha=0.3, linestyle='--')
    
    # Power segment: [0,10], U=3/4=0.75
    u_conc = 3/4
    p_conc = p_idle + (p_peak - p_idle) * u_conc
    e_conc = p_conc * 10
    ax_conc_energy.fill_between([0, 10], 0, p_conc, color='purple', alpha=0.5, 
                                label=f'Both tasks: U={u_conc:.2f}, P={p_conc:.1f}W')
    ax_conc_energy.text(5, p_conc/2, f'E={e_conc:.0f}J', ha='center', fontsize=9, fontweight='bold')
    
    # Idle power line
    ax_conc_energy.axhline(y=p_idle, color='gray', linestyle=':', linewidth=1.5, label=f'Idle Power ({p_idle}W)')
    ax_conc_energy.axhline(y=p_peak, color='red', linestyle='--', linewidth=1.5, label=f'Peak Power ({p_peak}W)')
    
    savings = total_e_seq - e_conc
    savings_pct = (savings / total_e_seq) * 100
    ax_conc_energy.text(21, 220, f'Total E = {e_conc:.0f}J', ha='right', fontsize=11, 
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontweight='bold')
    ax_conc_energy.text(21, 190, f'Savings: {savings:.0f}J ({savings_pct:.1f}%)', ha='right', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontweight='bold')
    ax_conc_energy.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('concurrency_heuristics_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: concurrency_heuristics_comparison.png")
    plt.show()


def visualize_event_timeline():
    """
    Visualize the event-based timeline simulation used in EarliestFeasible computation.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle('Event-Based Timeline Simulation for Earliest Feasible Start Time', 
                 fontsize=14, fontweight='bold')
    
    # Timeline with events
    events = [
        (0, 'Start', 'Task A starts', 2, 3),
        (5, 'Complete', 'Task A completes', -2, -3),
        (3, 'Start', 'Task B starts', 1, 2),
        (8, 'Complete', 'Task B completes', -1, -2),
        (6, 'Start', 'Task C starts', 1, 1),
        (12, 'Complete', 'Task C completes', -1, -1),
    ]
    events_sorted = sorted(events, key=lambda x: x[0])
    
    # VM capacity
    vm_cores = 4
    vm_mem = 8
    
    # Plot timeline
    ax.set_xlim(-1, 15)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('CPU Cores Used', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Capacity line
    ax.axhline(y=vm_cores, color='red', linestyle='--', linewidth=2, label=f'VM Capacity ({vm_cores} cores)')
    
    # Simulate resource usage
    used_cores = 0
    time_points = [0]
    core_usage = [0]
    
    for t, event_type, desc, delta_cores, delta_mem in events_sorted:
        time_points.append(t)
        core_usage.append(used_cores)
        used_cores += delta_cores
        time_points.append(t)
        core_usage.append(used_cores)
    
    time_points.append(15)
    core_usage.append(used_cores)
    
    # Plot resource usage
    ax.plot(time_points, core_usage, color='blue', linewidth=2.5, label='CPU Cores Used')
    ax.fill_between(time_points, 0, core_usage, color='blue', alpha=0.2)
    
    # Mark events
    for t, event_type, desc, delta_cores, delta_mem in events_sorted:
        color = 'green' if event_type == 'Start' else 'orange'
        marker = '^' if event_type == 'Start' else 'v'
        ax.plot(t, used_cores + delta_cores if event_type == 'Complete' else used_cores, 
               marker=marker, markersize=10, color=color, zorder=5)
        ax.axvline(x=t, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Example: Check feasibility for new task at t=4
    new_task_ready = 4
    new_task_cores = 2
    new_task_mem = 2
    
    ax.axvline(x=new_task_ready, color='purple', linestyle='-', linewidth=2, 
              label=f'New task ready (t={new_task_ready})', alpha=0.7)
    
    # Find earliest feasible
    # At t=4: used_cores=3 (A:2, B:1), available=1 < 2 (not feasible)
    # At t=5: used_cores=1 (B:1), available=3 >= 2 (feasible!)
    ax.scatter([5], [1], s=200, color='lime', marker='*', edgecolor='black', linewidth=2, 
              label=f'Earliest feasible start (t=5)', zorder=10)
    
    # Annotate
    ax.annotate('Not feasible\n(only 1 core free)', xy=(4, 3), xytext=(4, 4.2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
    
    ax.annotate('Feasible!\n(3 cores free)', xy=(5, 1), xytext=(7, 2.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('event_timeline_simulation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: event_timeline_simulation.png")
    plt.show()


if __name__ == '__main__':
    print("Generating concurrency-aware heuristics visualizations...")
    visualize_concurrent_scheduling()
    visualize_event_timeline()
    print("Done!")
