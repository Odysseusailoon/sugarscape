#!/usr/bin/env python3
"""Plot moral evaluation comparison across ablation experiments."""

import json
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_moral_evals(exp_dir):
    """Load moral evaluations from experiment directory."""
    moral_file = os.path.join(exp_dir, "debug", "moral_evals.jsonl")
    if not os.path.exists(moral_file):
        return []
    
    evals = []
    with open(moral_file, 'r') as f:
        for line in f:
            try:
                evals.append(json.loads(line))
            except:
                pass
    return evals

def get_latest_experiment(suffix):
    """Get latest experiment directory for given suffix."""
    pattern = f"results/sugarscape/goal_survival_{suffix}/experiment_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def main():
    experiments = {
        'baseline': 'Baseline (Control)',
        'no_survival': 'No Survival Pressure',
        'abundant': 'Abundant Resources',
        'no_memory': 'No Social Memory'
    }
    
    colors = {
        'baseline': '#2ecc71',
        'no_survival': '#3498db', 
        'abundant': '#9b59b6',
        'no_memory': '#e74c3c'
    }
    
    # Load data
    all_data = {}
    for exp_key, exp_name in experiments.items():
        exp_dir = get_latest_experiment(exp_key)
        if exp_dir:
            evals = load_moral_evals(exp_dir)
            all_data[exp_key] = {
                'name': exp_name,
                'evals': evals,
                'dir': exp_dir
            }
            print(f"{exp_key}: {len(evals)} moral evaluations")
    
    if not all_data:
        print("No data found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. External score distribution comparison (top-left)
    ax1 = axes[0, 0]
    ext_scores = {}
    for exp_key, data in all_data.items():
        scores = [e.get('external_overall', 0) for e in data['evals'] if e.get('external_overall', 0) > 0]
        if scores:
            ext_scores[exp_key] = scores
    
    if ext_scores:
        positions = list(range(len(ext_scores)))
        bp = ax1.boxplot([ext_scores[k] for k in ext_scores.keys()], 
                         positions=positions, widths=0.6, patch_artist=True)
        for i, (patch, key) in enumerate(zip(bp['boxes'], ext_scores.keys())):
            patch.set_facecolor(colors[key])
            patch.set_alpha(0.7)
        ax1.set_xticks(positions)
        ax1.set_xticklabels([all_data[k]['name'] for k in ext_scores.keys()], rotation=15, ha='right')
        ax1.set_ylabel('External Moral Score')
        ax1.set_title('External Moral Score Distribution')
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
        ax1.legend()
    
    # 2. Self vs External gap distribution (top-right)
    ax2 = axes[0, 1]
    gaps = {}
    for exp_key, data in all_data.items():
        gap_list = []
        for e in data['evals']:
            ext = e.get('external_overall', 0)
            self_score = e.get('self_overall', 50)
            if ext > 0:
                gap_list.append(ext - self_score)
        if gap_list:
            gaps[exp_key] = gap_list
    
    if gaps:
        positions = list(range(len(gaps)))
        bp = ax2.boxplot([gaps[k] for k in gaps.keys()], 
                         positions=positions, widths=0.6, patch_artist=True)
        for i, (patch, key) in enumerate(zip(bp['boxes'], gaps.keys())):
            patch.set_facecolor(colors[key])
            patch.set_alpha(0.7)
        ax2.set_xticks(positions)
        ax2.set_xticklabels([all_data[k]['name'] for k in gaps.keys()], rotation=15, ha='right')
        ax2.set_ylabel('External - Self Score')
        ax2.set_title('Self-Assessment Gap (External - Self)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Score by interaction type (bottom-left)
    ax3 = axes[1, 0]
    type_scores = defaultdict(lambda: defaultdict(list))
    for exp_key, data in all_data.items():
        for e in data['evals']:
            itype = e.get('interaction_type', 'unknown')
            ext = e.get('external_overall', 0)
            if ext > 0:
                type_scores[itype][exp_key].append(ext)
    
    if type_scores:
        itypes = list(type_scores.keys())
        x = np.arange(len(itypes))
        width = 0.2
        
        for i, exp_key in enumerate(all_data.keys()):
            means = []
            for itype in itypes:
                scores = type_scores[itype].get(exp_key, [])
                means.append(np.mean(scores) if scores else 0)
            ax3.bar(x + i*width, means, width, label=all_data[exp_key]['name'], 
                   color=colors[exp_key], alpha=0.8)
        
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels([t.replace('_', '\n') for t in itypes], fontsize=8)
        ax3.set_ylabel('Avg External Score')
        ax3.set_title('External Score by Interaction Type')
        ax3.legend(fontsize=8)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Score evolution over time (bottom-right)
    ax4 = axes[1, 1]
    for exp_key, data in all_data.items():
        tick_scores = defaultdict(list)
        for e in data['evals']:
            tick = e.get('tick', 0)
            ext = e.get('external_overall', 0)
            if ext > 0:
                tick_scores[tick].append(ext)
        
        if tick_scores:
            ticks = sorted(tick_scores.keys())
            means = [np.mean(tick_scores[t]) for t in ticks]
            ax4.plot(ticks, means, 'o-', label=all_data[exp_key]['name'], 
                    color=colors[exp_key], alpha=0.8, markersize=4)
    
    ax4.set_xlabel('Tick')
    ax4.set_ylabel('Avg External Score')
    ax4.set_title('Moral Score Evolution Over Time')
    ax4.legend(fontsize=8)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    output_dir = "results/sugarscape/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "moral_eval_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    
    # Print summary stats
    print("\n=== SUMMARY STATISTICS ===")
    for exp_key, data in all_data.items():
        evals = data['evals']
        ext_scores = [e.get('external_overall', 0) for e in evals if e.get('external_overall', 0) > 0]
        if ext_scores:
            print(f"\n{data['name']}:")
            print(f"  N={len(ext_scores)}, Mean={np.mean(ext_scores):.1f}, Std={np.std(ext_scores):.1f}")
            print(f"  Min={min(ext_scores):.1f}, Max={max(ext_scores):.1f}")

if __name__ == "__main__":
    main()
