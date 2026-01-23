#!/usr/bin/env python3
"""Plot moral evaluation histograms for comparison."""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def load_moral_evals(exp_dir):
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
    pattern = f"results/sugarscape/goal_survival_{suffix}/experiment_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def main():
    experiments = {
        'baseline': ('Baseline', '#2ecc71'),
        'no_survival': ('No Survival', '#3498db'), 
        'abundant': ('Abundant', '#9b59b6'),
        'no_memory': ('No Memory', '#e74c3c')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    all_scores = {}
    
    for i, (exp_key, (exp_name, color)) in enumerate(experiments.items()):
        ax = axes[i]
        exp_dir = get_latest_experiment(exp_key)
        
        if exp_dir:
            evals = load_moral_evals(exp_dir)
            ext_scores = [e.get('external_overall', 0) for e in evals if e.get('external_overall', 0) > 0]
            self_scores = [e.get('self_overall', 50) for e in evals if e.get('external_overall', 0) > 0]
            
            all_scores[exp_key] = ext_scores
            
            if ext_scores:
                # Histogram
                ax.hist(ext_scores, bins=20, alpha=0.7, color=color, edgecolor='white', label='External')
                ax.hist(self_scores, bins=20, alpha=0.4, color='gray', edgecolor='white', label='Self')
                ax.axvline(x=np.mean(ext_scores), color=color, linestyle='--', linewidth=2, label=f'Ext Mean: {np.mean(ext_scores):.1f}')
                ax.axvline(x=50, color='black', linestyle=':', alpha=0.5, label='Neutral (50)')
                
                ax.set_xlabel('Moral Score')
                ax.set_ylabel('Count')
                ax.set_title(f'{exp_name} (n={len(ext_scores)})')
                ax.legend(fontsize=8)
                ax.set_xlim(0, 100)
    
    plt.suptitle('Moral Score Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = "results/sugarscape/comparison_plots/moral_histogram.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Overlay comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for exp_key, (exp_name, color) in experiments.items():
        if exp_key in all_scores and all_scores[exp_key]:
            scores = all_scores[exp_key]
            ax2.hist(scores, bins=25, alpha=0.4, color=color, label=f'{exp_name} (μ={np.mean(scores):.1f})')
    
    ax2.axvline(x=50, color='black', linestyle='--', alpha=0.7, label='Neutral')
    ax2.set_xlabel('External Moral Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('External Moral Score Distribution - All Experiments', fontsize=14)
    ax2.legend()
    ax2.set_xlim(0, 100)
    
    output_file2 = "results/sugarscape/comparison_plots/moral_overlay.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")

if __name__ == "__main__":
    main()
