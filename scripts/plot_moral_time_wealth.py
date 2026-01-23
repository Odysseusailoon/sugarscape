#!/usr/bin/env python3
"""Plot moral score vs time and wealth relationships."""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_data(exp_dir):
    """Load moral evals and trade data."""
    moral_evals = []
    moral_file = os.path.join(exp_dir, "debug", "moral_evals.jsonl")
    if os.path.exists(moral_file):
        with open(moral_file, 'r') as f:
            for line in f:
                try:
                    moral_evals.append(json.loads(line))
                except:
                    pass
    
    # Load trade data for wealth info
    trades = []
    trade_file = os.path.join(exp_dir, "debug", "trade_dialogues.jsonl")
    if os.path.exists(trade_file):
        with open(trade_file, 'r') as f:
            for line in f:
                try:
                    trades.append(json.loads(line))
                except:
                    pass
    
    return moral_evals, trades

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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    all_data = {}
    for exp_key, (exp_name, color) in experiments.items():
        exp_dir = get_latest_experiment(exp_key)
        if exp_dir:
            moral_evals, trades = load_data(exp_dir)
            all_data[exp_key] = {
                'name': exp_name,
                'color': color,
                'moral_evals': moral_evals,
                'trades': trades
            }
    
    # 1. Moral score over time (top-left)
    ax1 = axes[0, 0]
    for exp_key, data in all_data.items():
        tick_scores = defaultdict(list)
        for e in data['moral_evals']:
            tick = e.get('tick', 0)
            ext = e.get('external_overall', 0)
            if ext > 0:
                tick_scores[tick].append(ext)
        
        if tick_scores:
            ticks = sorted(tick_scores.keys())
            means = [np.mean(tick_scores[t]) for t in ticks]
            stds = [np.std(tick_scores[t]) for t in ticks]
            ax1.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=6)
            ax1.fill_between(ticks, 
                           [m-s for m,s in zip(means, stds)],
                           [m+s for m,s in zip(means, stds)],
                           alpha=0.2, color=data['color'])
    
    ax1.set_xlabel('Tick', fontsize=11)
    ax1.set_ylabel('External Moral Score', fontsize=11)
    ax1.set_title('Moral Score Evolution Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # 2. Self score over time (top-right)
    ax2 = axes[0, 1]
    for exp_key, data in all_data.items():
        tick_scores = defaultdict(list)
        for e in data['moral_evals']:
            tick = e.get('tick', 0)
            self_score = e.get('self_overall', 50)
            if e.get('external_overall', 0) > 0:  # Only count valid entries
                tick_scores[tick].append(self_score)
        
        if tick_scores:
            ticks = sorted(tick_scores.keys())
            means = [np.mean(tick_scores[t]) for t in ticks]
            ax2.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=6)
    
    ax2.set_xlabel('Tick', fontsize=11)
    ax2.set_ylabel('Self Moral Score', fontsize=11)
    ax2.set_title('Self-Assessment Evolution Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Gap (External - Self) over time (bottom-left)
    ax3 = axes[1, 0]
    for exp_key, data in all_data.items():
        tick_gaps = defaultdict(list)
        for e in data['moral_evals']:
            tick = e.get('tick', 0)
            ext = e.get('external_overall', 0)
            self_score = e.get('self_overall', 50)
            if ext > 0:
                tick_gaps[tick].append(ext - self_score)
        
        if tick_gaps:
            ticks = sorted(tick_gaps.keys())
            means = [np.mean(tick_gaps[t]) for t in ticks]
            ax3.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=6)
    
    ax3.set_xlabel('Tick', fontsize=11)
    ax3.set_ylabel('Gap (External - Self)', fontsize=11)
    ax3.set_title('Self-Assessment Gap Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Moral score by interaction type over time (bottom-right)
    ax4 = axes[1, 1]
    
    # Focus on post_trade_reflection for baseline
    if 'baseline' in all_data:
        data = all_data['baseline']
        type_tick_scores = defaultdict(lambda: defaultdict(list))
        for e in data['moral_evals']:
            tick = e.get('tick', 0)
            itype = e.get('interaction_type', 'unknown')
            ext = e.get('external_overall', 0)
            if ext > 0:
                type_tick_scores[itype][tick].append(ext)
        
        type_colors = {
            'baseline_questionnaire': '#95a5a6',
            'post_trade_reflection': '#2ecc71',
            'event_identity_review': '#3498db',
            'end_of_life_report': '#e74c3c'
        }
        
        for itype, tick_scores in type_tick_scores.items():
            if tick_scores:
                ticks = sorted(tick_scores.keys())
                means = [np.mean(tick_scores[t]) for t in ticks]
                color = type_colors.get(itype, '#7f8c8d')
                label = itype.replace('_', ' ').title()
                ax4.plot(ticks, means, 'o-', label=label, color=color, linewidth=2, markersize=5)
    
    ax4.set_xlabel('Tick', fontsize=11)
    ax4.set_ylabel('External Moral Score', fontsize=11)
    ax4.set_title('Baseline: Score by Interaction Type', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = "results/sugarscape/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "moral_time_evolution.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Create wealth vs moral score plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get wealth data from trades
    ax5 = axes2[0]
    ax6 = axes2[1]
    
    for exp_key, data in all_data.items():
        # Extract wealth from trade data
        agent_wealth_moral = defaultdict(lambda: {'wealth': [], 'moral': []})
        
        for trade in data['trades']:
            if 'initiator_welfare_before' in trade:
                agent_id = trade.get('initiator_id')
                wealth = trade.get('initiator_welfare_before', 0)
                if agent_id and wealth > 0:
                    agent_wealth_moral[agent_id]['wealth'].append(wealth)
            if 'responder_welfare_before' in trade:
                agent_id = trade.get('responder_id')
                wealth = trade.get('responder_welfare_before', 0)
                if agent_id and wealth > 0:
                    agent_wealth_moral[agent_id]['wealth'].append(wealth)
        
        # Match with moral scores
        for e in data['moral_evals']:
            agent_id = e.get('agent_id')
            ext = e.get('external_overall', 0)
            if agent_id in agent_wealth_moral and ext > 0:
                agent_wealth_moral[agent_id]['moral'].append(ext)
        
        # Aggregate per agent
        wealth_list = []
        moral_list = []
        for agent_id, vals in agent_wealth_moral.items():
            if vals['wealth'] and vals['moral']:
                wealth_list.append(np.mean(vals['wealth']))
                moral_list.append(np.mean(vals['moral']))
        
        if wealth_list and moral_list:
            ax5.scatter(wealth_list, moral_list, alpha=0.5, label=data['name'], color=data['color'], s=30)
    
    ax5.set_xlabel('Average Wealth', fontsize=11)
    ax5.set_ylabel('Average External Moral Score', fontsize=11)
    ax5.set_title('Wealth vs Moral Score (per Agent)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    
    # Wealth evolution from trades
    for exp_key, data in all_data.items():
        tick_wealth = defaultdict(list)
        for trade in data['trades']:
            tick = trade.get('tick', 0)
            if 'initiator_welfare_before' in trade:
                tick_wealth[tick].append(trade['initiator_welfare_before'])
            if 'responder_welfare_before' in trade:
                tick_wealth[tick].append(trade['responder_welfare_before'])
        
        if tick_wealth:
            ticks = sorted(tick_wealth.keys())
            means = [np.mean(tick_wealth[t]) for t in ticks]
            ax6.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=5)
    
    ax6.set_xlabel('Tick', fontsize=11)
    ax6.set_ylabel('Average Wealth', fontsize=11)
    ax6.set_title('Wealth Evolution Over Time', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = os.path.join(output_dir, "moral_wealth_relation.png")
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file2}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for exp_key, data in all_data.items():
        evals = data['moral_evals']
        ext_scores = [e.get('external_overall', 0) for e in evals if e.get('external_overall', 0) > 0]
        self_scores = [e.get('self_overall', 50) for e in evals if e.get('external_overall', 0) > 0]
        if ext_scores:
            gap = np.mean(ext_scores) - np.mean(self_scores)
            print(f"{data['name']:15} | Ext: {np.mean(ext_scores):.1f} | Self: {np.mean(self_scores):.1f} | Gap: {gap:+.1f}")

if __name__ == "__main__":
    main()
