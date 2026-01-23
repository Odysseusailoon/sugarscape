#!/usr/bin/env python3
"""Plot wealth and moral score relationships."""

import json
import os
import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_latest_experiment(suffix):
    pattern = f"results/sugarscape/goal_survival_{suffix}/experiment_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)

def load_trade_csv(exp_dir):
    """Load trade history CSV."""
    trade_file = os.path.join(exp_dir, "debug", "trade_history.csv")
    if not os.path.exists(trade_file):
        return []
    trades = []
    with open(trade_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades

def load_moral_evals(exp_dir):
    """Load moral evaluations."""
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
            trades = load_trade_csv(exp_dir)
            moral_evals = load_moral_evals(exp_dir)
            all_data[exp_key] = {
                'name': exp_name,
                'color': color,
                'trades': trades,
                'moral_evals': moral_evals
            }
            print(f"{exp_key}: {len(trades)} trades, {len(moral_evals)} moral evals")
    
    # 1. Wealth evolution over time (top-left)
    ax1 = axes[0, 0]
    for exp_key, data in all_data.items():
        tick_wealth = defaultdict(list)
        for trade in data['trades']:
            try:
                tick = int(trade.get('tick', 0))
                wa = float(trade.get('welfare_a_before', 0))
                wb = float(trade.get('welfare_b_before', 0))
                if wa > 0:
                    tick_wealth[tick].append(wa)
                if wb > 0:
                    tick_wealth[tick].append(wb)
            except:
                pass
        
        if tick_wealth:
            ticks = sorted(tick_wealth.keys())
            means = [np.mean(tick_wealth[t]) for t in ticks]
            ax1.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=6)
    
    ax1.set_xlabel('Tick', fontsize=11)
    ax1.set_ylabel('Average Wealth (Welfare)', fontsize=11)
    ax1.set_title('Wealth Evolution Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Wealth vs Moral Score scatter (top-right)
    ax2 = axes[0, 1]
    for exp_key, data in all_data.items():
        # Build agent wealth from trades
        agent_wealth = defaultdict(list)
        for trade in data['trades']:
            try:
                agent_a_id = int(trade.get('agent_a_id', 0))
                agent_b_id = int(trade.get('agent_b_id', 0))
                wa = float(trade.get('welfare_a_before', 0))
                wb = float(trade.get('welfare_b_before', 0))
                if wa > 0:
                    agent_wealth[agent_a_id].append(wa)
                if wb > 0:
                    agent_wealth[agent_b_id].append(wb)
            except:
                pass
        
        # Build agent moral scores
        agent_moral = defaultdict(list)
        for e in data['moral_evals']:
            agent_id = e.get('agent_id')
            ext = e.get('external_overall', 0)
            if ext > 0 and agent_id:
                agent_moral[agent_id].append(ext)
        
        # Match
        wealth_list = []
        moral_list = []
        for agent_id in agent_wealth:
            if agent_id in agent_moral:
                wealth_list.append(np.mean(agent_wealth[agent_id]))
                moral_list.append(np.mean(agent_moral[agent_id]))
        
        if wealth_list and moral_list:
            ax2.scatter(wealth_list, moral_list, alpha=0.5, label=data['name'], 
                       color=data['color'], s=40, edgecolors='white', linewidth=0.5)
    
    ax2.set_xlabel('Average Wealth', fontsize=11)
    ax2.set_ylabel('Average External Moral Score', fontsize=11)
    ax2.set_title('Wealth vs Moral Score (per Agent)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Wealth change after trade (bottom-left)
    ax3 = axes[1, 0]
    for exp_key, data in all_data.items():
        tick_change = defaultdict(list)
        for trade in data['trades']:
            try:
                tick = int(trade.get('tick', 0))
                wa_before = float(trade.get('welfare_a_before', 0))
                wa_after = float(trade.get('welfare_a_after', 0))
                wb_before = float(trade.get('welfare_b_before', 0))
                wb_after = float(trade.get('welfare_b_after', 0))
                if wa_before > 0 and wa_after > 0:
                    tick_change[tick].append(wa_after - wa_before)
                if wb_before > 0 and wb_after > 0:
                    tick_change[tick].append(wb_after - wb_before)
            except:
                pass
        
        if tick_change:
            ticks = sorted(tick_change.keys())
            means = [np.mean(tick_change[t]) for t in ticks]
            ax3.plot(ticks, means, 'o-', label=data['name'], color=data['color'], linewidth=2, markersize=6)
    
    ax3.set_xlabel('Tick', fontsize=11)
    ax3.set_ylabel('Avg Wealth Change per Trade', fontsize=11)
    ax3.set_title('Trade Benefit Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Moral score vs wealth change correlation (bottom-right)
    ax4 = axes[1, 1]
    for exp_key, data in all_data.items():
        # Build agent-level data
        agent_wealth_change = defaultdict(list)
        agent_moral = defaultdict(list)
        
        for trade in data['trades']:
            try:
                agent_a_id = int(trade.get('agent_a_id', 0))
                agent_b_id = int(trade.get('agent_b_id', 0))
                wa_change = float(trade.get('welfare_a_after', 0)) - float(trade.get('welfare_a_before', 0))
                wb_change = float(trade.get('welfare_b_after', 0)) - float(trade.get('welfare_b_before', 0))
                agent_wealth_change[agent_a_id].append(wa_change)
                agent_wealth_change[agent_b_id].append(wb_change)
            except:
                pass
        
        for e in data['moral_evals']:
            agent_id = e.get('agent_id')
            ext = e.get('external_overall', 0)
            if ext > 0 and agent_id:
                agent_moral[agent_id].append(ext)
        
        # Match and plot
        changes = []
        morals = []
        for agent_id in agent_wealth_change:
            if agent_id in agent_moral and agent_wealth_change[agent_id]:
                changes.append(np.mean(agent_wealth_change[agent_id]))
                morals.append(np.mean(agent_moral[agent_id]))
        
        if changes and morals:
            ax4.scatter(changes, morals, alpha=0.5, label=data['name'], 
                       color=data['color'], s=40, edgecolors='white', linewidth=0.5)
    
    ax4.set_xlabel('Avg Wealth Change per Trade', fontsize=11)
    ax4.set_ylabel('Average External Moral Score', fontsize=11)
    ax4.set_title('Trade Benefit vs Moral Score', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = "results/sugarscape/comparison_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wealth_moral_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()
