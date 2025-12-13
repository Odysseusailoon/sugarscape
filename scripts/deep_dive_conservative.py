import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# Set style
sns.set_theme(style="whitegrid")

def load_batch_data(num_runs=50):
    """Load data from all batch runs."""
    all_data = []
    base_dir = "results/sugarscape"
    
    print(f"Loading data from {num_runs} runs...")
    
    for i in tqdm(range(num_runs)):
        run_name = f"batch_{i}_spice" # Focus on SugarSpice
        run_dir = os.path.join(base_dir, run_name)
        
        # Find the experiment folder
        if not os.path.exists(run_dir):
            continue
            
        subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("experiment_")]
        if not subdirs:
            continue
            
        exp_dir = os.path.join(run_dir, subdirs[0])
        
        try:
            with open(os.path.join(exp_dir, "final_state.json"), 'r') as f:
                final = json.load(f)
            
            for a in final['agents']:
                # Calculate Utility
                w = a['wealth']
                s = a.get('spice', 0)
                m_s = a['metabolism']
                m_p = a.get('metabolism_spice', 0)
                
                utility = float(w)
                if m_p > 0 and (m_s + m_p) > 0:
                    m_total = m_s + m_p
                    utility = (w ** (m_s/m_total)) * (s ** (m_p/m_total))
                
                all_data.append({
                    'RunID': i,
                    'Persona': a['persona'],
                    'Age': a['age'],
                    'Utility': utility,
                    'Vision': a['vision'],
                    'Metabolism': m_s,
                    'Efficiency': utility / a['age'] if a['age'] > 0 else 0
                })
        except FileNotFoundError:
            continue
                
    return pd.DataFrame(all_data)

def analyze_conservative_vision(df, output_dir):
    """Deep dive into Type A vision in SugarSpice."""
    
    # Filter for Type A
    df_a = df[df['Persona'] == 'A']
    
    # 1. Vision Distribution Comparison
    # Compare High Efficiency A vs Low Efficiency A
    median_eff = df_a['Efficiency'].median()
    df_a['Group'] = np.where(df_a['Efficiency'] > median_eff, 'High Efficiency (Top 50%)', 'Low Efficiency (Bottom 50%)')
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_a, x='Vision', hue='Group', element='step', stat='density', common_norm=False, bins=6)
    plt.title('Vision Distribution of Conservative Agents (SugarSpice)')
    plt.xlabel('Vision Range')
    plt.savefig(os.path.join(output_dir, "conservative_vision_dist.png"))
    plt.close()
    
    # 2. Vision vs Efficiency Scatter
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_a, x='Vision', y='Efficiency', x_jitter=0.2, scatter_kws={'alpha':0.3})
    plt.title('Correlation: Vision vs Efficiency (Type A)')
    plt.savefig(os.path.join(output_dir, "conservative_vision_efficiency_corr.png"))
    plt.close()
    
    # Generate Report
    mean_vision_all = df_a['Vision'].mean()
    mean_vision_high = df_a[df_a['Group'] == 'High Efficiency (Top 50%)']['Vision'].mean()
    corr = df_a['Vision'].corr(df_a['Efficiency'])
    
    report = f"""# Deep Dive: The Conservative Efficiency Paradox

## Context
In the complex **SugarSpice** environment, Conservative (Type A) agents achieved the highest resource accumulation efficiency (`Utility/Age = 1.88`), outperforming even the Risk-takers.

## Vision Analysis
- **Overall Mean Vision**: {mean_vision_all:.2f}
- **High Efficiency Group Mean Vision**: {mean_vision_high:.2f}
- **Correlation (Vision vs Efficiency)**: {corr:.3f}

## Interpretation
Is their efficiency driven by high vision?
If correlation is low, it suggests their strategy (safety first) is robust regardless of vision.
If correlation is high, it suggests only "Eagle-eyed Conservatives" are the true Pareto optimal agents.
"""
    with open(os.path.join(output_dir, "conservative_deep_dive.md"), 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_dir}")

def main():
    output_dir = "results/sugarscape/conservative_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_batch_data(50)
    if len(df) == 0:
        print("No data found.")
        return
        
    analyze_conservative_vision(df, output_dir)

if __name__ == "__main__":
    main()
