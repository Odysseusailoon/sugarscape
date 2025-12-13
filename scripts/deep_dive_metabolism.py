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
        
        if not os.path.exists(run_dir): continue
            
        subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("experiment_")]
        if not subdirs: continue
            
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
                    'Efficiency': utility / a['age'] if a['age'] > 0 else 0,
                    'Metabolism_Total': m_s + m_p,
                    'Metabolism_Sugar': m_s,
                    'Metabolism_Spice': m_p
                })
        except FileNotFoundError:
            continue
                
    return pd.DataFrame(all_data)

def analyze_conservative_metabolism(df, output_dir):
    """Analyze if Type A's success is due to low metabolism."""
    
    # Filter for Type A
    df_a = df[df['Persona'] == 'A']
    
    # 1. Metabolism vs Efficiency Scatter
    plt.figure(figsize=(10, 6))
    # Add slight jitter to x to see density
    sns.regplot(data=df_a, x='Metabolism_Total', y='Efficiency', x_jitter=0.2, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
    plt.title('Correlation: Total Metabolism vs Efficiency (Type A)')
    plt.xlabel('Total Metabolism (Sugar + Spice)')
    plt.ylabel('Efficiency (Utility / Age)')
    plt.savefig(os.path.join(output_dir, "conservative_metabolism_corr.png"))
    plt.close()
    
    # 2. Compare Metabolism Distributions across Efficiency Groups
    median_eff = df_a['Efficiency'].median()
    df_a['Group'] = np.where(df_a['Efficiency'] > median_eff, 'High Efficiency', 'Low Efficiency')
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_a, x='Metabolism_Total', hue='Group', element='step', stat='density', common_norm=False, bins=8)
    plt.title('Metabolism Distribution: High vs Low Efficiency Conservatives')
    plt.xlabel('Total Metabolism')
    plt.savefig(os.path.join(output_dir, "conservative_metabolism_dist.png"))
    plt.close()
    
    # Generate Report
    mean_metab_all = df_a['Metabolism_Total'].mean()
    mean_metab_high = df_a[df_a['Group'] == 'High Efficiency']['Metabolism_Total'].mean()
    mean_metab_low = df_a[df_a['Group'] == 'Low Efficiency']['Metabolism_Total'].mean()
    corr = df_a['Metabolism_Total'].corr(df_a['Efficiency'])
    
    report = f"""# Deep Dive: The Genetic Lottery of Conservatives

## Hypothesis
User suspects that Type A (Conservative) agents are efficient not because of their strategy, but because they survived the "Genetic Lottery" (i.e., they happen to have the lowest metabolism).

## Data Analysis (SugarSpice Environment)
- **Correlation (Metabolism vs Efficiency)**: {corr:.3f} (Negative = Lower metabolism leads to higher efficiency)
- **Mean Metabolism (High Efficiency Group)**: {mean_metab_high:.2f}
- **Mean Metabolism (Low Efficiency Group)**: {mean_metab_low:.2f}
- **Overall Mean Metabolism**: {mean_metab_all:.2f}

## Conclusion
Does low metabolism explain the success?
If correlation is strongly negative (e.g., < -0.5), then yes, they are just "lucky dieters".
If correlation is weak, then the strategy matters.
"""
    with open(os.path.join(output_dir, "conservative_metabolism_report.md"), 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_dir}")

def main():
    output_dir = "results/sugarscape/conservative_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_batch_data(50)
    if len(df) == 0:
        print("No data found.")
        return
        
    analyze_conservative_metabolism(df, output_dir)

if __name__ == "__main__":
    main()
