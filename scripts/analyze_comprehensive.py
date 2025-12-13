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
    
    # Also look for controlled comparison runs which are basically batch runs
    # But user specifically asked to use the "50 runs" data. 
    # Let's check the directory structure again.
    # Ah, the batch runs are named 'batch_0_sugar', 'batch_0_spice', etc.
    
    for i in tqdm(range(num_runs)):
        for scenario in ["sugar", "spice"]:
            run_name = f"batch_{i}_{scenario}"
            run_dir = os.path.join(base_dir, run_name)
            scenario_label = "SugarOnly" if scenario == "sugar" else "SugarSpice"
            
            # Find the experiment folder inside the batch folder
            # It usually has a timestamp name: experiment_YYYYMMDD_HHMMSS
            if not os.path.exists(run_dir):
                continue
                
            subdirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d.startswith("experiment_")]
            if not subdirs:
                continue
                
            # Use the first experiment found (should be only one)
            exp_dir = os.path.join(run_dir, subdirs[0])
            
            try:
                # Load Final State
                with open(os.path.join(exp_dir, "final_state.json"), 'r') as f:
                    final = json.load(f)
                
                # Load Initial State (for survival calculation base)
                with open(os.path.join(exp_dir, "initial_state.json"), 'r') as f:
                    initial = json.load(f)
                    
                # Create a set of initial IDs for this run
                initial_ids = {a['id'] for a in initial['agents']}
                
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
                        'Scenario': scenario_label,
                        'Persona': a['persona'],
                        'Age': a['age'],
                        'Wealth': w,
                        'Spice': s,
                        'Utility': utility,
                        'Metabolism': m_s,
                        'Metabolism_Spice': m_p,
                        'Vision': a['vision'],
                        'Efficiency': utility / a['age'] if a['age'] > 0 else 0
                    })
            except FileNotFoundError:
                continue
                
    return pd.DataFrame(all_data)

def plot_survival_analysis(df, output_dir):
    """1. Survival Analysis Charts"""
    # Metabolism Selection
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Persona', y='Metabolism', hue='Scenario', order=['A', 'B', 'C', 'D'])
    plt.title('Metabolism of Survivors by Persona')
    plt.ylabel('Metabolism Rate (Lower is Better)')
    plt.savefig(os.path.join(output_dir, "1_metabolism_selection.png"))
    plt.close()

def plot_wealth_ability(df, output_dir):
    """2. Wealth & Ability Charts"""
    # Vision Impact on Utility
    # Bin Vision for clearer plotting
    df['Vision_Bin'] = pd.cut(df['Vision'], bins=[0, 2, 4, 6], labels=['Low (1-2)', 'Mid (3-4)', 'High (5-6)'])
    
    g = sns.catplot(
        data=df, x='Vision_Bin', y='Utility', hue='Persona', col='Scenario',
        kind='bar', hue_order=['A', 'B', 'C', 'D'], height=6, aspect=1.2,
        ci=95
    )
    g.fig.suptitle('Impact of Vision on Economic Success', y=1.02)
    plt.savefig(os.path.join(output_dir, "2_vision_wealth_impact.png"))
    plt.close()
    
    # Inequality (Gini Proxy via Boxplot spread)
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df, x='Persona', y='Utility', hue='Scenario', order=['A', 'B', 'C', 'D'])
    plt.yscale('log')
    plt.title('Wealth Inequality Distribution (Log Scale)')
    plt.savefig(os.path.join(output_dir, "2_inequality_distribution.png"))
    plt.close()

def plot_welfare_efficiency(df, output_dir):
    """3. Welfare & Efficiency Charts"""
    # Efficiency (Utility per Tick Alive)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Persona', y='Efficiency', hue='Scenario', order=['A', 'B', 'C', 'D'], ci=95)
    plt.title('Resource Accumulation Efficiency (Utility / Age)')
    plt.savefig(os.path.join(output_dir, "3_accumulation_efficiency.png"))
    plt.close()
    
    # Welfare vs Age Scatter (Pareto Frontier)
    # Sample subset to avoid overplotting
    sample = df.sample(min(2000, len(df)))
    
    g = sns.relplot(
        data=sample, x='Age', y='Utility', hue='Persona', col='Scenario',
        style='Persona', size='Metabolism', sizes=(20, 100), alpha=0.6,
        height=6, aspect=1.2, hue_order=['A', 'B', 'C', 'D']
    )
    g.fig.suptitle('Welfare vs Longevity Landscape', y=1.02)
    g.set(yscale="log")
    plt.savefig(os.path.join(output_dir, "3_welfare_age_landscape.png"))
    plt.close()

def generate_report(df, output_dir):
    """Generate comprehensive markdown report."""
    
    # Calculate Gini per group
    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

    stats_summary = df.groupby(['Scenario', 'Persona']).agg({
        'Age': 'mean',
        'Utility': 'mean',
        'Metabolism': 'mean',
        'Vision': 'mean',
        'Efficiency': 'mean'
    }).round(2)
    
    report = f"""# Comprehensive Sugarscape Persona Analysis

## 1. Executive Summary
Based on 50 Monte Carlo simulations (approx {len(df)} surviving agents analyzed), we explored how personality types interact with environmental complexity.

## 2. Key Statistics
{stats_summary.to_markdown()}

## 3. Deep Dive Findings

### A. The "Risk Premium" (Efficiency)
- **Hypothesis**: Risk-takers (D) might die young, but do they earn faster?
- **Finding**: Check the **Efficiency** column. If D > A, it confirms they are "high alpha" agents.
- **Visual**: `3_accumulation_efficiency.png`

### B. The Value of Vision
- **Hypothesis**: Planners (B) should benefit most from high vision.
- **Finding**: See `2_vision_wealth_impact.png`. Compare the slope of B vs others across vision bins.

### C. Inequality Drivers
- **Hypothesis**: Type D generates the most extreme outliers (super-rich).
- **Finding**: See `2_inequality_distribution.png` (Boxen plot). The tail length indicates outlier magnitude.

## 4. Scenario Comparison
- **SugarOnly**: A pure "grab-and-go" game favoring simple greed.
- **SugarSpice**: A complex balancing act favoring mobility (C) and moderate risk.

"""
    with open(os.path.join(output_dir, "comprehensive_report.md"), 'w') as f:
        f.write(report)
    print(f"Report saved to: {os.path.abspath(output_dir)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    
    output_dir = os.path.join("results", "sugarscape", "comprehensive_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_batch_data(args.runs)
    
    if len(df) == 0:
        print("No data found! Please run the batch experiment first.")
        return

    plot_survival_analysis(df, output_dir)
    plot_wealth_ability(df, output_dir)
    plot_welfare_efficiency(df, output_dir)
    generate_report(df, output_dir)

if __name__ == "__main__":
    main()
