import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import random
from tqdm import tqdm
from scipy import stats

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig

def run_single_pair(run_id, seed):
    """Run one pair of experiments (SugarOnly vs SugarSpice) with same seed."""
    
    # Common Config
    base_config = {
        "seed": seed,
        "initial_population": 250,
        "max_ticks": 500,
        "width": 50,
        "height": 50,
        "sugar_growback_rate": 1,
        "initial_wealth_range": (5, 25),
        "metabolism_range": (1, 4),
        "vision_range": (1, 6),
        "max_age_range": (60, 100),
        "enable_personas": True
    }
    
    # 1. SugarOnly
    config_sugar = SugarscapeConfig(
        **base_config,
        enable_spice=False
    )
    sim_sugar = SugarSimulation(config=config_sugar, experiment_name=f"batch_{run_id}_sugar")
    sim_sugar.run()
    
    # 2. SugarSpice
    config_spice = SugarscapeConfig(
        **base_config,
        enable_spice=True,
        max_spice_capacity=4,
        spice_growback_rate=1,
        initial_spice_range=(5, 25),
        metabolism_spice_range=(1, 4)
    )
    sim_spice = SugarSimulation(config=config_spice, experiment_name=f"batch_{run_id}_spice")
    sim_spice.run()
    
    return sim_sugar.logger.run_dir, sim_spice.logger.run_dir

def extract_agent_data(run_dir, run_id, scenario):
    with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
        data = json.load(f)
        
    rows = []
    for a in data['agents']:
        # Calculate Utility
        w = a['wealth']
        s = a.get('spice', 0)
        m_s = a['metabolism']
        m_p = a.get('metabolism_spice', 0)
        
        utility = float(w)
        if m_p > 0:
            m_total = m_s + m_p
            if m_total > 0:
                utility = (w ** (m_s/m_total)) * (s ** (m_p/m_total))
        
        rows.append({
            'RunID': run_id,
            'Scenario': scenario,
            'Persona': a['persona'],
            'Age': a['age'],
            'Wealth': w,
            'Utility': utility
        })
    return rows

def run_batch(num_runs=50):
    print(f"Starting Batch Experiment: {num_runs} runs...")
    
    all_data = []
    
    # Use a fixed master seed to generate run seeds for reproducibility of the batch
    master_rng = random.Random(999)
    
    for i in tqdm(range(num_runs)):
        run_seed = master_rng.randint(1000, 999999)
        dir_sugar, dir_spice = run_single_pair(i, run_seed)
        
        all_data.extend(extract_agent_data(dir_sugar, i, "SugarOnly"))
        all_data.extend(extract_agent_data(dir_spice, i, "SugarSpice"))
        
    return pd.DataFrame(all_data)

def analyze_batch_results(df):
    output_dir = os.path.join("results", "sugarscape", "batch_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating Statistics...")
    
    # Group by Scenario and Persona
    stats_df = df.groupby(['Scenario', 'Persona'])[['Age', 'Utility']].agg(['mean', 'std', 'count']).reset_index()
    
    # 1. Visualization: Survival with CI
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Persona', y='Age', hue='Scenario', order=['A', 'B', 'C', 'D'], ci=95, capsize=.1)
    plt.title(f'Survival Analysis (N={len(df["RunID"].unique())//2} Runs)')
    plt.ylabel('Mean Age (Years)')
    plt.savefig(os.path.join(output_dir, "batch_survival.png"))
    plt.close()
    
    # 2. Visualization: Wealth with CI
    plt.figure(figsize=(10, 6))
    # Use Log scale for wealth visualization? Or standard?
    # Wealth is usually power law, so log scale helps visualization but barplot mean is sensitive to outliers.
    # Let's use standard scale for mean comparison.
    sns.barplot(data=df, x='Persona', y='Utility', hue='Scenario', order=['A', 'B', 'C', 'D'], ci=95, capsize=.1)
    plt.title(f'Economic Success (N={len(df["RunID"].unique())//2} Runs)')
    plt.ylabel('Mean Utility')
    plt.savefig(os.path.join(output_dir, "batch_wealth.png"))
    plt.close()
    
    # 3. Statistical Testing (T-test)
    report_lines = ["# Batch Experiment Statistical Report", "", f"Total Runs: {len(df['RunID'].unique())//2} pairs", ""]
    
    personas = ['A', 'B', 'C', 'D']
    
    report_lines.append("## 1. Survival Analysis (Age)")
    report_lines.append("| Persona | Scenario | Mean Age | Std Dev |")
    report_lines.append("|---|---|---|---|")
    for p in personas:
        for s in ['SugarOnly', 'SugarSpice']:
            sub = df[(df['Persona']==p) & (df['Scenario']==s)]
            report_lines.append(f"| {p} | {s} | {sub['Age'].mean():.2f} | {sub['Age'].std():.2f} |")
            
    report_lines.append("\n## 2. Wealth Analysis (Utility)")
    report_lines.append("| Persona | Scenario | Mean Utility | Std Dev |")
    report_lines.append("|---|---|---|---|")
    for p in personas:
        for s in ['SugarOnly', 'SugarSpice']:
            sub = df[(df['Persona']==p) & (df['Scenario']==s)]
            report_lines.append(f"| {p} | {s} | {sub['Utility'].mean():.2f} | {sub['Utility'].std():.2f} |")
            
    # T-Tests between Personas within Scenario
    report_lines.append("\n## 3. Statistical Significance (T-test)")
    report_lines.append("Comparing D (Risk-taker) vs A (Conservative) within scenarios:")
    
    for s in ['SugarOnly', 'SugarSpice']:
        d_vals = df[(df['Persona']=='D') & (df['Scenario']==s)]['Utility']
        a_vals = df[(df['Persona']=='A') & (df['Scenario']==s)]['Utility']
        t_stat, p_val = stats.ttest_ind(d_vals, a_vals, equal_var=False)
        sig = "**SIGNIFICANT**" if p_val < 0.05 else "Not Significant"
        report_lines.append(f"- **{s}**: D vs A Wealth Difference: p={p_val:.4e} ({sig})")
        
    with open(os.path.join(output_dir, "batch_report.md"), 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Report saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50, help="Number of experiment pairs to run")
    args = parser.parse_args()
    
    df = run_batch(args.runs)
    analyze_batch_results(df)

if __name__ == "__main__":
    main()
