import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig

def run_experiment(mode, enable_spice, seed=42):
    print(f"\nRunning Controlled Experiment ({mode}: Spice={'ON' if enable_spice else 'OFF'})...")
    
    # Strictly controlled configuration
    # We use identical seeds and identical ranges for shared parameters
    config = SugarscapeConfig(
        # Global Settings
        seed=seed,
        initial_population=250,
        max_ticks=500,
        
        # Environment
        width=50,
        height=50,
        sugar_growback_rate=1,
        
        # Agent Attributes (Shared)
        initial_wealth_range=(5, 25),
        metabolism_range=(1, 4),
        vision_range=(1, 6),
        max_age_range=(60, 100),
        
        # Persona Settings (Shared)
        enable_personas=True,
        # Default distribution: A:0.36, B:0.29, C:0.21, D:0.14
        
        # Spice Settings (Variable)
        enable_spice=enable_spice,
        max_spice_capacity=4 if enable_spice else 0,
        spice_growback_rate=1 if enable_spice else 0,
        initial_spice_range=(5, 25) if enable_spice else (0, 0),
        metabolism_spice_range=(1, 4) if enable_spice else (0, 0)
    )
    
    sim = SugarSimulation(config=config, experiment_name=f"controlled_{mode}")
    sim.run()
    return str(sim.logger.run_dir)

def analyze_comparison(dir_sugar, dir_spice):
    print(f"\nAnalyzing Comparison Results...")
    
    def load_data(run_dir, label):
        with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
            data = json.load(f)
        
        rows = []
        for a in data['agents']:
            # Calculate Standardized Welfare
            # For SugarOnly: W = w_s
            # For SugarSpice: W = CobbDouglas
            # To compare fairly, we might look at 'Survival Count' or 'Relative Rank'
            # But raw wealth values are not directly comparable (Spice adds value).
            # So we focus on *Ranking within the scenario*.
            
            rows.append({
                'Scenario': label,
                'Persona': a['persona'],
                'Age': a['age'],
                'Wealth': a['wealth'], # Sugar only wealth for direct comparison? Or Utility?
                # Let's track both
                'Utility': _calc_utility(a)
            })
        return pd.DataFrame(rows)

    def _calc_utility(a):
        w = a['wealth']
        s = a.get('spice', 0)
        m_s = a['metabolism']
        m_p = a.get('metabolism_spice', 0)
        
        if m_p == 0: return w
        m_total = m_s + m_p
        return (w ** (m_s/m_total)) * (s ** (m_p/m_total))

    df_sugar = load_data(dir_sugar, "SugarOnly")
    df_spice = load_data(dir_spice, "SugarSpice")
    df = pd.concat([df_sugar, df_spice])
    
    output_dir = os.path.join("results", "sugarscape", "controlled_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Survival Rate Comparison (Age)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Persona', y='Age', hue='Scenario', order=['A', 'B', 'C', 'D'], ci=None)
    plt.title('Average Age by Persona (Survival)')
    plt.savefig(os.path.join(output_dir, "survival_comparison.png"))
    plt.close()
    
    # 2. Wealth/Utility Comparison (Log Scale)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Persona', y='Utility', hue='Scenario', order=['A', 'B', 'C', 'D'])
    plt.yscale('log')
    plt.title('Economic Performance by Persona')
    plt.savefig(os.path.join(output_dir, "wealth_comparison.png"))
    plt.close()
    
    # 3. Generate Report Table
    summary = df.groupby(['Scenario', 'Persona'])[['Age', 'Utility']].mean().reset_index()
    summary = summary.pivot(index='Persona', columns='Scenario', values=['Age', 'Utility'])
    
    report = f"""# Controlled Persona Comparison

## Experiment Controls
- **Seed**: 42 (Identical initialization sequence for Sugar layer and Agents)
- **Population**: 250
- **Grid**: 50x50
- **Duration**: 500 ticks

## Key Findings

### 1. Survival (Mean Age)
{summary['Age'].to_markdown()}

### 2. Wealth (Mean Utility)
{summary['Utility'].to_markdown()}

## Conclusion
How does the addition of a second resource constraint (Spice) flip the leaderboard?
"""
    with open(os.path.join(output_dir, "report.md"), 'w') as f:
        f.write(report)
        
    print(f"Analysis saved to: {output_dir}")

def main():
    # Use same seed to ensure 'Agent 1' in both runs has same metabolism_sugar, vision, etc.
    # (Except Spice attributes which are only generated in the second run)
    seed = 12345
    
    dir_sugar = run_experiment("SugarOnly", False, seed)
    dir_spice = run_experiment("SugarSpice", True, seed)
    
    analyze_comparison(dir_sugar, dir_spice)

if __name__ == "__main__":
    main()
