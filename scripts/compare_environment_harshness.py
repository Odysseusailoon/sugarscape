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

def run_experiment(env_type, config):
    print(f"\nRunning {env_type} Environment Experiment...")
    experiment_name = f"comparison_{env_type}"
    sim = SugarSimulation(config=config, experiment_name=experiment_name)
    sim.run(steps=500)
    return str(sim.logger.run_dir)

def analyze_comparison(harsh_dir, abundant_dir):
    print("\nAnalyzing Comparison Results...")
    
    def load_final_state(run_dir):
        with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
            return json.load(f)
            
    harsh_data = load_final_state(harsh_dir)
    abundant_data = load_final_state(abundant_dir)
    
    # Create DataFrame for analysis
    data = []
    
    for agent in harsh_data['agents']:
        data.append({
            'Environment': 'Harsh',
            'Metabolism': agent['metabolism'],
            'Vision': agent['vision'],
            'Wealth': agent['wealth']
        })
        
    for agent in abundant_data['agents']:
        data.append({
            'Environment': 'Abundant',
            'Metabolism': agent['metabolism'],
            'Vision': agent['vision'],
            'Wealth': agent['wealth']
        })
        
    df = pd.DataFrame(data)
    
    # Output directory
    output_dir = os.path.join("results", "sugarscape", "comparison_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Survival Analysis (Distribution of Attributes)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Metabolism Distribution
    sns.histplot(data=df, x='Metabolism', hue='Environment', element='step', stat='density', common_norm=False, ax=axes[0])
    axes[0].set_title('Survival Selection: Metabolism')
    
    # Vision Distribution
    sns.histplot(data=df, x='Vision', hue='Environment', element='step', stat='density', common_norm=False, ax=axes[1])
    axes[1].set_title('Survival Selection: Vision')
    
    plt.savefig(os.path.join(output_dir, "survival_selection.png"))
    plt.close()
    
    # 2. Wealth Correlation Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Harsh
    harsh_corr = df[df['Environment']=='Harsh'][['Wealth', 'Metabolism', 'Vision']].corr()
    sns.heatmap(harsh_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Correlation Matrix (Harsh)')
    
    # Abundant
    abundant_corr = df[df['Environment']=='Abundant'][['Wealth', 'Metabolism', 'Vision']].corr()
    sns.heatmap(abundant_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Correlation Matrix (Abundant)')
    
    plt.savefig(os.path.join(output_dir, "wealth_correlations.png"))
    plt.close()
    
    # Generate Text Report
    report = f"""# Environment Comparison: Harsh vs Abundant

## 1. Experiment Setup
- **Harsh**: High population density (400 agents), low regrowth (alpha=1).
- **Abundant**: Low population density (100 agents), high regrowth (alpha=2).

## 2. Key Findings

### A. Survival Selection
- **Harsh Env**: Mean Metabolism = {df[df['Environment']=='Harsh']['Metabolism'].mean():.2f}, Mean Vision = {df[df['Environment']=='Harsh']['Vision'].mean():.2f}
- **Abundant Env**: Mean Metabolism = {df[df['Environment']=='Abundant']['Metabolism'].mean():.2f}, Mean Vision = {df[df['Environment']=='Abundant']['Vision'].mean():.2f}

### B. Wealth Determinants (Correlation with Wealth)
- **Harsh**: Metabolism ({harsh_corr.loc['Wealth', 'Metabolism']:.2f}), Vision ({harsh_corr.loc['Wealth', 'Vision']:.2f})
- **Abundant**: Metabolism ({abundant_corr.loc['Wealth', 'Metabolism']:.2f}), Vision ({abundant_corr.loc['Wealth', 'Vision']:.2f})

### C. Conclusion
Does "Smart" (High Vision) matter more when resources are abundant?
"""
    with open(os.path.join(output_dir, "comparison_report.md"), 'w') as f:
        f.write(report)
        
    print(f"Analysis saved to: {output_dir}")

def main():
    # 1. Harsh Environment (The "Power Law" setup we just did)
    harsh_config = SugarscapeConfig(
        initial_population=400,
        metabolism_range=(1, 5),
        vision_range=(1, 10),
        sugar_growback_rate=1
    )
    
    # 2. Abundant Environment (Rich resources, low pressure)
    abundant_config = SugarscapeConfig(
        initial_population=100,  # Few people
        metabolism_range=(1, 5),
        vision_range=(1, 10),
        sugar_growback_rate=4    # Fast regrowth!
    )
    
    harsh_dir = run_experiment("Harsh", harsh_config)
    abundant_dir = run_experiment("Abundant", abundant_config)
    
    analyze_comparison(harsh_dir, abundant_dir)

if __name__ == "__main__":
    main()
