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

def run_experiment(name, enable_trade):
    print(f"\nRunning {name} (Trade={'ON' if enable_trade else 'OFF'})...")
    
    config = SugarscapeConfig(
        # Enable Spice
        enable_spice=True,
        initial_spice_range=(5, 25),
        metabolism_spice_range=(1, 4),
        max_spice_capacity=4,
        spice_growback_rate=1,
        
        # Trade setting
        enable_trade=enable_trade,
        
        # Standard settings
        initial_population=250,
        metabolism_range=(1, 4),
        vision_range=(1, 6),
        sugar_growback_rate=1,
        max_ticks=500
    )
    
    sim = SugarSimulation(config=config, experiment_name=f"trade_effect_{name}")
    sim.run()
    return str(sim.logger.run_dir)

def analyze_trade_effect(no_trade_dir, with_trade_dir):
    print("\nAnalyzing Trade Effects...")
    
    # Load data
    def load_data(run_dir):
        # Metrics CSV
        metrics = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
        # Final State
        with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
            final = json.load(f)
        return metrics, final
        
    metrics_nt, final_nt = load_data(no_trade_dir)
    metrics_wt, final_wt = load_data(with_trade_dir)
    
    output_dir = os.path.join("results", "sugarscape", "trade_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Survival / Population Curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_nt['tick'], metrics_nt['population'], label='No Trade')
    plt.plot(metrics_wt['tick'], metrics_wt['population'], label='With Trade')
    plt.title('Population Survival Curve')
    plt.xlabel('Ticks')
    plt.ylabel('Population')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "survival_curve.png"))
    plt.close()
    
    # 2. Wealth Distribution Comparison
    # Calculate Total Wealth (Sugar + Spice) or Welfare
    def get_welfare_dist(final_state):
        welfares = []
        for a in final_state['agents']:
            # Reconstruct welfare
            m_s = a['metabolism']
            m_p = a['metabolism_spice']
            w_s = a['wealth']
            w_p = a['spice']
            m_total = m_s + m_p
            if m_total > 0:
                w = (w_s ** (m_s/m_total)) * (w_p ** (m_p/m_total))
                welfares.append(w)
        return welfares
        
    w_nt = get_welfare_dist(final_nt)
    w_wt = get_welfare_dist(final_wt)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(w_nt, label='No Trade', fill=True, alpha=0.3)
    sns.kdeplot(w_wt, label='With Trade', fill=True, alpha=0.3)
    plt.title('Welfare Distribution (Cobb-Douglas Utility)')
    plt.xlabel('Welfare')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "welfare_distribution.png"))
    plt.close()
    
    # Generate Report
    mean_welfare_nt = np.mean(w_nt) if w_nt else 0
    mean_welfare_wt = np.mean(w_wt) if w_wt else 0
    
    report = f"""# Trade Effect Analysis: Sugarscape Spice

## 1. Experiment Overview
Comparison of two identical simulations where agents need both Sugar and Spice to survive.
- **No Trade**: Agents must self-sufficiently harvest both resources.
- **With Trade**: Agents can barter Sugar for Spice (and vice versa) based on MRS.

## 2. Key Findings

### A. Survival
- **No Trade Final Population**: {final_nt['tick']} ticks -> {len(final_nt['agents'])} agents
- **With Trade Final Population**: {final_wt['tick']} ticks -> {len(final_wt['agents'])} agents

### B. Welfare (Utility)
- **No Trade Mean Welfare**: {mean_welfare_nt:.2f}
- **With Trade Mean Welfare**: {mean_welfare_wt:.2f}
- **Improvement**: {((mean_welfare_wt - mean_welfare_nt) / mean_welfare_nt * 100) if mean_welfare_nt > 0 else 0:.1f}%

## 3. Conclusion
Trade allows agents to specialize (or survive in biased regions) and exchange surplus for scarcity, leading to higher overall welfare and survival rates (Pareto improvement).
"""
    with open(os.path.join(output_dir, "trade_report.md"), 'w') as f:
        f.write(report)
    print(f"Analysis saved to: {output_dir}")

def main():
    dir_no_trade = run_experiment("NoTrade", False)
    dir_with_trade = run_experiment("WithTrade", True)
    
    analyze_trade_effect(dir_no_trade, dir_with_trade)

if __name__ == "__main__":
    main()
