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

def run_experiment(mode, enable_spice):
    print(f"\nRunning Persona Experiment ({mode}: Spice={'ON' if enable_spice else 'OFF'})...")
    
    config = SugarscapeConfig(
        # Persona settings
        enable_personas=True,
        
        # Spice settings
        enable_spice=enable_spice,
        initial_spice_range=(5, 25),
        metabolism_spice_range=(1, 4),
        max_spice_capacity=4,
        spice_growback_rate=1,
        
        # Standard settings
        initial_population=250,
        metabolism_range=(1, 4),
        vision_range=(1, 6),
        sugar_growback_rate=1,
        max_ticks=500
    )
    
    sim = SugarSimulation(config=config, experiment_name=f"persona_{mode}")
    sim.run()
    return str(sim.logger.run_dir)

def analyze_personas(run_dir, mode):
    print(f"\nAnalyzing {mode} Results...")
    
    # Load Final State
    with open(os.path.join(run_dir, "final_state.json"), 'r') as f:
        final = json.load(f)
        
    with open(os.path.join(run_dir, "initial_state.json"), 'r') as f:
        initial = json.load(f)
    
    # Process Agents
    def process_agents(state_data):
        rows = []
        for a in state_data['agents']:
            # Calculate Welfare
            m_s = a['metabolism']
            m_p = a.get('metabolism_spice', 0)
            w_s = a['wealth']
            w_p = a.get('spice', 0)
            
            welfare = w_s
            if m_p > 0:
                m_total = m_s + m_p
                welfare = (w_s ** (m_s/m_total)) * (w_p ** (m_p/m_total))
                
            rows.append({
                'Persona': a['persona'],
                'Wealth': w_s,
                'Welfare': welfare,
                'Age': a['age']
            })
        return pd.DataFrame(rows)
        
    df_init = process_agents(initial)
    df_final = process_agents(final)
    
    output_dir = os.path.join("results", "sugarscape", "persona_analysis", mode)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Survival Rate by Persona
    # Count initial and final per persona
    init_counts = df_init['Persona'].value_counts()
    final_counts = df_final['Persona'].value_counts()
    
    # Since replacement is ON, population is constant. 
    # But "Survival" in evolutionary sense means "Does this persona trait persist or dominate?"
    # Wait, in our simulation, replacement agents get RANDOM persona based on distribution?
    # Actually, `_create_agent` in simulation.py samples persona from the Config distribution.
    # So the *population share* should remain roughly constant unless we implemented inheritance/evolution.
    # HOWEVER, we can check the *Age* of survivors. 
    # High average age = better survival strategy.
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_final, x='Persona', y='Age', order=['A', 'B', 'C', 'D'])
    plt.title(f'Survival Capability: Age Distribution by Persona ({mode})')
    plt.savefig(os.path.join(output_dir, "age_by_persona.png"))
    plt.close()
    
    # 2. Wealth Accumulation by Persona
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_final, x='Persona', y='Welfare', order=['A', 'B', 'C', 'D'])
    plt.yscale('log')
    plt.title(f'Economic Success: Welfare by Persona ({mode})')
    plt.savefig(os.path.join(output_dir, "welfare_by_persona.png"))
    plt.close()
    
    # 3. Generate Report
    means = df_final.groupby('Persona')[['Welfare', 'Age']].mean()
    
    report = f"""# Persona Analysis: {mode}

## 1. Scenario
- **Mode**: {mode}
- **Personas**:
  - A (Conservative): Safety first, avoids risk.
  - B (Foresight): Looks for long-term capacity.
  - C (Nomad): Likes novelty/exploration.
  - D (Risk-taker): Greed maximization.

## 2. Key Findings (Final Population)

### A. Who Lives Longest? (Mean Age)
{means['Age'].to_markdown()}

### B. Who is Richest? (Mean Welfare)
{means['Welfare'].to_markdown()}

## 3. Interpretation
"""
    with open(os.path.join(output_dir, "report.md"), 'w') as f:
        f.write(report)
        
    print(f"Analysis saved to: {output_dir}")

def main():
    # Experiment 1: Single Resource (Classic Sugarscape)
    dir_sugar = run_experiment("SugarOnly", False)
    analyze_personas(dir_sugar, "SugarOnly")
    
    # Experiment 2: Dual Resource (Spice)
    # Testing if "Planning" (Persona B) helps balance two resources
    dir_spice = run_experiment("SugarSpice", True)
    analyze_personas(dir_spice, "SugarSpice")

if __name__ == "__main__":
    main()
