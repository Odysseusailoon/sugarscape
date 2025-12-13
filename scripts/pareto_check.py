import sys
import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

def load_batch_data(num_runs=50):
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
                    'Efficiency': utility / a['age'] if a['age'] > 0 else 0
                })
        except FileNotFoundError:
            continue
                
    return pd.DataFrame(all_data)

def check_pareto(df):
    print("\n--- Pareto Optimality Check (SugarSpice) ---")
    summary = df.groupby('Persona')[['Age', 'Utility', 'Efficiency']].mean()
    print(summary)
    
    personas = ['A', 'B', 'C', 'D']
    dominated = {}
    
    for p1 in personas:
        is_dominated = False
        by_whom = []
        val1 = summary.loc[p1]
        
        for p2 in personas:
            if p1 == p2: continue
            val2 = summary.loc[p2]
            
            # Check if p2 dominates p1 (Strictly better in one, not worse in other)
            if (val2['Age'] >= val1['Age'] and val2['Utility'] > val1['Utility']) or \
               (val2['Age'] > val1['Age'] and val2['Utility'] >= val1['Utility']):
                is_dominated = True
                by_whom.append(p2)
        
        if is_dominated:
            dominated[p1] = by_whom
            print(f"Persona {p1} is DOMINATED by {by_whom}")
        else:
            print(f"Persona {p1} is on the Pareto Frontier (for Age/Utility)")

def analyze_efficiency_bias(df):
    print("\n--- Efficiency Bias Analysis ---")
    # Check if high efficiency is correlated with low age (Early Death Bonus)
    corr = df['Age'].corr(df['Efficiency'])
    print(f"Correlation (Age vs Efficiency): {corr:.4f}")
    
    # Compare Efficiency of "Young Dead" vs "Old Survivors" within Type A
    df_a = df[df['Persona']=='A']
    median_age = df_a['Age'].median()
    eff_young = df_a[df_a['Age'] < median_age]['Efficiency'].mean()
    eff_old = df_a[df_a['Age'] >= median_age]['Efficiency'].mean()
    
    print(f"Type A Efficiency (Young < {median_age}): {eff_young:.2f}")
    print(f"Type A Efficiency (Old >= {median_age}): {eff_old:.2f}")
    
    if eff_young > eff_old:
        print("CONCLUSION: Type A's high efficiency is inflated by early deaths (Early Death Bonus).")
    else:
        print("CONCLUSION: Type A maintains efficiency over time.")

def main():
    df = load_batch_data(50)
    if len(df) == 0:
        print("No data.")
        return
        
    check_pareto(df)
    analyze_efficiency_bias(df)

if __name__ == "__main__":
    main()
