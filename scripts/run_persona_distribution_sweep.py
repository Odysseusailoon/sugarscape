import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root + scripts to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig

# Reuse existing plot generators
import plot_existing
import analyze_sugarscape_deep


def _utility(agent: Dict[str, Any]) -> float:
    """Cobb-Douglas utility for Sugar+Spice; reduces to sugar wealth when no spice metabolism."""
    w = float(agent.get("wealth", 0))
    s = float(agent.get("spice", 0))
    m_s = float(agent.get("metabolism", 0))
    m_p = float(agent.get("metabolism_spice", 0))
    if m_p <= 0:
        return w
    m_total = m_s + m_p
    if m_total <= 0 or w <= 0 or s <= 0:
        return 0.0
    return float((w ** (m_s / m_total)) * (s ** (m_p / m_total)))


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if len(values) == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cum = np.cumsum(values)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def persona_plots(run_dir: str):
    """Generate persona-focused plots into run_dir/plots."""
    run_dir = str(run_dir)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(run_dir, "initial_state.json"), "r") as f:
        initial = json.load(f)
    with open(os.path.join(run_dir, "final_state.json"), "r") as f:
        final = json.load(f)

    def to_df(state: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for a in state.get("agents", []):
            rows.append(
                {
                    "Persona": a.get("persona", "A"),
                    "Age": a.get("age", 0),
                    "Wealth": a.get("wealth", 0),
                    "Spice": a.get("spice", 0),
                    "Metabolism": a.get("metabolism", 0),
                    "Metabolism_Spice": a.get("metabolism_spice", 0),
                    "Vision": a.get("vision", 0),
                    "Utility": _utility(a),
                    "X": (a.get("pos") or [None, None])[0],
                    "Y": (a.get("pos") or [None, None])[1],
                }
            )
        return pd.DataFrame(rows)

    df_init = to_df(initial)
    df_final = to_df(final)

    persona_order = ["A", "B", "C", "D", "E"]

    # 1) Persona share (initial vs final)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, df, title in [(axes[0], df_init, "Initial"), (axes[1], df_final, "Final")]:
        counts = df["Persona"].value_counts().reindex(persona_order).fillna(0).astype(int)
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f"Persona Counts ({title})")
        ax.set_xlabel("Persona")
        ax.set_ylabel("Agents")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "persona_counts.png"), dpi=200)
    plt.close()

    # 2) Age distribution by persona (final)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_final, x="Persona", y="Age", order=persona_order)
    plt.title("Age Distribution by Persona (Final)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "age_by_persona.png"), dpi=200)
    plt.close()

    # 3) Utility distribution by persona (final, log)
    df_plot = df_final.copy()
    df_plot["Utility_plot"] = df_plot["Utility"].replace(0, np.nan)
    plt.figure(figsize=(8, 5))
    sns.boxenplot(data=df_plot, x="Persona", y="Utility_plot", order=persona_order)
    plt.yscale("log")
    plt.title("Utility Distribution by Persona (Final, log scale)")
    plt.ylabel("Utility (Cobb-Douglas if spice enabled)")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "utility_by_persona.png"), dpi=200)
    plt.close()

    # 4) Spatial distribution by persona (final)
    try:
        sugar_cap = np.array(final.get("sugar_capacity", []))
        if sugar_cap.size:
            plt.figure(figsize=(8, 7))
            plt.imshow(sugar_cap.T, origin="lower", cmap="YlOrBr", alpha=0.55)
            palette = {"A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e", "D": "#d62728", "E": "#9467bd"}
            for p in persona_order:
                sub = df_final[df_final["Persona"] == p]
                plt.scatter(sub["X"], sub["Y"], s=12, alpha=0.75, label=p, c=palette.get(p))
            plt.title("Final Spatial Distribution by Persona")
            plt.legend(title="Persona")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "spatial_personas.png"), dpi=200)
            plt.close()
    except Exception:
        # Non-critical
        pass

    # 5) Persona summary report (avoid optional 'tabulate' dependency)
    summary = (
        df_final.groupby("Persona")[["Age", "Wealth", "Utility"]]
        .agg(["mean", "median"])
        .reindex(persona_order)
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns.to_list()]
    summary = summary.reset_index()

    gini_by_persona = (
        df_final.groupby("Persona")["Wealth"].apply(lambda s: _gini(s.values)).reindex(persona_order).reset_index()
    )
    gini_by_persona.columns = ["Persona", "Wealth_Gini"]

    util_gini_by_persona = (
        df_final.groupby("Persona")["Utility"].apply(lambda s: _gini(s.values)).reindex(persona_order).reset_index()
    )
    util_gini_by_persona.columns = ["Persona", "Utility_Gini"]

    gini_table = gini_by_persona.merge(util_gini_by_persona, on="Persona", how="outer")

    def _to_md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    report = []
    report.append("# Persona Sweep: Per-Run Summary\n")
    report.append("## Final Population Summary\n")
    report.append(_to_md_table(summary.round(3)))
    report.append("\n\n## Inequality (Gini)\n")
    report.append(_to_md_table(gini_table.round(3)))
    report.append("\n\n## Notes\n")
    report.append("- Utility is Sugar wealth in SugarOnly runs; Cobb-Douglas utility when Spice is enabled.\n")
    with open(os.path.join(run_dir, "persona_report.md"), "w") as f:
        f.write("\n".join(report))


def parse_args():
    p = argparse.ArgumentParser(description="Run Sugarscape persona_distribution sweep and generate plots.")
    p.add_argument("--only-persona-plots", action="store_true", help="Only generate persona plots for --run-dir.")
    p.add_argument("--run-dir", type=str, default="", help="Existing run directory to analyze (used with --only-persona-plots).")
    p.add_argument("--variant", choices=["sugar", "spice"], default="spice", help="Run SugarOnly or Sugar+Spice.")
    p.add_argument("--difficulty", choices=["standard", "easy", "harsh", "desert"], default="standard")
    p.add_argument("--ticks", type=int, default=500)
    p.add_argument("--population", type=int, default=500)
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--height", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweeps", type=str, default="all",
                   help="Comma-separated list of sweep names to run, or 'all', 'samaritan' (altruism comparison only), 'original' (without samaritan)")
    return p.parse_args()


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")

    if args.only_persona_plots:
        if not args.run_dir:
            raise SystemExit("--run-dir is required when using --only-persona-plots")
        persona_plots(args.run_dir)
        print(f"✓ Persona plots generated in: {os.path.join(args.run_dir, 'plots')}")
        return

    # A small, reasonable sweep (fast enough to run locally)
    all_sweeps: Dict[str, Dict[str, float]] = {
        "baseline": {"A": 0.36, "B": 0.29, "C": 0.21, "D": 0.14, "E": 0.0},
        "uniform": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "E": 0.0},
        "planner_heavy": {"A": 0.15, "B": 0.55, "C": 0.15, "D": 0.15, "E": 0.0},
        "risk_heavy": {"A": 0.15, "B": 0.15, "C": 0.15, "D": 0.55, "E": 0.0},
        # Altruism comparison sweeps
        "all_selfish": {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "E": 0.0},  # No Samaritans
        "all_samaritan": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 1.0},  # 100% Samaritans
        "mixed_25pct_samaritan": {"A": 0.1875, "B": 0.1875, "C": 0.1875, "D": 0.1875, "E": 0.25},  # 25% Samaritans
        "mixed_50pct_samaritan": {"A": 0.125, "B": 0.125, "C": 0.125, "D": 0.125, "E": 0.50},  # 50% Samaritans
    }

    # Filter sweeps based on --sweeps argument
    if args.sweeps == "all":
        sweeps = all_sweeps
    elif args.sweeps == "samaritan":
        # Only altruism comparison sweeps
        sweeps = {k: v for k, v in all_sweeps.items() if "samaritan" in k or k == "all_selfish"}
    elif args.sweeps == "original":
        # Original sweeps without samaritan-focused ones
        sweeps = {k: v for k, v in all_sweeps.items() if k in ["baseline", "uniform", "planner_heavy", "risk_heavy"]}
    else:
        # Comma-separated list
        requested = [s.strip() for s in args.sweeps.split(",")]
        sweeps = {k: v for k, v in all_sweeps.items() if k in requested}
        if not sweeps:
            raise SystemExit(f"No matching sweeps found. Available: {list(all_sweeps.keys())}")

    print(f"Running sweeps: {list(sweeps.keys())}")

    for name, dist in sweeps.items():
        experiment_type = f"persona_dist_{args.variant}_{args.difficulty}_N{args.population}_T{args.ticks}_{name}"
        print(f"\n=== Running sweep: {experiment_type} ===")

        config = SugarscapeConfig(
            seed=args.seed,
            width=args.width,
            height=args.height,
            initial_population=args.population,
            max_ticks=args.ticks,
            # Standard Sugarscape-ish defaults
            max_sugar_capacity=4,
            sugar_growback_rate=1,
            enable_spice=(args.variant == "spice"),
            max_spice_capacity=4,
            spice_growback_rate=1,
            enable_trade=False,
            # Agent defaults
            initial_wealth_range=(5, 25),
            initial_spice_range=(5, 25),
            metabolism_range=(1, 4),
            metabolism_spice_range=(1, 4),
            vision_range=(1, 6),
            max_age_range=(60, 100),
            # Personas
            enable_personas=True,
            persona_distribution=dist,
        )

        # Difficulty presets (match scripts/run_sugarscape.py)
        if args.difficulty == "easy":
            config.sugar_growback_rate = 2
            config.max_sugar_capacity = 6
            config.spice_growback_rate = 2
            config.max_spice_capacity = 6
        elif args.difficulty == "harsh":
            config.sugar_growback_rate = 1
            config.max_sugar_capacity = 2
            config.spice_growback_rate = 1
            config.max_spice_capacity = 2
        elif args.difficulty == "desert":
            config.sugar_growback_rate = 0
            config.max_sugar_capacity = 4
            config.spice_growback_rate = 0
            config.max_spice_capacity = 4

        sim = SugarSimulation(config=config, experiment_name=experiment_type)
        sim.run()

        run_dir = str(sim.logger.run_dir)
        plots_dir = str(sim.logger.plots_dir)

        # Extra plots: wealth distribution, lorenz curve, gini series, spatial, mobility
        plot_existing.plot_all(run_dir, plots_dir)

        # Deep analysis (selection effects, correlations, report) into run_dir/plots
        try:
            with open(os.path.join(run_dir, "initial_state.json"), "r") as f:
                initial_state = json.load(f)
            with open(os.path.join(run_dir, "final_state.json"), "r") as f:
                final_state = json.load(f)
            df_final = analyze_sugarscape_deep.analyze_selection_effects(initial_state, final_state, plots_dir)
            analyze_sugarscape_deep.analyze_wealth_determinants(df_final, plots_dir)
            analyze_sugarscape_deep.generate_report(run_dir, df_final)
        except Exception as e:
            print(f"Warning: deep analysis failed for {run_dir}: {e}")

        # Persona-focused plots + report
        try:
            persona_plots(run_dir)
        except Exception as e:
            print(f"Warning: persona plots failed for {run_dir}: {e}")

        print(f"✓ Completed: {run_dir}")


if __name__ == "__main__":
    main()


