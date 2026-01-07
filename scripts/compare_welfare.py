"""Script to compare welfare metrics between different experiment runs.

Usage:
    python scripts/compare_welfare.py <llm_metrics_csv> <baseline_metrics_csv> <output_dir>
    
Example:
    python scripts/compare_welfare.py \
        results/sugarscape/llm_experiment/experiment_20241219_120000/metrics.csv \
        results/sugarscape/baseline/experiment_20241219_110000/metrics.csv \
        results/sugarscape/comparisons/
"""

import sys
from pathlib import Path
from redblackbench.sugarscape.welfare_plots import WelfarePlotter


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    
    llm_csv = sys.argv[1]
    baseline_csv = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Validate inputs
    if not Path(llm_csv).exists():
        print(f"Error: LLM metrics CSV not found: {llm_csv}")
        sys.exit(1)
    
    if not Path(baseline_csv).exists():
        print(f"Error: Baseline metrics CSV not found: {baseline_csv}")
        sys.exit(1)
    
    # Generate comparison plots
    print(f"Comparing welfare metrics:")
    print(f"  LLM:      {llm_csv}")
    print(f"  Baseline: {baseline_csv}")
    print(f"  Output:   {output_dir}")
    print()
    
    WelfarePlotter.generate_comparison_plots(
        llm_csv_path=llm_csv,
        baseline_csv_path=baseline_csv,
        output_dir=output_dir
    )
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()


