#!/bin/bash
EXP_DIR="results/sugarscape/goal_survival/experiment_20260120_002825_192002_9195"

echo "=== Experiment Monitor: $(date) ==="
echo ""

# Check if process running
if pgrep -f "run_goal_experiment" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED or STOPPED"
fi
echo ""

echo "=== Progress ==="
TICKS=$(($(wc -l < "$EXP_DIR/metrics.csv") - 1))
echo "Ticks completed: $TICKS / 50"
echo "LLM calls: $(wc -l < "$EXP_DIR/debug/llm_interactions.jsonl")"
echo "Trade encounters: $(($(wc -l < "$EXP_DIR/debug/trade_history.csv") - 1))"
echo ""

if [ $TICKS -gt 0 ]; then
    echo "=== Latest metrics ==="
    tail -3 "$EXP_DIR/metrics.csv" | column -t -s','
fi
