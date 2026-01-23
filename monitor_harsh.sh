#!/bin/bash
EXP_DIR="results/sugarscape/goal_survival/experiment_20260120_011303_370357_1214"

echo "=== Harsh Experiment Monitor: $(date) ==="
echo ""

# Check if process running
if pgrep -f "run_goal_experiment.*harsh" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED or STOPPED"
fi
echo ""

echo "=== Progress ==="
TICKS=$(($(wc -l < "$EXP_DIR/metrics.csv" 2>/dev/null || echo 1) - 1))
echo "Ticks completed: $TICKS / 100"
echo "LLM calls: $(wc -l < "$EXP_DIR/debug/llm_interactions.jsonl" 2>/dev/null || echo 0)"
echo "Trade encounters: $(($(wc -l < "$EXP_DIR/debug/trade_history.csv" 2>/dev/null || echo 1) - 1))"
echo "Deaths: $(($(wc -l < "$EXP_DIR/debug/death_records.csv" 2>/dev/null || echo 1) - 1))"
echo ""

if [ $TICKS -gt 0 ]; then
    echo "=== Latest metrics ==="
    tail -3 "$EXP_DIR/metrics.csv" 2>/dev/null | cut -d',' -f1-5
fi
