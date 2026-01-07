#!/bin/bash
# Limited parallel experiment runner - 2 experiments at a time

export OPENROUTER_API_KEY='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'

echo "=== LIMITED PARALLEL EXPERIMENT RUNNER ==="
echo "Running 2 experiments at a time to avoid rate limiting"
echo ""

# Function to run an experiment and wait
run_exp() {
    local exp_num=$1
    echo "Starting experiment $exp_num..."
    ./run_experiments.sh $exp_num
    echo "✓ Experiment $exp_num completed"
}

# Run experiments in pairs
echo "=== BATCH 1: Experiments 9,10 (Basic - fast) ==="
run_exp 9 &
run_exp 10 &
wait
echo "✓ Batch 1 complete"
echo ""

echo "=== BATCH 2: Experiments 5,6 (Qwen3-30B) ==="
run_exp 5 &
run_exp 6 &
wait
echo "✓ Batch 2 complete"
echo ""

echo "=== BATCH 3: Experiments 7,8 (Qwen3-30B) ==="
run_exp 7 &
run_exp 8 &
wait
echo "✓ Batch 3 complete"
echo ""

echo "=== BATCH 4: Experiments 1,2 (Kimi-K2) ==="
run_exp 1 &
run_exp 2 &
wait
echo "✓ Batch 4 complete"
echo ""

echo "=== BATCH 5: Experiments 3,4 (Kimi-K2) ==="
run_exp 3 &
run_exp 4 &
wait
echo "✓ Batch 5 complete"
echo ""

echo "🎉 ALL 10 EXPERIMENTS COMPLETED!"
