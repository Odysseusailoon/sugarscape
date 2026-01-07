#!/bin/bash

# Parallel Experiment Runner for Sugarscape
# Launches all 10 experiments in separate tmux sessions

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sugarscape Parallel Experiment Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo "Please install tmux: brew install tmux (macOS) or apt install tmux (Ubuntu)"
    exit 1
fi

# Set API key
export OPENROUTER_API_KEY='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'

# Check if we're in the right directory
if [ ! -f "run_experiments.sh" ]; then
    echo -e "${RED}Error: run_experiments.sh not found${NC}"
    echo "Please run this script from the RedBlackBench directory"
    exit 1
fi

echo -e "${YELLOW}Launching 10 experiments in parallel tmux sessions...${NC}"
echo ""

# Launch all experiments
for i in {1..10}; do
    if [ $i -le 8 ]; then
        # LLM experiments need API key
        tmux new-session -d -s "exp$i" "cd /Users/yifeichen/RedBlackBench && export OPENROUTER_API_KEY='$OPENROUTER_API_KEY' && ./run_experiments.sh $i"
        echo -e "${GREEN}✓ Launched exp$i (LLM)${NC}"
    else
        # Basic agent experiments don't need API key
        tmux new-session -d -s "exp$i" "cd /Users/yifeichen/RedBlackBench && ./run_experiments.sh $i"
        echo -e "${GREEN}✓ Launched exp$i (Basic)${NC}"
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All 10 experiments launched successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Monitor progress:${NC}"
echo "  tmux ls                           # List all sessions"
echo "  tmux attach-session -t exp1       # Monitor experiment 1"
echo "  tmux kill-session -t exp1         # Kill experiment 1 if needed"
echo ""
echo -e "${YELLOW}Expected completion:${NC}"
echo "  LLM experiments (1-8): ~15-30 minutes each"
echo "  Basic experiments (9-10): ~2-5 minutes each"
echo "  Total wall-clock time: ~15-30 minutes"
echo ""
echo -e "${YELLOW}Results will be saved in:${NC}"
echo "  results/sugarscape/experiment_*/"
echo ""
echo -e "${BLUE}Happy experimenting! 🚀${NC}"
