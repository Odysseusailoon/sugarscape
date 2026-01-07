#!/bin/bash

# Sugarscape LLM Experiment Runner
# Usage: ./run_experiments.sh [experiment_number|all|lb]
#
# Supports:
# - Single provider mode (OpenRouter)
# - Load-balanced mode (OpenRouter + AIHubMix)

set -e  # Exit on any error

# Auto-load environment variables from .env if present (do not echo secrets)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
fi

# Prefer venv python if present (enables matplotlib plots, consistent deps)
PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# API key check function (only for LLM experiments)
check_api_key() {
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo -e "${RED}Error: OPENROUTER_API_KEY environment variable not set${NC}"
        echo "Please run: export OPENROUTER_API_KEY='your-key-here'"
        exit 1
    fi
}

# Check both API keys for load-balanced mode
check_lb_api_keys() {
    check_api_key
    if [ -z "$AIHUBMIX_API_KEY" ]; then
        echo -e "${YELLOW}Warning: AIHUBMIX_API_KEY not set, using OpenRouter only${NC}"
    fi
}

# Load-balanced experiment function
run_lb_experiment() {
    local exp_name=$1
    local model=$2
    local goal=$3
    local difficulty=$4
    local population=$5
    local ticks=$6

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Load-Balanced Experiment: $exp_name${NC}"
    echo -e "${BLUE}Model: $model | Goal: $goal | Difficulty: $difficulty${NC}"
    echo -e "${BLUE}Population: $population | Ticks: $ticks${NC}"
    echo -e "${BLUE}========================================${NC}"

    check_lb_api_keys

    # Determine provider mode
    PROVIDER_ARG="--providers both"
    if [ -z "$AIHUBMIX_API_KEY" ]; then
        PROVIDER_ARG="--providers openrouter"
    fi

    $PYTHON scripts/run_sugarscape_loadbalanced.py \
        --model "$model" \
        --goal-preset "$goal" \
        --difficulty "$difficulty" \
        --ticks "$ticks" \
        --population "$population" \
        --strategy round_robin \
        --max-concurrent 5 \
        $PROVIDER_ARG \
        --seed 42

    echo -e "${GREEN}✓ Experiment $exp_name completed${NC}"
    echo ""
}

# Base command function
run_experiment() {
    local exp_num=$1
    local agent_type=$2
    local goal=$3
    local difficulty=$4

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment $exp_num: $agent_type + $goal + $difficulty${NC}"
    echo -e "${BLUE}========================================${NC}"

    if [ "$agent_type" = "basic" ]; then
        # Basic agents - no LLM, no API key needed
        $PYTHON scripts/run_sugarscape.py \
            --mode basic \
            --goal-preset "$goal" \
            --difficulty "$difficulty" \
            --ticks 100 \
            --population 100 \
            --trade-rounds 2 \
            --seed 42
    else
        # LLM agents - check API key first
        check_api_key
        $PYTHON scripts/run_sugarscape.py \
            --mode llm \
            --model "$agent_type" \
            --goal-preset "$goal" \
            --difficulty "$difficulty" \
            --ticks 100 \
            --population 100 \
            --trade-rounds 2 \
            --seed 42
    fi

    echo -e "${GREEN}✓ Experiment $exp_num completed${NC}"
    echo ""
}

# Experiment definitions
case "$1" in
    1)
        run_experiment 1 "moonshotai/kimi-k2-thinking" "survival" "standard"
        ;;
    2)
        run_experiment 2 "moonshotai/kimi-k2-thinking" "survival" "harsh"
        ;;
    3)
        run_experiment 3 "moonshotai/kimi-k2-thinking" "egalitarian" "standard"
        ;;
    4)
        run_experiment 4 "moonshotai/kimi-k2-thinking" "egalitarian" "harsh"
        ;;
    5)
        run_experiment 5 "qwen/qwen3-30b-a3b-thinking-2507" "survival" "standard"
        ;;
    6)
        run_experiment 6 "qwen/qwen3-30b-a3b-thinking-2507" "survival" "harsh"
        ;;
    7)
        run_experiment 7 "qwen/qwen3-30b-a3b-thinking-2507" "egalitarian" "standard"
        ;;
    8)
        run_experiment 8 "qwen/qwen3-30b-a3b-thinking-2507" "egalitarian" "harsh"
        ;;
    9)
        run_experiment 9 "basic" "survival" "standard"
        ;;
    10)
        run_experiment 10 "basic" "egalitarian" "standard"
        ;;
    all)
        echo -e "${YELLOW}Running all 10 experiments sequentially...${NC}"
        echo -e "${YELLOW}This will take considerable time and tokens!${NC}"
        echo ""

        for i in {1..10}; do
            case $i in
                1) run_experiment $i "moonshotai/kimi-k2-thinking" "survival" "standard" ;;
                2) run_experiment $i "moonshotai/kimi-k2-thinking" "survival" "harsh" ;;
                3) run_experiment $i "moonshotai/kimi-k2-thinking" "egalitarian" "standard" ;;
                4) run_experiment $i "moonshotai/kimi-k2-thinking" "egalitarian" "harsh" ;;
                5) run_experiment $i "qwen/qwen3-30b-a3b-thinking-2507" "survival" "standard" ;;
                6) run_experiment $i "qwen/qwen3-30b-a3b-thinking-2507" "survival" "harsh" ;;
                7) run_experiment $i "qwen/qwen3-30b-a3b-thinking-2507" "egalitarian" "standard" ;;
                8) run_experiment $i "qwen/qwen3-30b-a3b-thinking-2507" "egalitarian" "harsh" ;;
                9) run_experiment $i "basic" "survival" "standard" ;;
                10) run_experiment $i "basic" "egalitarian" "standard" ;;
            esac
        done

        echo -e "${GREEN}🎉 All 10 experiments completed!${NC}"
        echo -e "${BLUE}Results saved in: results/sugarscape/${NC}"
        ;;
    test)
        echo -e "${YELLOW}Running small test experiment...${NC}"
        $PYTHON scripts/run_sugarscape.py \
            --mode llm \
            --model moonshotai/kimi-k2-thinking \
            --goal-preset survival \
            --difficulty standard \
            --ticks 10 \
            --population 10 \
            --trade-rounds 2 \
            --seed 42
        echo -e "${GREEN}✓ Test completed${NC}"
        ;;
    lb)
        # Load-balanced experiment: qwen3-30b survival standard
        echo -e "${YELLOW}Running load-balanced experiment (OpenRouter + AIHubMix)...${NC}"
        run_lb_experiment "qwen3_survival" \
            "qwen/qwen3-30b-a3b-thinking-2507" \
            "survival" \
            "standard" \
            100 \
            100
        ;;
    lb-small)
        # Small load-balanced test
        echo -e "${YELLOW}Running small load-balanced test...${NC}"
        run_lb_experiment "qwen3_test" \
            "qwen/qwen3-30b-a3b-thinking-2507" \
            "survival" \
            "standard" \
            20 \
            10
        ;;
    lb-trade)
        # Load-balanced test with trading enabled
        # Uses smaller grid (10x10) with 8 agents for higher trade probability
        echo -e "${YELLOW}Running load-balanced test with TRADE enabled...${NC}"
        check_lb_api_keys
        
        $PYTHON scripts/run_sugarscape_loadbalanced.py \
            --model "qwen/qwen3-vl-30b-a3b-thinking" \
            --goal-preset survival \
            --difficulty standard \
            --ticks 15 \
            --population 8 \
            --grid-size 10 \
            --strategy round_robin \
            --max-concurrent 2 \
            --enable-trade \
            --trade-mode dialogue \
            --trade-rounds 4 \
            --providers openrouter \
            --seed 42
        
        echo -e "${GREEN}✓ Trade experiment completed${NC}"
        ;;
    lb-*)
        # Parse lb-<population>-<ticks> format
        IFS='-' read -ra PARTS <<< "$1"
        if [ ${#PARTS[@]} -ge 3 ]; then
            POP=${PARTS[1]}
            TICKS=${PARTS[2]}
            echo -e "${YELLOW}Running load-balanced experiment: pop=$POP, ticks=$TICKS${NC}"
            run_lb_experiment "custom_lb" \
                "qwen/qwen3-30b-a3b-thinking-2507" \
                "survival" \
                "standard" \
                "$POP" \
                "$TICKS"
        else
            echo "Usage: $0 lb-<population>-<ticks>"
            echo "Example: ./run_experiments.sh lb-50-20"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [1-10|all|test|lb|lb-small|lb-trade|lb-<pop>-<ticks>]"
        echo ""
        echo "Single Provider Experiments:"
        echo "  1-4: Kimi-K2 model (OpenRouter)"
        echo "  5-8: Qwen3-30B model (OpenRouter)"
        echo "  9-10: Basic agents (no LLM)"
        echo "  all: Run all 10 experiments"
        echo "  test: Run small test experiment (10 ticks, 10 agents)"
        echo ""
        echo "Load-Balanced Experiments (OpenRouter + AIHubMix):"
        echo "  lb: Full experiment (100 agents, 100 ticks)"
        echo "  lb-small: Small test (20 agents, 10 ticks)"
        echo "  lb-trade: Trade-enabled test (8 agents, 15 ticks, 10x10 grid)"
        echo "  lb-<pop>-<ticks>: Custom (e.g., lb-50-20)"
        echo ""
        echo "Environment Variables:"
        echo "  OPENROUTER_API_KEY: Required for all LLM experiments"
        echo "  AIHUBMIX_API_KEY: Optional, enables dual-provider load balancing"
        echo ""
        echo "Example:"
        echo "  ./run_experiments.sh 5       # OpenRouter only"
        echo "  ./run_experiments.sh lb      # Load-balanced full experiment"
        echo "  ./run_experiments.sh lb-30-15 # Custom: 30 agents, 15 ticks"
        exit 1
        ;;
esac
