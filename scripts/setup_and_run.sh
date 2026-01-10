#!/bin/bash
#
# All-in-One Setup Script for Sugarscape Experiments with vLLM
#
# This script handles:
#   1. Python environment setup (dependencies)
#   2. Model download from HuggingFace
#   3. vLLM server startup with LoRA support
#   4. Running sugarscape experiments
#
# Usage:
#   ./scripts/setup_and_run.sh setup      # Install dependencies only
#   ./scripts/setup_and_run.sh download   # Download models only
#   ./scripts/setup_and_run.sh server     # Start vLLM server only
#   ./scripts/setup_and_run.sh run        # Run experiment (server must be running)
#   ./scripts/setup_and_run.sh all        # Do everything (setup + download + server + run)
#
# Environment variables (override defaults):
#   HF_BASE_MODEL    - HuggingFace path for base model
#   HF_LORA_ADAPTER  - HuggingFace path for LoRA adapter
#   LOCAL_MODEL_DIR  - Local directory to store models
#   VLLM_PORT        - Port for vLLM server (default: 8000)
#

set -e  # Exit on error

#=============================================================================
# CONFIGURATION - Edit these values for your setup
#=============================================================================

# HuggingFace model paths
HF_BASE_MODEL="${HF_BASE_MODEL:-Qwen/Qwen3-14B}"
HF_LORA_ADAPTER="${HF_LORA_ADAPTER:-spacezenmasterr/redblackbench-qwen3-14b-sft-v2}"

# Local storage paths
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-./models}"
BASE_MODEL_NAME="Qwen3-14B"
LORA_ADAPTER_NAME="redblackbench-qwen3-14b-sft-v2"

# vLLM server settings
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_MEMORY="${GPU_MEMORY:-0.9}"
MAX_LORA_RANK="${MAX_LORA_RANK:-128}"

# Experiment settings (use SFT version by default)
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-scripts/run_goal_experiment_sft.py}"

#=============================================================================
# COLORS FOR OUTPUT
#=============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

#=============================================================================
# SETUP FUNCTIONS
#=============================================================================

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        print_warning "nvidia-smi not found - GPU may not be available"
    fi
}

setup_environment() {
    print_header "Setting Up Python Environment"

    check_python
    check_gpu

    echo -e "\nInstalling dependencies..."

    # Upgrade pip
    python3 -m pip install --upgrade pip

    # Install sugarscape dependencies
    if [ -f "requirements-sugarscape.txt" ]; then
        pip install -r requirements-sugarscape.txt
        print_success "Installed requirements-sugarscape.txt"
    else
        print_warning "requirements-sugarscape.txt not found, installing from pyproject.toml"
        pip install -e ".[sugarscape]"
    fi

    # Install the package in editable mode
    pip install -e .
    print_success "Installed redblackbench package"

    # Install vLLM (may require specific CUDA version)
    echo -e "\nInstalling vLLM..."
    pip install vllm
    print_success "Installed vLLM"

    # Install huggingface_hub for model downloads
    pip install huggingface_hub[cli]
    print_success "Installed huggingface_hub"

    print_success "Environment setup complete!"
}

#=============================================================================
# MODEL DOWNLOAD FUNCTIONS
#=============================================================================

download_models() {
    print_header "Downloading Models from HuggingFace"

    # Create model directory
    mkdir -p "$LOCAL_MODEL_DIR"

    LOCAL_BASE_PATH="$LOCAL_MODEL_DIR/$BASE_MODEL_NAME"
    LOCAL_LORA_PATH="$LOCAL_MODEL_DIR/$LORA_ADAPTER_NAME"

    # Check if HF token is set for private repos
    if [ -z "$HF_TOKEN" ]; then
        print_warning "HF_TOKEN not set. If models are private, set: export HF_TOKEN=your_token"
    fi

    # Download base model
    echo -e "\nDownloading base model: $HF_BASE_MODEL"
    if [ -d "$LOCAL_BASE_PATH" ] && [ "$(ls -A $LOCAL_BASE_PATH 2>/dev/null)" ]; then
        print_warning "Base model already exists at $LOCAL_BASE_PATH"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping base model download"
        else
            rm -rf "$LOCAL_BASE_PATH"
            huggingface-cli download "$HF_BASE_MODEL" --local-dir "$LOCAL_BASE_PATH"
            print_success "Base model downloaded to $LOCAL_BASE_PATH"
        fi
    else
        huggingface-cli download "$HF_BASE_MODEL" --local-dir "$LOCAL_BASE_PATH"
        print_success "Base model downloaded to $LOCAL_BASE_PATH"
    fi

    # Download LoRA adapter
    echo -e "\nDownloading LoRA adapter: $HF_LORA_ADAPTER"
    if [ -d "$LOCAL_LORA_PATH" ] && [ "$(ls -A $LOCAL_LORA_PATH 2>/dev/null)" ]; then
        print_warning "LoRA adapter already exists at $LOCAL_LORA_PATH"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping LoRA adapter download"
        else
            rm -rf "$LOCAL_LORA_PATH"
            huggingface-cli download "$HF_LORA_ADAPTER" --local-dir "$LOCAL_LORA_PATH"
            print_success "LoRA adapter downloaded to $LOCAL_LORA_PATH"
        fi
    else
        huggingface-cli download "$HF_LORA_ADAPTER" --local-dir "$LOCAL_LORA_PATH"
        print_success "LoRA adapter downloaded to $LOCAL_LORA_PATH"
    fi

    # Summary
    echo -e "\n${GREEN}Model locations:${NC}"
    echo "  Base model: $LOCAL_BASE_PATH"
    echo "  LoRA adapter: $LOCAL_LORA_PATH"
}

#=============================================================================
# VLLM SERVER FUNCTIONS
#=============================================================================

check_server_running() {
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if check_server_running; then
            print_success "vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 5
    done

    print_error "Server did not start within timeout"
    return 1
}

start_server() {
    print_header "Starting vLLM Server"

    LOCAL_BASE_PATH="$LOCAL_MODEL_DIR/$BASE_MODEL_NAME"
    LOCAL_LORA_PATH="$LOCAL_MODEL_DIR/$LORA_ADAPTER_NAME"

    # Check if model exists
    if [ ! -d "$LOCAL_BASE_PATH" ]; then
        print_error "Base model not found at $LOCAL_BASE_PATH"
        print_error "Run './scripts/setup_and_run.sh download' first"
        exit 1
    fi

    # Check if server already running
    if check_server_running; then
        print_warning "vLLM server already running on port $VLLM_PORT"
        return 0
    fi

    echo "Starting vLLM server..."
    echo "  Base model: $LOCAL_BASE_PATH"
    echo "  Port: $VLLM_PORT"
    echo "  GPU Memory: $GPU_MEMORY"

    # Build command
    VLLM_CMD="vllm serve $LOCAL_BASE_PATH \
        --host $VLLM_HOST \
        --port $VLLM_PORT \
        --gpu-memory-utilization $GPU_MEMORY \
        --dtype bfloat16 \
        --trust-remote-code \
        --enable-prefix-caching \
        --max-model-len 4096"

    # Add LoRA support if adapter exists
    if [ -d "$LOCAL_LORA_PATH" ] && [ "$(ls -A $LOCAL_LORA_PATH 2>/dev/null)" ]; then
        echo "  LoRA adapter: $LOCAL_LORA_PATH"
        VLLM_CMD="$VLLM_CMD \
            --enable-lora \
            --max-lora-rank $MAX_LORA_RANK \
            --lora-modules $LORA_ADAPTER_NAME=$LOCAL_LORA_PATH"
    else
        print_warning "LoRA adapter not found, starting base model only"
    fi

    echo -e "\n${YELLOW}Server will start in foreground. Use Ctrl+C to stop.${NC}"
    echo -e "${YELLOW}For background mode, run: nohup ./scripts/setup_and_run.sh server &${NC}\n"

    # Start server
    eval $VLLM_CMD
}

start_server_background() {
    print_header "Starting vLLM Server (Background)"

    LOCAL_BASE_PATH="$LOCAL_MODEL_DIR/$BASE_MODEL_NAME"
    LOCAL_LORA_PATH="$LOCAL_MODEL_DIR/$LORA_ADAPTER_NAME"

    # Check if model exists
    if [ ! -d "$LOCAL_BASE_PATH" ]; then
        print_error "Base model not found at $LOCAL_BASE_PATH"
        exit 1
    fi

    # Check if server already running
    if check_server_running; then
        print_warning "vLLM server already running on port $VLLM_PORT"
        return 0
    fi

    # Build command
    VLLM_CMD="vllm serve $LOCAL_BASE_PATH \
        --host $VLLM_HOST \
        --port $VLLM_PORT \
        --gpu-memory-utilization $GPU_MEMORY \
        --dtype bfloat16 \
        --trust-remote-code \
        --enable-prefix-caching \
        --max-model-len 4096 \
        --disable-log-requests"

    # Add LoRA support if adapter exists
    if [ -d "$LOCAL_LORA_PATH" ] && [ "$(ls -A $LOCAL_LORA_PATH 2>/dev/null)" ]; then
        VLLM_CMD="$VLLM_CMD \
            --enable-lora \
            --max-lora-rank $MAX_LORA_RANK \
            --lora-modules $LORA_ADAPTER_NAME=$LOCAL_LORA_PATH"
    fi

    # Start in background
    LOG_FILE="vllm_server.log"
    echo "Starting server in background, logging to $LOG_FILE"
    nohup bash -c "$VLLM_CMD" > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > vllm_server.pid
    print_success "Server started with PID $SERVER_PID"

    # Wait for server to be ready
    wait_for_server
}

stop_server() {
    print_header "Stopping vLLM Server"

    if [ -f "vllm_server.pid" ]; then
        PID=$(cat vllm_server.pid)
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            rm vllm_server.pid
            print_success "Server stopped (PID $PID)"
        else
            print_warning "Server not running"
            rm vllm_server.pid
        fi
    else
        # Try to find by port
        PID=$(lsof -ti :$VLLM_PORT 2>/dev/null || true)
        if [ -n "$PID" ]; then
            kill $PID
            print_success "Server stopped (PID $PID)"
        else
            print_warning "No server found on port $VLLM_PORT"
        fi
    fi
}

#=============================================================================
# EXPERIMENT FUNCTIONS
#=============================================================================

run_experiment() {
    print_header "Running Sugarscape Experiment"

    # Check if server is running
    if ! check_server_running; then
        print_error "vLLM server is not running on port $VLLM_PORT"
        print_error "Start the server first: ./scripts/setup_and_run.sh server"
        exit 1
    fi

    print_success "vLLM server is running"

    # Check if experiment script exists
    if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
        print_error "Experiment script not found: $EXPERIMENT_SCRIPT"
        exit 1
    fi

    echo "Running: python3 $EXPERIMENT_SCRIPT"
    python3 "$EXPERIMENT_SCRIPT"

    print_success "Experiment complete!"
}

#=============================================================================
# ALL-IN-ONE FUNCTION
#=============================================================================

run_all() {
    print_header "All-in-One Setup and Run"

    echo "This will:"
    echo "  1. Install Python dependencies"
    echo "  2. Download models from HuggingFace"
    echo "  3. Start vLLM server (background)"
    echo "  4. Run the sugarscape experiment"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    setup_environment
    download_models
    start_server_background
    run_experiment

    print_header "All Done!"
    echo "Results are in: results/sugarscape/"
    echo "Server log: vllm_server.log"
    echo "To stop server: ./scripts/setup_and_run.sh stop"
}

#=============================================================================
# HELP
#=============================================================================

show_help() {
    echo "Sugarscape Experiment Setup Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  setup     Install Python dependencies (pip packages)"
    echo "  download  Download models from HuggingFace"
    echo "  server    Start vLLM server (foreground)"
    echo "  start     Start vLLM server (background)"
    echo "  stop      Stop background vLLM server"
    echo "  run       Run sugarscape experiment (SFT v2 model)"
    echo "  run-base  Run sugarscape experiment (base model)"
    echo "  all       Do everything (setup + download + start + run)"
    echo "  status    Check if vLLM server is running"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_BASE_MODEL     HuggingFace path for base model (default: $HF_BASE_MODEL)"
    echo "  HF_LORA_ADAPTER   HuggingFace path for LoRA adapter"
    echo "  LOCAL_MODEL_DIR   Local directory for models (default: $LOCAL_MODEL_DIR)"
    echo "  VLLM_PORT         vLLM server port (default: $VLLM_PORT)"
    echo "  GPU_MEMORY        GPU memory fraction (default: $GPU_MEMORY)"
    echo "  HF_TOKEN          HuggingFace token (for private repos)"
    echo ""
    echo "Examples:"
    echo "  # Full setup with custom model paths"
    echo "  HF_BASE_MODEL=Qwen/Qwen2.5-14B-Instruct \\"
    echo "  HF_LORA_ADAPTER=myuser/my-lora \\"
    echo "  $0 all"
    echo ""
    echo "  # Just start the server"
    echo "  $0 start"
    echo ""
    echo "  # Run experiment (server must be running)"
    echo "  $0 run"
}

show_status() {
    print_header "Status Check"

    echo "Configuration:"
    echo "  HF Base Model: $HF_BASE_MODEL"
    echo "  HF LoRA Adapter: $HF_LORA_ADAPTER"
    echo "  Local Model Dir: $LOCAL_MODEL_DIR"
    echo "  vLLM Port: $VLLM_PORT"
    echo ""

    LOCAL_BASE_PATH="$LOCAL_MODEL_DIR/$BASE_MODEL_NAME"
    LOCAL_LORA_PATH="$LOCAL_MODEL_DIR/$LORA_ADAPTER_NAME"

    echo "Models:"
    if [ -d "$LOCAL_BASE_PATH" ]; then
        print_success "Base model: $LOCAL_BASE_PATH"
    else
        print_warning "Base model not downloaded"
    fi

    if [ -d "$LOCAL_LORA_PATH" ]; then
        print_success "LoRA adapter: $LOCAL_LORA_PATH"
    else
        print_warning "LoRA adapter not downloaded"
    fi

    echo ""
    echo "Server:"
    if check_server_running; then
        print_success "vLLM server running on port $VLLM_PORT"
    else
        print_warning "vLLM server not running"
    fi
}

#=============================================================================
# MAIN
#=============================================================================

# Change to repo root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

case "${1:-help}" in
    setup)
        setup_environment
        ;;
    download)
        download_models
        ;;
    server)
        start_server
        ;;
    start)
        start_server_background
        ;;
    stop)
        stop_server
        ;;
    run)
        run_experiment
        ;;
    run-base)
        EXPERIMENT_SCRIPT="scripts/run_goal_experiment.py" run_experiment
        ;;
    all)
        run_all
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
