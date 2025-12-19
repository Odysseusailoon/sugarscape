I will update `scripts/run_sugarscape.py` to support command-line arguments for choosing between "basic" and "llm" agent modes.

### 1. Argument Parsing
I will add `argparse` to `scripts/run_sugarscape.py` to support the following flags:
*   `--mode`: Choice between `basic` (default) and `llm`.
*   `--ticks`: Number of simulation ticks (default: 100).
*   `--population`: Initial population size (default: 50 for speed).
*   `--model`: LLM model ID (default: `thudm/glm-4.1v-9b-thinking` as tested).
*   `--ratio`: Ratio of LLM agents (default: 1.0 if mode is `llm`).
*   `--goal`: Custom goal prompt for LLM agents.

### 2. Configuration Logic
I will implement logic to construct the `SugarscapeConfig` object based on these arguments:
*   **Basic Mode**: Uses default configuration.
*   **LLM Mode**: Sets `enable_llm_agents=True`, uses the provided model/ratio, and enables spice (V2) by default as it's more interesting for LLMs.

### 3. Execution Flow
The script will initialize the `SugarSimulation` with the custom config and run the loop, printing stats as before.

### Example Usage
Once implemented, you will be able to run:
*   **Basic**: `python3 scripts/run_sugarscape.py --mode basic`
*   **LLM**: `python3 scripts/run_sugarscape.py --mode llm --model thudm/glm-4.1v-9b-thinking`
