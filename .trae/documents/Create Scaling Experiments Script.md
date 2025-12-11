I will create the `experiments/run_scaling_experiments.py` script using the exact model IDs you provided.

**Model Mapping:**

* **GPT-5 Thinking**: `openai/gpt-5.1`

* **Gemini 3 Pro**: `google/gemini-3-pro-preview`

* **Kimi K2 Thinking**: `moonshotai/kimi-k2-thinking`

* **Qwen 3 (235B)**: `qwen/qwen3-vl-235b-a22b-thinking`

**Script Workflow:**

1. **Pre-flight Verification**: The script will first run `redblackbench provider-check` for each of the 4 models using your API key (from environment variables) to confirm access.
2. **Parallel Execution**: Once verified, it will generate the 20 configuration files (4 models × 5 team sizes) and launch the experiments in parallel.
3. **Output**: Results will be organized by model and team size.

<br />

i will cross reference the eval yamls.

i will enable thinking of the model

i will make sure they have enough max token limit, i dont care about budget

I will implement this in `experiments/run_scaling_experiments.py`.
