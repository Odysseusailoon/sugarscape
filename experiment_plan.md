# Sugarscape LLM Experiment Plan

## 🎯 Experiment Design

**Objective**: Compare how different LLM models and goals perform in resource-constrained environments with trading.

### Variables
- **Models**: Kimi-K2 (SOTA), Qwen3-30B-A3B-Thinking-2507 (Medium)
- **Goals**: Survival (self-preserving), Egalitarian (society-focused)
- **Environments**: Standard (normal resources), Harsh (scarce resources)
- **Fixed Parameters**: 100 ticks, 100 agents, trade-rounds=2, seed=42

### Experiment Matrix (10 experiments)
| Exp | Agent Type | Goal | Environment | Command |
|-----|------------|------|-------------|---------|
| 1 | Kimi-K2 | Survival | Standard | `run_exp 1` |
| 2 | Kimi-K2 | Survival | Harsh | `run_exp 2` |
| 3 | Kimi-K2 | Egalitarian | Standard | `run_exp 3` |
| 4 | Kimi-K2 | Egalitarian | Harsh | `run_exp 4` |
| 5 | Qwen3-30B | Survival | Standard | `run_exp 5` |
| 6 | Qwen3-30B | Survival | Harsh | `run_exp 6` |
| 7 | Qwen3-30B | Egalitarian | Standard | `run_exp 7` |
| 8 | Qwen3-30B | Egalitarian | Harsh | `run_exp 8` |
| 9 | Basic Agents | Survival | Standard | `run_exp 9` |
| 10 | Basic Agents | Egalitarian | Standard | `run_exp 10` |

## 🚀 Quick Commands

```bash
# Set your API key first
export OPENROUTER_API_KEY='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'

# Run individual experiments
./run_experiments.sh 1    # Kimi-K2 + Survival + Standard
./run_experiments.sh 5    # Qwen3-30B + Survival + Standard
./run_experiments.sh 9    # Basic Agents + Survival + Standard

# Run all 10 experiments in parallel (recommended)
./run_all_parallel.sh

# Or run sequentially
./run_experiments.sh all
```

## 📊 Results Table

| Exp | Agent Type | Goal | Environment | Survival Rate | Mean Welfare | Utilitarian Welfare | Nash Welfare | Rawlsian Welfare | Welfare Gini | Gini-Adjusted Welfare |
|-----|------------|------|-------------|---------------|--------------|-------------------|---------------|------------------|--------------|----------------------|
| 1   | Kimi-K2 | Survival | Standard |     |     |     |     |     |     |     |
| 2   | Kimi-K2 | Survival | Harsh    |     |     |     |     |     |     |     |
| 3   | Kimi-K2 | Egalitarian | Standard |     |     |     |     |     |     |     |
| 4   | Kimi-K2 | Egalitarian | Harsh    |     |     |     |     |     |     |     |
| 5   | Qwen3-30B | Survival | Standard |     |     |     |     |     |     |     |
| 6   | Qwen3-30B | Survival | Harsh    |     |     |     |     |     |     |     |
| 7   | Qwen3-30B | Egalitarian | Standard |     |     |     |     |     |     |     |
| 8   | Qwen3-30B | Egalitarian | Harsh    |     |     |     |     |     |     |     |
| 9   | Basic | Survival | Standard |     |     |     |     |     |     |     |
| 10  | Basic | Egalitarian | Standard |     |     |     |     |     |     |     |

## 💰 Token Cost Estimate (Final)

### Model Pricing (OpenRouter)
**Kimi-K2 Thinking** (SOTA):
- Input: $0.60 per million tokens
- Output: $2.50 per million tokens
- Context: 262K tokens (very large)
- **Cost per experiment**: ~$12.87

**Qwen3-30B-A3B-Thinking-2507** (Efficient):
- Input: $0.075 per million tokens
- Output: $0.28 per million tokens
- Context: ~128K tokens
- **Cost per experiment**: ~$1.50

### Additional Verified Thinking Models
- **Qwen3-235B**: Large model, verified working
- **Qwen3-8B**: Small model, verified working
- **Qwen3-80B**: Balanced model, verified working
- **Ernie-21B**: Baidu model, verified working
- **GLM-4.1V**: Tsinghua model, verified working

### Total Cost Breakdown
- **Kimi-K2 experiments (4)**: 4 × $12.87 = **$51.48**
- **Qwen3-30B experiments (4)**: 4 × $1.50 = **$6.00**
- **Basic agents experiments (2)**: 2 × $0 = **$0.00**
- **Grand total**: **$57.48** (~$5.75 per experiment average)

### Cost Optimization
- **50 ticks**: ~$28.74 total
- **50 agents**: ~$14.37 total
- **No trading**: Saves ~20%

## 🔧 Setup Commands

```bash
# 1. Install dependencies
pip install matplotlib seaborn openai

# 2. Set API key
export OPENROUTER_API_KEY='your-key-here'

# 3. Make scripts executable
chmod +x run_experiments.sh

# 4. Run a test experiment first
python scripts/run_sugarscape.py \
  --mode llm \
  --model moonshotai/kimi-k2-thinking \
  --goal-preset survival \
  --difficulty standard \
  --ticks 10 \
  --population 10 \
  --trade-rounds 2
```

## 📈 Analysis Framework

### Key Questions
1. **Model Performance**: Does Kimi-K2 create better outcomes than Qwen-72B?
2. **Goal Effectiveness**: Do "egalitarian" agents actually reduce inequality?
3. **Environmental Stress**: How does harsh environment affect different combinations?
4. **Trade Dynamics**: How do goals influence negotiation success?

### Metrics to Compare
- **Efficiency**: `utilitarian_welfare`, `average_welfare`
- **Equity**: `rawlsian_welfare`, `welfare_gini`
- **Balance**: `nash_welfare`, `gini_adjusted_welfare`
- **Robustness**: `survival_rate`, population stability

## 📁 Output Files

Each experiment creates:
- `results/sugarscape/experiment_*/metrics.csv` - Time series data
- `results/sugarscape/experiment_*/plots/` - Welfare visualizations
- `results/sugarscape/experiment_*/config.json` - Experiment configuration

## 🎯 Expected Insights

- **Model differences**: Reasoning quality impact on social behavior
- **Goal alignment**: Whether agents follow their programmed motivations
- **Environment effects**: How scarcity changes strategy effectiveness
- **Trading behavior**: Goal influence on negotiation and cooperation

Ready to run experiments! Start with `./run_experiments.sh 1` 🚀
