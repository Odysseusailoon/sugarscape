# Changelog - Claude - 2026-01-22

## Unified Event-Triggered Identity Review System

**Purpose:** Consolidate periodic Identity Review and Event-Triggered Reflection into a single event-driven system. Identity reviews now only happen when significant events occur, providing more meaningful and contextual reflection.

### Changes:

1. **`redblackbench/sugarscape/prompts.py`**
   - Renamed section from "EVENT-TRIGGERED REFLECTION PROMPT" to "EVENT-TRIGGERED IDENTITY REVIEW PROMPT"
   - Upgraded `build_event_triggered_reflection_prompt()` to full identity review:
     - Added `recent_interactions` parameter for trade context
     - Added full resource status (days supply, age)
     - Added deep reflection questions about identity and values
     - Added `IDENTITY_ASSESSMENT` output (strongly_altruist → strongly_exploiter)
     - Added `identity_shift` (±0.3) in JSON output
     - Added `core_identity_update` option (can change core goal)

2. **`redblackbench/sugarscape/llm_agent.py`**
   - Updated `async_event_triggered_reflection()` to be a full identity review:
     - Added `identity_before` / `identity_after` tracking
     - Added recent interactions gathering from trade memory
     - Added retry loop with format validation (same as periodic review)
     - Now stores in `identity_review_history` (unified with periodic)
     - Sets `last_identity_review_tick` for tracking

3. **`redblackbench/sugarscape/config.py`**
   - Changed `enable_identity_review` default: `True` → `False` (periodic disabled)
   - Added `enable_event_triggered_identity_review: bool = True` (recommended)
   - Updated comments to explain event-driven approach

4. **`redblackbench/sugarscape/simulation.py`**
   - Updated Phase 2.6 to check `enable_event_triggered_identity_review` config
   - Added comments listing trigger events

### Trigger Events (unchanged):
- `defrauded`: Agent was cheated in a trade
- `successful_cooperation`: Completed a mutually beneficial trade
- `resources_critical`: Resources dropped to critical level
- `trade_rejected`: Trade proposal was rejected
- `witnessed_death`: Observed another agent die nearby

### Benefits:
- More meaningful reflections tied to actual experiences
- Reduced LLM calls (no wasteful periodic reviews when nothing happened)
- Core identity can evolve at any significant moment, not just every N ticks

---

## Added Real-time Single Trade Evaluation

**Purpose:** Evaluate each trade as it happens (not just at simulation end), enabling per-trade fairness/cooperation scoring for real-time tracking.

### New Files/Functions:

1. **`redblackbench/sugarscape/evaluator.py`**
   - Added `SingleTradeEvaluation` dataclass with objective and LLM-based metrics
   - Added `compute_objective_trade_fairness()` - computes fairness without LLM
   - Added `evaluate_single_trade()` - async function to evaluate a single trade
   - Added `build_single_trade_prompt()` - builds LLM prompt for trade evaluation
   - Added `parse_single_trade_response()` - parses LLM evaluation response
   - Added `single_trade_eval_to_dict()` - JSON serialization helper

2. **`redblackbench/sugarscape/config.py`**
   - Added `enable_realtime_trade_eval: bool = False` - enable per-trade evaluation
   - Added `realtime_trade_eval_use_llm: bool = False` - use LLM (expensive) or objective-only

3. **`redblackbench/sugarscape/debug_logger.py`**
   - Added `SingleTradeEvalRecord` dataclass
   - Added `log_single_trade_eval()` method to DebugLogger
   - Added `single_trade_evals.csv` and `single_trade_evals.jsonl` output files
   - Updated `get_summary()` to include single trade eval statistics

4. **`redblackbench/sugarscape/trade.py`**
   - Added `_evaluate_single_trade()` async method to DialogueTradeSystem
   - Integrated evaluation calls after both `_record_trade()` locations

### Output Metrics (per trade):

**Objective (computed without LLM):**
- `trade_fairness_objective`: -1 to 1 scale (negative = unfair to acceptor)
- `net_transfer_ratio`: 0 to 1 (1.0 = perfectly balanced exchange)
- `urgency_exploitation`: True if CRITICAL party got bad deal

**LLM-based (optional, 1-7 scale):**
- `trade_fairness`: Was this trade fair?
- `cooperation_signal`: Does this show cooperation or self-interest?
- `brief_reason`: 1-sentence explanation

### Usage:

```python
config = SugarscapeConfig(
    enable_trade=True,
    enable_realtime_trade_eval=True,     # Enable per-trade evaluation
    realtime_trade_eval_use_llm=False,   # False = objective metrics only (fast/cheap)
)
```

### Output Files:
- `single_trade_evals.csv` - Summary of all trade evaluations
- `single_trade_evals.jsonl` - Full details with all fields

---

## Added T=0 Baseline Belief Capture

**Purpose:** Prove that value changes emerge from interactions, not pre-existing in LLM. Addresses reviewer concern: "How do you know these LLMs weren't already like this?"

### Changes:

1. **`redblackbench/sugarscape/agent.py`**
   - Added `baseline_snapshot: Dict[str, Any]` field to `SugarAgent`
   - Added `capture_baseline(tick)` method to snapshot initial beliefs
   - Updated `to_checkpoint_dict()` and `restore_from_checkpoint()` to include baseline

2. **`redblackbench/sugarscape/simulation.py`**
   - Added `_capture_baselines()` method to capture all agents' baseline at T=0
   - Modified `run()` to call `_capture_baselines()` + identity_review at T=0 before any interactions
   - Saves baseline to `baseline_beliefs.json` in experiment output

### Usage:
- Baseline is automatically captured before any step() is called
- Compare `baseline_beliefs.json` (T=0) with later `identity_review_history` to show evolution
- Key metrics: `belief_ledger`, `self_identity_leaning`, `policy_list`

---

## Changed Default: enable_origin_identity = True

**File:** `redblackbench/sugarscape/config.py`

Changed `enable_origin_identity` default from `False` to `True` so that:
- T=0 identity_review runs automatically
- Agents get origin identity (altruist/exploiter/survivor) by default
- Baseline beliefs are captured with identity context

---

## Removed Survival Alert Policy-Fixing Messages

**File:** `redblackbench/sugarscape/prompts.py`

### Changes:

1. **Identity Review Prompt (build_identity_review_prompt)**
   - Removed `survival_warning` block that told agents "HARD TRUTH: Your current strategy is NOT working" and suggested fixing policies
   - Removed `trade_analysis` block warning about policies causing trade rejections
   - Simplified reflection questions - removed policy-fixing prompts

2. **Post-Encounter Reflection (build_post_encounter_reflection_prompt)**
   - Removed `survival_context` that told agents to reconsider their policies when low on resources
   - Removed `outcome_guidance` that asked agents to question their policies after failed trades

3. **Trade Intent Decision (build_trade_intent_prompt)**
   - Removed `survival_warning` that pushed agents to trade when resources were low

4. **Negotiation Prompt (build_negotiation_user_prompt)**
   - Removed `survival_warning` that told agents "A bad trade is better than no trade"

### Rationale:
These warnings were biasing agent behavior by explicitly telling them to change their policies when facing survival pressure. Removing them allows agents to make more autonomous decisions based on their own goals and identity, rather than being guided by the simulation framework.

---

## Enabled Independent LLM Evaluation

**Files:**
- `redblackbench/sugarscape/config.py`
- `redblackbench/sugarscape/simulation.py`
- `redblackbench/sugarscape/evaluator.py`

### Changes:
1. **Config**: Added `enable_llm_evaluation` (default True) and `llm_evaluator_model` (default "openai/gpt-4o-mini").
2. **Simulation**: Added `_run_evaluation()` method called at simulation end.
3. **Evaluator**: Updated `compute_behavioral_metrics` to use `decider_id` in `NO_TRADE` events for accurate rejection attribution.

### Purpose:
Provides independent, objective assessment of agent behavior (trustworthiness, cooperativeness, etc.) using a separate LLM model, avoiding self-report bias.
