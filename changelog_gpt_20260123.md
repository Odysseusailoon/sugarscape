# Changelog - GPT - 2026-01-23

## Feature: Resource Specialization for Meaningful Trade

**Problem**: With independent random metabolism (both 1-4), 67% of agents end up "balanced" (difference < 2), meaning they have similar demand for both resources. This makes trade pointless - Cobb-Douglas utility is maximized at balance, so trading away from balance hurts welfare.

**Solution**: Added `enable_resource_specialization` config option (default: True) that creates complementary demand:
- **Sugar specialists** (50%): High sugar metabolism (3-4), low spice metabolism (1-2)
  - Will want to trade their excess Spice for Sugar
- **Spice specialists** (50%): High spice metabolism (3-4), low sugar metabolism (1-2)
  - Will want to trade their excess Sugar for Spice

**Files Changed**:
- **`redblackbench/sugarscape/config.py`**:
  - Added `enable_resource_specialization: bool = True`
  - Added `specialization_ratio: float = 0.5` (fraction of Sugar specialists)
  - Added `specialization_high_metabolism: Tuple[int, int] = (3, 4)`
  - Added `specialization_low_metabolism: Tuple[int, int] = (1, 2)`

- **`redblackbench/sugarscape/simulation.py`**:
  - Modified `_create_agent()` to implement specialization logic when enabled
  - Falls back to original independent random metabolism when disabled

**Impact**: Trade now creates real value through comparative advantage!

---

## Critical Fix: Death Record Statistics Were Always Zero

**Problem**: Death records showed `gathered sugar=0, spice=0, cells visited=0` for all agents, making it appear that agents were not moving or harvesting. This was a **logging/statistics bug**, not a simulation bug.

**Root Cause**: The `debug_logger.init_agent()` and `debug_logger.update_agent_harvest()` methods existed but were **never called**, so `_agent_lifetime_stats` was always empty.

**Files Changed**:
- **`redblackbench/sugarscape/simulation.py`**:
  - Added call to `debug_logger.init_agent(agent_id)` when creating new agents
  - Added tracking of harvest amounts in Phase 1a (standard agents) and Phase 1b (LLM agents with rule_based_movement)
  - Added tracking of harvest amounts for LLM-based movement mode (Step 5)
  - Fixed `DeathRecord` creation to include `unique_cells_visited` and `max_displacement` from agent state

**Verification**: Movement and harvest work correctly - test showed 10/10 agents moved and harvested 204 sugar + 231 spice over 10 ticks.

## Fixes (review-driven)

- Fixed baseline questionnaire prompt text for non-zero ticks (no longer claims "no interactions" after T=0).
- Hardened `parse_questionnaire_response()` to extract the largest decodable JSON object when models omit ```json fences.
- Baseline questionnaire parsing now strips thinking blocks before JSON extraction (reduces parse failures on thinking models).
- Event-triggered reflection no longer clears pending events before the LLM call; events are preserved on transient provider failures.
- `resources_critical` reflection events now use the real simulation tick (passed from `SugarSimulation`) instead of falling back to 0.
- Added `decider_id/decider_name` to `NO_TRADE` logs to disambiguate who rejected/walked away/excluded, enabling correct evaluator attribution.
- Fixed behavioral evaluator exclusion attribution by accepting both `NO_TRADE.outcome="EXCLUSION"` and `"EXCLUDED"` (prevents undercounting `excluded_partners/was_excluded`).

## External Moral Evaluator (per reflection moment)

- Added `redblackbench/sugarscape/moral_evaluator.py`: an external “judge” LLM that scores moral dimensions from **agent prompts + agent raw replies + (for trades) full conversation transcript including smalltalk**, returning high-variance 0-100 scores, polarity, and overall.
- Added config switches/knobs in `redblackbench/sugarscape/config.py`:
  - `enable_external_moral_evaluation`, `external_moral_evaluator_model/provider`
  - `moral_overall_transform/moral_overall_tanh_k` and `moral_self_tanh_k` to increase curve variance.
- Extended `redblackbench/sugarscape/debug_logger.py` with:
  - `MoralEvalRecord`, `moral_evals.jsonl` (full audit input/output), and `moral_scores.csv` (curve-ready self vs external).
- Hooked external evaluation into:
  - T=0 baseline questionnaire (`LLMSugarAgent.async_baseline_questionnaire`)
  - Every post-encounter reflection (`DialogueTradeSystem._reflect_agent`, includes full transcript)
  - Event-triggered identity reviews (`LLMSugarAgent.async_event_reflection`)
  - End-of-life reports (`LLMSugarAgent.async_end_of_life_report`)

## Ablations: chat-only encounters + protocol-only trading

- Added config flags in `redblackbench/sugarscape/config.py`:
  - `encounter_protocol_mode`: `"full" | "chat_only" | "protocol_only"`
  - `enable_encounter_dialogue`: run encounter dialogue even when `enable_trade=False`
- Updated `redblackbench/sugarscape/simulation.py` to instantiate/run `DialogueTradeSystem` when encounters are enabled (even without trade), and to support `trade_mode="protocol"` routing.
- Updated `redblackbench/sugarscape/trade.py`:
  - `chat_only`: runs small talk + post-encounter reflection, records `NO_TRADE` with `CHAT_ONLY`, no negotiation/transfer
  - `protocol_only`: skips small talk + trade intent; runs JSON-only negotiation prompts (no MESSAGE), still uses the same reflection system
  - Added small-talk-aware conversation highlights for reflection
- Updated `redblackbench/sugarscape/prompts.py`:
  - Identity/event reflections suppress trade-specific phrasing when `enable_trade=False`
  - Post-encounter reflection prompt supports `interaction_domain="trade"|"social"`
  - Negotiation system prompt supports `protocol_only=True` (no speaking; JSON-only output)

## Last Words Feature (witnessed_death enhancement)

- **`redblackbench/sugarscape/simulation.py`**: When an agent dies and has generated an end-of-life report, their "last words" (advice or final reflection excerpt) are now extracted and passed to nearby agents as part of the `witnessed_death` event.
  - Uses `advice` field from end-of-life report as last words
  - Falls back to first 200 chars of `final_reflection` if no advice provided
  - Non-LLM agents or agents without end-of-life reports have no last words (graceful fallback)

- **`redblackbench/sugarscape/llm_agent.py`**: Updated `get_pending_events_summary()` to format last words when displaying `witnessed_death` events in reflection prompts.
  - Format: `"- Tick X: You witnessed NAME's death (CAUSE)\n  Their last words: \"...\"`
  - Last words provide emotional/philosophical context for the surviving agent's reflection

## Docs

- Updated `redblackbench/sugarscape/EXPERIMENTS.md` to map the experiment suite to an ICML paper narrative (“seed & soil” + mechanism/ablation), revise the mixed-society outcome from one-way “corruption” to **redemption-dominant / survival-buffer** framing, and add a paper-first visualization plan (forced alignment, catalyst effect, price of anonymity).
- Extended `redblackbench/sugarscape/EXPERIMENTS.md` with **Exp 9–10 “Seed Validation”** using a RedBlack-stage SFT-aligned model as a seed: Exp 9 (SFT Redemption: 20% SFT + 80% exploiter) and Exp 10 (SFT Enlightenment: 20% SFT + 80% normie), plus associated metrics (seed exposure / conditional uplift), hypotheses (H6), expected outcomes, matrix rows, and visualization recommendations.
- **Detailed System Documentation**: Added comprehensive descriptions of the **Moral Scoring System** (6 dimensions, penalty logic), **Trust/Reputation System** (Global vs. Personal), and **Belief Evolution System** (Identity Review, Event-Triggered Reflection) to `EXPERIMENTS.md`.
- **Results Tables**: Inserted **Table 1** (Soil Effect), **Table 2** (Seed Effect), **Table 3** (Enlightenment Speed), and **Table A1** (Cause of Death Analysis) into `EXPERIMENTS.md`.
- **New Normie Ablations**: Added **Experiments 11-14** to `EXPERIMENTS.md` to investigate the role of dialogue (Protocol Only vs. Functional vs. Full) and reputation scope (Local vs. Global) in normie societies.
  - **Exp 11**: Protocol Only (No Dialogue).
  - **Exp 12**: Functional (No Small Talk).
  - **Exp 13**: Local Reputation Only.
  - **Exp 14**: Global Reputation Only.

## Analysis Script

- Updated `scripts/analyze_formal_experiments.py` to compute additional metrics required for the paper tables:
  - **Identity Shift**: Added `identity_shift_metrics` to compute Δ `self_identity_leaning` (Table 3).
  - **Cooperation Speed**: Added `t90` (Time to 90% Cooperation) to `propagation_metrics` (Table 3).
  - **Cause of Death**: Added `cause_of_death_metrics` to compute deception/trust stats by survival group (Table A1).

## vLLM + LoRA usability

- Fixed `SugarSimulation` to actually honor `SugarscapeConfig.llm_vllm_base_url` when constructing vLLM-backed providers (agents/evaluator/moral evaluator and checkpoint restore path).
- Updated `VLLMProvider.generate()` to accept common extra kwargs like `max_tokens` / `chat_template_kwargs` (call-sites already pass these).
- Updated `scripts/run_goal_experiment_sft.py` to work with any vLLM-served LoRA adapter alias (default now matches `scripts/setup_and_run.sh`’s `LORA_ADAPTER_NAME=Qwen3-14B-LoRA`, override via `--model` or `SFT_LORA_NAME`).
