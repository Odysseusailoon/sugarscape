# Changelog - GPT - 2026-01-23

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

