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

