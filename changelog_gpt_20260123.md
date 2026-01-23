# Changelog - GPT - 2026-01-23

## Fixes (review-driven)

- Fixed baseline questionnaire prompt text for non-zero ticks (no longer claims "no interactions" after T=0).
- Hardened `parse_questionnaire_response()` to extract the largest decodable JSON object when models omit ```json fences.
- Baseline questionnaire parsing now strips thinking blocks before JSON extraction (reduces parse failures on thinking models).
- Event-triggered reflection no longer clears pending events before the LLM call; events are preserved on transient provider failures.
- `resources_critical` reflection events now use the real simulation tick (passed from `SugarSimulation`) instead of falling back to 0.
- Added `decider_id/decider_name` to `NO_TRADE` logs to disambiguate who rejected/walked away/excluded, enabling correct evaluator attribution.

