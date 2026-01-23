# Changelog - Claude - 2026-01-22

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
