## Goal

Implement a turn-based opinion phase where agents iteratively decide willingness to speak (0–3), the highest-willing agent speaks first, ties are broken randomly, and the process repeats until every agent has spoken exactly once; then proceed to a simultaneous final vote.

## Key Changes

* Replace concurrent initial opinions with an iterative, willingness-driven speaking order.

* Preserve simultaneous final vote phase.

* Record per-turn opinion messages and the final votes in trajectory.

## API Additions

* Add `get_willingness_to_speak(round_context: dict, team_identifier: str, seen_messages: list[dict]) -> int` to `BaseAgent` (`redblackbench/agents/base.py`).

*  Add `build_willingness_prompt(...)` in `prompts.py` for LLM agents.

* LLMAgent implements `get_willingness_to_speak` using a new prompt format that outputs `WILLINGNESS: [0-3]`.

## Prompt Updates

* New prompt template:

  * System: unchanged

  * Willingness: asks for `WILLINGNESS: 0|1|2|3` given current round context and latest team channel messages.

  * Speaking message: reuse existing initial opinion prompt to produce `RECOMMENDATION` and `REASONING` (this becomes the "words" the agent speaks).

## Deliberation Flow

* Modify `Deliberation._gather_initial_opinions` (`redblackbench/teams/deliberation.py`) to:

  1. Initialize `spoken = set()` and `team_channel = []` (list of dicts with `agent_id`, `message`, `round_num`, `order`).
  2. While `len(spoken) < len(agents)`:

     * Concurrently collect willingness from all agents not in `spoken` using `get_willingness_to_speak(round_context, team_identifier, team_channel)`.

     * Find max willingness value; tie-break with `random.choice` among tied agent IDs.

     * Selected agent produces their opinion via existing `get_initial_opinion(...)`; append to `team_channel` and mark as spoken.

     * Record this turn in trajectory (see next section).
  3. Return the ordered list of `AgentResponse` (one per agent) for compatibility with existing downstream code.

* Keep `Deliberation._gather_final_votes` unchanged (still concurrent), but pass teammate opinions if desired.

* `Deliberation.deliberate(...)` continues to call the two phases.

## Coordinator & Trajectory

* In `_deliberate_with_trajectory` (`redblackbench/game/coordinator.py`):

  * Still call the deliberation phase, but:

    * After each speaking turn, call `TrajectoryCollector.add_timestep` with `timestep_type=INITIAL_OPINIONS` and an `ActionRecord` using `action_type="individual_opinion"`, `phase="opinion_turn"`, and the order index.

    * At final votes, keep current `record_final_votes` behavior.

* Update `TrajectoryCollector.record_initial_opinions` (`redblackbench/trajectory/collector.py`) to support ordered turns:

  * Accept the ordered `AgentResponse` list but emit one `ActionRecord` per response using `action_type="individual_opinion"` and `phase="opinion_turn"` with a `turn_index` in metadata.

  * Leave existing fields intact for compatibility.

## Random Tie-Breaking

* Use Python `random.choice` for ties.

* Optionally accept `seed` in `GameConfig` for reproducibility; if present, seed `random` at game start.

## Metrics

* No change to scoring or final metrics.

* Trajectory now includes granular opinion turns, enabling analysis of turn order effects.

## Backward Compatibility

* `Team.make_choice` returns the same type (`Choice`).

* `DeliberationResult.initial_opinions` remains `List[AgentResponse]` so existing code still works.

* Tests relying on `initial_opinions` continue to pass; new tests will check sequencing and willingness behavior.

## Tests

* Add tests in `tests/test_agents.py` for `LLMAgent.get_willingness_to_speak` parsing `WILLINGNESS: N`.

* Add tests in `tests/test_game.py` or `tests/test_trajectory.py` to verify:

  * All agents speak exactly once in opinion phase.

  * Tie-breaking randomness selects among top willingness.

  * Trajectory contains ordered `individual_opinion` actions.

## Security & Performance

* Respect existing provider usage and env vars for API keys.

* Keep willingness calls short prompts to limit token usage.

* Concurrency only for willingness polling and final votes; speaking is sequential.

## Acceptance Criteria

* Opinion phase executes per spec with willingness scores and single speaking per agent.

* Final vote remains simultaneous.

* Unit tests cover willingness parsing and turn ordering.

* Trajectory records show ordered opinion turns and final votes.

