I will implement LLM support and history optimization while maintaining backward compatibility with existing configurations and experiments.

### 1. Backward-Compatible Agent Updates
I will modify `redblackbench/sugarscape/agent.py`:
- **State**: Initialize `self.recent_history = deque(maxlen=15)` in `__post_init__`. This avoids changing the `__init__` signature, preserving compatibility with existing instantiation logic.
- **Logic**:
    - Keep `visited_cells` for the `unique_visited` metric (preserving reporting compatibility).
    - Update **Nomad (Persona C)** to use `recent_history` for its novelty bonus calculation, implementing the requested optimization.

### 2. `LLMSugarAgent` Implementation
I will create `redblackbench/sugarscape/llm_agent.py`:
- **Class**: Inherits from `SugarAgent` to ensure it works within the existing `SugarEnvironment` without modification.
- **Features**:
    - Uses `recent_history` for the prompt context.
    - Supports V1 (Sugar) and V2 (Sugar+Spice) modes dynamically.
    - Configurable Goal prompts (Survival/Utility).

### 3. Infrastructure Fixes
- **OpenRouter**: Implement `_call_api_with_retry` in `redblackbench/providers/openrouter_provider.py` to fix the detected bug.

### 4. Configuration
I will update `redblackbench/sugarscape/config.py` with optional LLM settings:
- `enable_llm: bool = False` (Default False preserves existing behavior).
- `llm_agent_ratio: float = 0.0`.
- `llm_history_limit: int = 15`.

### 5. Verification
- Verify that standard agents (Persona A-D) still function correctly (re-run `reproduce_nomad.py` logic).
- Verify LLM agents can be instantiated and generate prompts.
