Based on the requirement to support "Thinking" models via OpenRouter and hide the thinking process from other agents, I have a clear plan.

## **Goal**

1. **Support Thinking Models:** Update `OpenRouterProvider` to request `include_reasoning=True` from OpenRouter.
2. **Capture Reasoning:** Extract the `reasoning` field from the API response (distinct from `content`).
3. **Log but Hide:**

   * Log the reasoning internally (e.g., to trajectory file) so we can analyze it.

   * **Do NOT** include the reasoning in the returned `content` that gets broadcast to the team channel. The user explicitly said: *"log the thinking but dont let other team members know those thinking token."*

   * This differs from my previous search result (which suggested prepending it); instead, I will store it in the `AgentResponse` metadata but strip it from the public message.

## **Implementation Plan**

### **1. Update** **`OpenRouterProvider`**

* Modify `redblackbench/providers/openrouter_provider.py`.

* Add `include_reasoning=True` to the `extra_body` in the API call.

* In `generate()`, return a structured object or a special format that separates `reasoning` from `content`.

  * *Correction:* Since `generate` currently returns `str`, I will prepend it with a special delimiter (e.g., `__THINKING_START__...__THINKING_END__`) so the `LLMAgent` can parse and separate it.

### **2. Update** **`LLMAgent`**

* Modify `redblackbench/agents/llm_agent.py`.

* Update parsing logic (`_parse_choice`, `_parse_reasoning`) to look for the thinking block.

* **Crucial Step:** When the agent speaks (in `get_initial_opinion`), ensure only the *final* `content` (without thinking) is returned as the `reasoning` (message) field of `AgentResponse`.

* Store the "thinking" part in a new field in `AgentResponse` (e.g., `private_thought`) or just in the `raw_response`.

### **3. Update** **`TrajectoryCollector`**

* Ensure the `raw_response` or the new `private_thought` field is saved to the trajectory file so we can analyze the hidden thinking later.

### **4. Refactor** **`cli.py`** **& Config**

* Ensure the `provider-check` and main `run` loop pass the `include_reasoning` flag.

* Update the configuration to use the requested "Thinking" models by default.

## **Testing**

* I will use the `provider-check` command to verify that reasoning is being received from OpenRouter for each model we choosed in that script  but separated correctly.

Shall I proceed with this plan?
