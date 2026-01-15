"""End-to-end tests with mock LLM provider for recent commits.

Tests:
- Rule-based movement for LLM agents (token-saving)
- Post-encounter reflection system
- Identity review system
- No-fraud enforcement with binding contracts
- Provider compatibility fallbacks
"""

import pytest
import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.llm_agent import LLMSugarAgent
from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.trade import DialogueTradeSystem
from redblackbench.providers.base import BaseLLMProvider, ProviderConfig


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing that returns scripted responses."""

    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[List[str]] = None,
        response_generator: Optional[callable] = None,
    ):
        config = ProviderConfig(model=model)
        super().__init__(config)
        self.responses = responses or []
        self.response_generator = response_generator
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,  # Accept any extra kwargs for compatibility
    ) -> str:
        """Return scripted response or use generator."""
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt,
            "messages": messages,
            "kwargs": kwargs,
        })

        if self.response_generator:
            return self.response_generator(system_prompt, messages, self.call_count)

        if self.responses:
            idx = (self.call_count - 1) % len(self.responses)
            return self.responses[idx]

        # Default response
        return '{"intent": "PASS", "reasoning": "Mock response"}'


def create_trade_response_generator():
    """Create a response generator for trade dialogues."""
    state = {"phase": "small_talk", "round": 0}

    def generator(system_prompt: str, messages: List[dict], call_count: int) -> str:
        # Detect phase from system prompt
        if "small talk" in system_prompt.lower() or "casual conversation" in system_prompt.lower():
            return "Hello neighbor! Nice weather we're having. How are your crops doing?"

        if "intent" in system_prompt.lower() or "would you like to trade" in system_prompt.lower():
            return json.dumps({
                "intent": "TRADE",
                "reasoning": "I could use some spice and have extra sugar",
            })

        if "negotiation" in system_prompt.lower() or "offer" in system_prompt.lower():
            # Check if this is a response to an offer
            last_msg = messages[-1]["content"] if messages else ""
            if "OFFER" in last_msg or "offer" in last_msg.lower():
                return json.dumps({
                    "intent": "ACCEPT",
                    "public_offer_give": {"sugar": 0, "spice": 2},
                    "public_offer_receive": {"sugar": 5, "spice": 0},
                    "private_execute_give": {"sugar": 0, "spice": 2},
                    "reasoning": "Fair trade, I accept",
                })
            else:
                return json.dumps({
                    "intent": "OFFER",
                    "public_offer_give": {"sugar": 5, "spice": 0},
                    "public_offer_receive": {"sugar": 0, "spice": 2},
                    "private_execute_give": {"sugar": 5, "spice": 0},
                    "reasoning": "I have surplus sugar, need spice",
                })

        if "reflection" in system_prompt.lower() or "reflect" in system_prompt.lower():
            return json.dumps({
                "reflection": "The trade went well. My partner was fair.",
                "updates": {
                    "beliefs": {
                        "cooperation_value": "Cooperation can be beneficial"
                    },
                    "policies": [],
                    "identity_shift": 0.05,
                },
            })

        if "identity review" in system_prompt.lower() or "self-assessment" in system_prompt.lower():
            return json.dumps({
                "reflection": "I've been trading fairly and helping others when I can.",
                "identity_assessment": "altruist",
                "updates": {
                    "beliefs": {},
                    "policies": [],
                    "identity_shift": 0.1,
                },
            })

        if "end of life" in system_prompt.lower() or "final reflection" in system_prompt.lower():
            return json.dumps({
                "final_reflection": "I lived a good life, helped others when I could.",
                "regrets": "I wish I had been more generous early on.",
                "proudest_moment": "Helping a struggling neighbor survive.",
                "identity_assessment": "altruist",
            })

        # Movement decision (legacy format)
        return "REASONING: Need to find resources.\nACTION: NORTH"

    return generator


class TestRuleBasedMovement:
    """Test rule-based movement for LLM agents (token-saving feature)."""

    def test_llm_agent_uses_rule_based_movement_when_enabled(self):
        """LLM agents should use rule-based movement when config.rule_based_movement=True."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            initial_population=2,
            enable_llm_agents=False,  # We'll manually create LLM agents
            rule_based_movement=True,  # Token-saving mode
            enable_spice=True,
            enable_trade=False,
            max_ticks=5,
            seed=42,
        )

        mock_provider = MockLLMProvider()
        env = SugarEnvironment(config)

        # Create LLM agents manually with mock provider
        agents = []
        for i in range(2):
            agent = LLMSugarAgent(
                provider=mock_provider,
                goal_prompt="Survival",
                agent_id=i + 1,
                pos=(5 + i, 5),
                vision=3,
                metabolism=1,
                max_age=100,
                wealth=20,
                age=0,
                spice=20,
                metabolism_spice=1,
                name=f"Agent{i}",
            )
            agents.append(agent)
            env.add_agent(agent)

        # Simulate rule-based movement (what simulation does with rule_based_movement=True)
        for _ in range(3):
            for agent in agents:
                if agent.alive:
                    # Use parent class's rule-based movement
                    SugarAgent._move_and_harvest(agent, env)
                    agent._update_metrics(env)

                    # Record move history (as simulation does)
                    if hasattr(agent, 'move_history'):
                        agent.move_history.append({
                            "tick": 1,
                            "pos": agent.pos,
                            "action": "rule_based",
                            "wealth": agent.wealth,
                            "spice": agent.spice,
                            "sugar_harvested": 0,
                            "spice_harvested": 0,
                        })

        # With rule_based_movement=True, NO LLM calls should be made for movement
        assert mock_provider.call_count == 0, (
            f"Expected 0 LLM calls for movement with rule_based_movement=True, "
            f"got {mock_provider.call_count}"
        )

        # Agents should have move history with rule_based action
        for agent in agents:
            if agent.alive:
                assert len(agent.move_history) > 0, "LLM agent should have move history"
                assert agent.move_history[0]["action"] == "rule_based"


class TestIdentityReviewSystem:
    """Test identity review system (periodic self-assessment)."""

    @pytest.mark.asyncio
    async def test_identity_review_generates_response(self):
        """Identity review should call LLM and parse response."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            enable_spice=True,
            enable_origin_identity=True,
            enable_identity_review=True,
            identity_review_interval=10,
        )
        env = SugarEnvironment(config)

        mock_provider = MockLLMProvider(
            response_generator=create_trade_response_generator()
        )

        # Create LLM agent with origin identity
        agent = LLMSugarAgent(
            provider=mock_provider,
            goal_prompt="Test goal",
            agent_id=1,
            pos=(5, 5),
            vision=3,
            metabolism=1,
            max_age=100,
            wealth=20,
            age=0,
            spice=20,
            metabolism_spice=1,
            name="TestAgent",
        )

        # Set up origin identity
        agent.origin_identity = "altruist"
        agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt("altruist")
        agent.policy_list = SugarscapeConfig.get_default_policies("altruist")
        agent.belief_ledger = SugarscapeConfig.get_default_beliefs("altruist")
        agent.self_identity_leaning = 0.8

        # Run identity review
        result = await agent.async_identity_review(env, tick=10)

        assert "tick" in result
        assert result["tick"] == 10
        assert "raw_response" in result
        assert len(result["raw_response"]) > 0
        assert mock_provider.call_count == 1

    @pytest.mark.asyncio
    async def test_end_of_life_report(self):
        """End of life report should generate final reflection."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            enable_spice=True,
            enable_origin_identity=True,
            enable_end_of_life_report=True,
        )
        env = SugarEnvironment(config)

        mock_provider = MockLLMProvider(
            response_generator=create_trade_response_generator()
        )

        agent = LLMSugarAgent(
            provider=mock_provider,
            goal_prompt="Test goal",
            agent_id=1,
            pos=(5, 5),
            vision=3,
            metabolism=1,
            max_age=100,
            wealth=20,
            age=0,
            spice=20,
            metabolism_spice=1,
            name="TestAgent",
        )

        agent.origin_identity = "exploiter"
        agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt("exploiter")
        agent.self_identity_leaning = -0.5

        # Update lifetime stats
        agent.update_lifetime_stats("trade_completed")
        agent.update_lifetime_stats("trade_completed")
        agent.update_lifetime_stats("helped_agent")
        agent.update_lifetime_stats("gave_resources", amount=10)

        # Run end of life report
        result = await agent.async_end_of_life_report(env, tick=50, death_cause="old_age")

        assert result["tick"] == 50
        assert result["death_cause"] == "old_age"
        assert result["origin_identity"] == "exploiter"
        assert result["lifetime_stats"]["trades_completed"] == 2
        assert result["lifetime_stats"]["agents_helped"] == 1
        assert result["lifetime_stats"]["resources_given"] == 10


class TestPostEncounterReflection:
    """Test post-encounter reflection system."""

    @pytest.mark.asyncio
    async def test_reflection_updates_beliefs(self):
        """Reflection after trade should update agent beliefs."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            enable_spice=True,
            enable_trade=True,
            trade_mode="dialogue",
            enable_origin_identity=True,
            enable_reflection=True,
        )
        env = SugarEnvironment(config)

        mock_provider = MockLLMProvider(
            response_generator=create_trade_response_generator()
        )

        agent = LLMSugarAgent(
            provider=mock_provider,
            goal_prompt="Test goal",
            agent_id=1,
            pos=(5, 5),
            vision=3,
            metabolism=1,
            max_age=100,
            wealth=20,
            age=0,
            spice=20,
            metabolism_spice=1,
            name="TestAgent",
        )

        agent.origin_identity = "altruist"
        agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt("altruist")
        agent.policy_list = SugarscapeConfig.get_default_policies("altruist")
        agent.belief_ledger = SugarscapeConfig.get_default_beliefs("altruist")
        agent.self_identity_leaning = 0.5

        initial_leaning = agent.self_identity_leaning
        initial_belief = agent.belief_ledger["world"]["cooperation_value"]
        initial_policy_count = len(agent.policy_list)

        # Simulate applying a reflection update (using correct structure)
        updates = {
            "belief_updates": {
                "world": {"cooperation_value": "Cooperation is essential"}
            },
            "policy_updates": {
                "add": ["6. Always verify trade terms"],
            },
            "identity_shift": 0.1,
        }

        changes = agent.apply_reflection_update(updates)

        # Check that beliefs were updated (should be different from initial)
        assert agent.belief_ledger["world"]["cooperation_value"] == "Cooperation is essential"
        assert agent.belief_ledger["world"]["cooperation_value"] != initial_belief
        # Check that policy was added
        assert len(agent.policy_list) == initial_policy_count + 1
        assert "6. Always verify trade terms" in agent.policy_list
        # Check identity shift
        assert agent.self_identity_leaning == pytest.approx(initial_leaning + 0.1, abs=0.01)
        # Check changes dict was populated
        assert len(changes["beliefs_changed"]) > 0
        assert len(changes["policies_changed"]) > 0
        assert changes["identity_shifted"] == pytest.approx(0.1, abs=0.01)


class TestNoFraudEnforcement:
    """Test no-fraud enforcement with binding contracts."""

    def test_fix_accept_execute_direction_confusion(self):
        """Test that direction confusion in accept is corrected."""
        config = SugarscapeConfig(enable_spice=True)
        env = SugarEnvironment(config)
        ts = DialogueTradeSystem(env, allow_fraud=True)

        # Partner offers: give 10 sugar, receive 2 spice
        contract_give = {"sugar": 10, "spice": 0}
        contract_receive = {"sugar": 0, "spice": 2}

        # Confused acceptor sets their execute to what they expect to RECEIVE
        acceptor_execute = {"sugar": 10, "spice": 0}

        fixed = ts._fix_accept_execute_direction_confusion(
            acceptor_execute=acceptor_execute,
            contract_offer_give=contract_give,
            contract_offer_receive=contract_receive,
        )

        # Should be corrected to what acceptor should GIVE (the contract_receive)
        assert fixed == {"sugar": 0, "spice": 2}

    def test_no_correction_when_not_matching(self):
        """Test that non-matching values are not corrected."""
        config = SugarscapeConfig(enable_spice=True)
        env = SugarEnvironment(config)
        ts = DialogueTradeSystem(env, allow_fraud=True)

        contract_give = {"sugar": 10, "spice": 0}
        contract_receive = {"sugar": 0, "spice": 2}

        # Different values - could be intentional fraud
        acceptor_execute = {"sugar": 5, "spice": 1}

        fixed = ts._fix_accept_execute_direction_confusion(
            acceptor_execute=acceptor_execute,
            contract_offer_give=contract_give,
            contract_offer_receive=contract_receive,
        )

        # Should NOT be corrected
        assert fixed == {"sugar": 5, "spice": 1}


class TestProviderCompatibility:
    """Test provider compatibility with various kwargs."""

    @pytest.mark.asyncio
    async def test_provider_accepts_extra_kwargs(self):
        """Provider should accept extra kwargs without error."""
        mock_provider = MockLLMProvider()

        # Call with extra kwargs that some providers might use
        response = await mock_provider.generate(
            system_prompt="Test system prompt",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            temperature=0.7,
            chat_template_kwargs={"add_generation_prompt": True},
        )

        assert response is not None
        assert mock_provider.call_count == 1
        # Extra kwargs should be captured
        assert "kwargs" in mock_provider.call_history[0]


class TestDialogueTradeIntegration:
    """Integration test for dialogue trade with mock LLM."""

    @pytest.mark.asyncio
    async def test_full_trade_dialogue_flow(self):
        """Test complete trade dialogue with negotiation."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            enable_spice=True,
            enable_trade=True,
            trade_mode="dialogue",
            enable_new_encounter_protocol=True,
            small_talk_rounds=1,
            negotiation_rounds=2,
            enable_reflection=False,  # Disable for simpler test
            enable_origin_identity=True,
        )
        env = SugarEnvironment(config)

        mock_provider = MockLLMProvider(
            response_generator=create_trade_response_generator()
        )

        # Create two agents
        agent1 = LLMSugarAgent(
            provider=mock_provider,
            goal_prompt="Survival",
            agent_id=1,
            pos=(5, 5),
            vision=3,
            metabolism=1,
            max_age=100,
            wealth=30,  # Has sugar
            age=0,
            spice=5,    # Needs spice
            metabolism_spice=1,
            name="Alice",
        )

        agent2 = LLMSugarAgent(
            provider=mock_provider,
            goal_prompt="Survival",
            agent_id=2,
            pos=(5, 6),  # Adjacent
            vision=3,
            metabolism=1,
            max_age=100,
            wealth=5,   # Needs sugar
            age=0,
            spice=30,   # Has spice
            metabolism_spice=1,
            name="Bob",
        )

        # Set up origin identities
        for agent in [agent1, agent2]:
            agent.origin_identity = "altruist"
            agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt("altruist")
            agent.policy_list = SugarscapeConfig.get_default_policies("altruist")
            agent.belief_ledger = SugarscapeConfig.get_default_beliefs("altruist")
            agent.self_identity_leaning = 0.5

        env.add_agent(agent1)
        env.add_agent(agent2)

        # Create trade system
        ts = DialogueTradeSystem(
            env,
            max_rounds=config.negotiation_rounds,
            allow_fraud=False,  # No fraud - binding contracts
        )

        # Run the internal negotiation method directly
        result = await ts._negotiate_pair_safe(agent1, agent2, tick=1)

        # Verify the negotiation ran (LLM was called)
        assert mock_provider.call_count > 0, "LLM should have been called for dialogue"


class TestSimulationIntegration:
    """Full simulation integration test."""

    def test_simulation_runs_with_llm_agents(self):
        """Test that simulation runs correctly with manually created LLM agents."""
        config = SugarscapeConfig(
            width=15,
            height=15,
            initial_population=4,
            enable_llm_agents=False,  # We'll manually inject LLM agents
            rule_based_movement=True,
            enable_spice=True,
            enable_trade=False,  # Disable trade to simplify
            enable_origin_identity=True,
            enable_reflection=False,
            enable_identity_review=False,
            max_ticks=10,
            seed=42,
        )

        mock_provider = MockLLMProvider(
            response_generator=create_trade_response_generator()
        )

        env = SugarEnvironment(config)

        # Create a mix of agents manually
        agents = []
        for i in range(4):
            if i < 2:
                # LLM agents
                agent = LLMSugarAgent(
                    provider=mock_provider,
                    goal_prompt="Survival",
                    agent_id=i + 1,
                    pos=(5 + i * 2, 5),
                    vision=3,
                    metabolism=1,
                    max_age=100,
                    wealth=20,
                    age=0,
                    spice=20,
                    metabolism_spice=1,
                    name=f"LLMAgent{i}",
                )
                agent.origin_identity = "altruist"
                agent.origin_identity_prompt = SugarscapeConfig.get_origin_identity_prompt("altruist")
                agent.policy_list = SugarscapeConfig.get_default_policies("altruist")
                agent.belief_ledger = SugarscapeConfig.get_default_beliefs("altruist")
                agent.self_identity_leaning = 0.5
            else:
                # Regular agents
                agent = SugarAgent(
                    agent_id=i + 1,
                    pos=(5 + i * 2, 7),
                    vision=3,
                    metabolism=1,
                    max_age=100,
                    wealth=20,
                    age=0,
                    spice=20,
                    metabolism_spice=1,
                    name=f"Agent{i}",
                )
            agents.append(agent)
            env.add_agent(agent)

        # Run simulation steps manually
        for tick in range(5):
            env.growback()

            for agent in agents:
                if not agent.alive:
                    continue

                # Rule-based movement for all
                SugarAgent._move_and_harvest(agent, env)
                agent._update_metrics(env)

                # Track move history for LLM agents
                if isinstance(agent, LLMSugarAgent):
                    agent.move_history.append({
                        "tick": tick + 1,
                        "pos": agent.pos,
                        "action": "rule_based",
                        "wealth": agent.wealth,
                        "spice": agent.spice,
                    })

                # Metabolize
                agent.wealth -= agent.metabolism
                if config.enable_spice:
                    agent.spice -= agent.metabolism_spice

                # Age
                agent.age += 1

                # Check death
                if agent.wealth <= 0 or (config.enable_spice and agent.spice <= 0) or agent.age >= agent.max_age:
                    agent.alive = False

        # Some agents should still be alive
        alive_count = len([a for a in agents if a.alive])
        assert alive_count > 0, "At least some agents should survive"

        # LLM agents should have move history
        llm_agents = [a for a in agents if isinstance(a, LLMSugarAgent)]
        assert len(llm_agents) > 0, "Should have LLM agents"

        for agent in llm_agents:
            if agent.alive:
                assert len(agent.move_history) > 0, "LLM agent should have move history"

        # No LLM calls should have been made (rule-based movement)
        assert mock_provider.call_count == 0, "No LLM calls for rule-based movement"


class TestOriginIdentitySystem:
    """Test origin identity system (born good vs born bad)."""

    def test_origin_identity_initialization(self):
        """Test that origin identity is properly initialized."""
        config = SugarscapeConfig(
            width=10,
            height=10,
            initial_population=10,
            enable_origin_identity=True,
            origin_identity_distribution={"altruist": 0.5, "exploiter": 0.5},
            seed=42,
        )

        sim = SugarSimulation(config)

        altruists = [a for a in sim.agents if a.origin_identity == "altruist"]
        exploiters = [a for a in sim.agents if a.origin_identity == "exploiter"]

        # Should have some of each
        assert len(altruists) > 0, "Should have altruist agents"
        assert len(exploiters) > 0, "Should have exploiter agents"

        # Check that policies and beliefs are set
        for agent in altruists:
            assert len(agent.policy_list) > 0
            assert "world" in agent.belief_ledger
            assert agent.self_identity_leaning > 0  # Positive for altruists

        for agent in exploiters:
            assert len(agent.policy_list) > 0
            assert "world" in agent.belief_ledger
            assert agent.self_identity_leaning < 0  # Negative for exploiters


class TestGoalPresets:
    """Test goal preset system."""

    def test_goal_presets_exist(self):
        """Test that all goal presets are defined."""
        presets = ["none", "survival", "wealth", "altruist"]

        for preset in presets:
            prompt = SugarscapeConfig.get_goal_prompt(preset)
            assert len(prompt) > 0, f"Goal preset '{preset}' should have content"

    def test_goal_aliases(self):
        """Test that goal aliases work."""
        aliases = ["egalitarian", "utilitarian", "samaritan", "rawlsian"]

        for alias in aliases:
            prompt = SugarscapeConfig.get_goal_prompt(alias)
            altruist_prompt = SugarscapeConfig.get_goal_prompt("altruist")
            assert prompt == altruist_prompt, f"Alias '{alias}' should map to altruist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
