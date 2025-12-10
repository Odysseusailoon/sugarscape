"""Tests for agents and deliberation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List

from redblackbench.agents.base import BaseAgent, AgentResponse
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.game.scoring import Choice
from redblackbench.teams.deliberation import Deliberation, DeliberationResult
from redblackbench.teams.team import Team


class MockProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: List[str]):
        """Initialize with predetermined responses."""
        self.responses = responses
        self.response_index = 0
        self.config = MagicMock()
        self.config.model = "mock-model"
    
    async def generate(self, system_prompt: str, messages: List[dict]) -> str:
        """Return the next predetermined response."""
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return response


class TestAgentResponse:
    """Tests for AgentResponse."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = AgentResponse(
            choice=Choice.BLACK,
            reasoning="Cooperation is optimal",
            confidence=0.9,
            raw_response="RECOMMENDATION: BLACK\nREASONING: Cooperation is optimal",
        )
        
        d = response.to_dict()
        
        assert d["choice"] == "BLACK"
        assert d["reasoning"] == "Cooperation is optimal"
        assert d["confidence"] == 0.9


class TestLLMAgent:
    """Tests for LLMAgent."""
    
    @pytest.mark.asyncio
    async def test_parse_choice_recommendation(self):
        """Test parsing choice from RECOMMENDATION format."""
        provider = MockProvider([
            "RECOMMENDATION: BLACK\nREASONING: Trust builds cooperation"
        ])
        agent = LLMAgent("agent_1", "Team A", provider)
        
        response = await agent.get_initial_opinion(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        
        assert response.choice == Choice.BLACK
    
    @pytest.mark.asyncio
    async def test_parse_choice_vote(self):
        """Test parsing choice from VOTE format."""
        provider = MockProvider([
            "Initial opinion here",
            "VOTE: RED\nREASONING: Protecting against defection"
        ])
        agent = LLMAgent("agent_1", "Team A", provider)
        
        # Get initial opinion first
        await agent.get_initial_opinion(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        
        # Now get final vote
        response = await agent.get_final_vote(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A",
            []
        )
        
        assert response.choice == Choice.RED
    
    @pytest.mark.asyncio
    async def test_default_to_black_on_parse_failure(self):
        """Test that parsing failure defaults to BLACK."""
        provider = MockProvider([
            "I'm not sure what to do here."
        ])
        agent = LLMAgent("agent_1", "Team A", provider)
        
        response = await agent.get_initial_opinion(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        
        # Should default to BLACK (cooperative)
        assert response.choice == Choice.BLACK

    @pytest.mark.asyncio
    async def test_willingness_parsing(self):
        provider = MockProvider([
            "WILLINGNESS: 3",
        ])
        agent = LLMAgent("agent_1", "Team A", provider)
        w = await agent.get_willingness_to_speak(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A",
            []
        )
        assert w == 3
    
    def test_reset(self):
        """Test agent reset clears history."""
        provider = MockProvider([])
        agent = LLMAgent("agent_1", "Team A", provider)
        
        agent.conversation_history = [{"role": "user", "content": "test"}]
        agent.reset()
        
        assert agent.conversation_history == []


class TestDeliberation:
    """Tests for the Deliberation class."""
    
    @pytest.mark.asyncio
    async def test_majority_vote_black_wins(self):
        """Test that majority BLACK results in BLACK choice."""
        # Create agents that will vote BLACK
        responses = ["RECOMMENDATION: BLACK\nREASONING: Cooperate",
                     "VOTE: BLACK\nREASONING: Cooperate"]
        agents = [
            LLMAgent(f"agent_{i}", "Team A", MockProvider(responses))
            for i in range(3)
        ]
        
        deliberation = Deliberation(agents)
        result = await deliberation.deliberate(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        
        assert result.final_choice == Choice.BLACK
        assert len(result.initial_opinions) == 3
        assert len(result.final_votes) == 3
    
    @pytest.mark.asyncio
    async def test_unanimous_detection(self):
        """Test detection of unanimous votes."""
        responses = ["RECOMMENDATION: BLACK\nREASONING: Cooperate",
                     "VOTE: BLACK\nREASONING: Cooperate"]
        agents = [
            LLMAgent(f"agent_{i}", "Team A", MockProvider(responses))
            for i in range(3)
        ]
        
        deliberation = Deliberation(agents)
        result = await deliberation.deliberate(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        
        assert result.was_unanimous is True

    @pytest.mark.asyncio
    async def test_willingness_ordering(self):
        class WillingAgent(BaseAgent):
            def __init__(self, agent_id, team_name, will):
                super().__init__(agent_id, team_name)
                self._will = will
            async def get_initial_opinion(self, round_context, team_identifier):
                return AgentResponse(Choice.BLACK, f"Agent {self.agent_id} speaks", raw_response="RECOMMENDATION: BLACK")
            async def get_final_vote(self, round_context, team_identifier, teammate_opinions):
                return AgentResponse(Choice.BLACK, "Final", raw_response="VOTE: BLACK")
            async def get_willingness_to_speak(self, round_context, team_identifier, seen_messages):
                return self._will

        agents = [
            WillingAgent("a1", "Team A", 1),
            WillingAgent("a2", "Team A", 3),
            WillingAgent("a3", "Team A", 2),
        ]
        deliberation = Deliberation(agents)
        res = await deliberation._gather_initial_opinions(
            {"current_round": 1, "total_rounds": 10, "multiplier": 1,
             "team_a_score": 0, "team_b_score": 0, "total_score": 0,
             "max_possible": 150, "history": []},
            "A"
        )
        order_ids = [a.agent_id for a, _ in res]
        assert order_ids[0] == "a2"


class TestTeam:
    """Tests for the Team class."""
    
    @pytest.mark.asyncio
    async def test_team_make_choice(self):
        """Test team making a choice through deliberation."""
        responses = ["RECOMMENDATION: BLACK\nREASONING: Cooperate",
                     "VOTE: BLACK\nREASONING: Cooperate"]
        agents = [
            LLMAgent(f"agent_{i}", "Team A", MockProvider(responses))
            for i in range(3)
        ]
        
        team = Team("Team A", agents)
        
        # Create a mock game state
        class MockGameState:
            def get_round_context(self):
                return {
                    "current_round": 1, "total_rounds": 10, "multiplier": 1,
                    "team_a_score": 0, "team_b_score": 0, "total_score": 0,
                    "max_possible": 150, "history": []
                }
        
        choice = await team.make_choice(MockGameState(), "A")
        
        assert choice == Choice.BLACK
        assert len(team.deliberation_history) == 1
    
    def test_team_properties(self):
        """Test team property accessors."""
        agents = [MagicMock() for _ in range(5)]
        team = Team("Test Team", agents)
        
        assert team.name == "Test Team"
        assert team.size == 5
    
    def test_reset(self):
        """Test team reset."""
        agents = [MagicMock() for _ in range(3)]
        for agent in agents:
            agent.reset = MagicMock()
        
        team = Team("Test Team", agents)
        team.deliberation_history = [MagicMock()]
        
        team.reset()
        
        assert team.deliberation_history == []
        for agent in agents:
            agent.reset.assert_called_once()
