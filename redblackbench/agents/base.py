"""Base agent interface for RedBlackBench."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from redblackbench.game.scoring import Choice


@dataclass
class AgentResponse:
    """Response from an agent during deliberation or voting.
    
    Attributes:
        choice: The agent's choice (RED or BLACK)
        reasoning: The agent's reasoning for their choice
        confidence: Optional confidence score (0-1)
        raw_response: The raw LLM response text
    """
    choice: Choice
    reasoning: str
    confidence: Optional[float] = None
    raw_response: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "choice": str(self.choice),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }


class BaseAgent(ABC):
    """Abstract base class for game agents.
    
    Agents participate in team deliberations and vote on choices.
    """
    
    def __init__(self, agent_id: str, team_name: str):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            team_name: Name of the team this agent belongs to
        """
        self.agent_id = agent_id
        self.team_name = team_name
        self.conversation_history: List[dict] = []
    
    @abstractmethod
    async def get_initial_opinion(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> AgentResponse:
        """Get the agent's initial opinion before seeing teammates' views.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            
        Returns:
            Agent's initial response with choice and reasoning
        """
        pass
    
    @abstractmethod
    async def get_final_vote(
        self,
        round_context: dict,
        team_identifier: str,
        teammate_opinions: List[AgentResponse],
    ) -> AgentResponse:
        """Get the agent's final vote after seeing all teammates' opinions.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            teammate_opinions: List of all teammates' initial opinions
            
        Returns:
            Agent's final vote with choice and reasoning
        """
        pass
    
    def reset(self) -> None:
        """Reset the agent's conversation history for a new game."""
        self.conversation_history = []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, team={self.team_name})"

    @abstractmethod
    async def get_willingness_to_speak(
        self,
        round_context: dict,
        team_identifier: str,
        seen_messages: list,
    ) -> int:
        """Return willingness to speak (0-3) given current context.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B'
            seen_messages: Ordered list of prior team channel messages
        
        Returns:
            Integer in [0, 3] indicating willingness level
        """
        pass
