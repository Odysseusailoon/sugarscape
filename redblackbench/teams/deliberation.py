"""Deliberation mechanism for team decision-making."""

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING

from redblackbench.agents.base import AgentResponse
from redblackbench.game.scoring import Choice

if TYPE_CHECKING:
    from redblackbench.agents.base import BaseAgent


@dataclass
class DeliberationResult:
    """Result of a team deliberation process.
    
    Attributes:
        final_choice: The team's final choice determined by majority vote
        initial_opinions: All agents' initial opinions
        final_votes: All agents' final votes after seeing opinions
        vote_counts: Dictionary of choice -> vote count
        was_unanimous: Whether all agents voted the same way
    """
    final_choice: Choice
    initial_opinions: List[AgentResponse] = field(default_factory=list)
    final_votes: List[AgentResponse] = field(default_factory=list)
    vote_counts: dict = field(default_factory=dict)
    was_unanimous: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "final_choice": str(self.final_choice),
            "initial_opinions": [op.to_dict() for op in self.initial_opinions],
            "final_votes": [v.to_dict() for v in self.final_votes],
            "vote_counts": {str(k): v for k, v in self.vote_counts.items()},
            "was_unanimous": self.was_unanimous,
        }


class Deliberation:
    """Manages the deliberation process for a team of agents.
    
    The deliberation follows a two-phase process:
    1. Initial Opinion Phase: Each agent shares their opinion independently
    2. Final Vote Phase: After seeing all opinions, each agent casts their final vote
    
    The team's choice is determined by majority vote in the final phase.
    """
    
    def __init__(self, agents: List["BaseAgent"]):
        """Initialize deliberation with a list of agents.
        
        Args:
            agents: List of agents participating in deliberation
        """
        self.agents = agents
    
    async def _gather_initial_opinions(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> List[AgentResponse]:
        """Gather initial opinions from all agents concurrently.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            
        Returns:
            List of initial opinions from all agents
        """
        tasks = [
            agent.get_initial_opinion(round_context, team_identifier)
            for agent in self.agents
        ]
        return await asyncio.gather(*tasks)
    
    async def _gather_final_votes(
        self,
        round_context: dict,
        team_identifier: str,
        all_opinions: List[AgentResponse],
    ) -> List[AgentResponse]:
        """Gather final votes from all agents after sharing opinions.
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            all_opinions: All initial opinions to share with agents
            
        Returns:
            List of final votes from all agents
        """
        tasks = [
            agent.get_final_vote(round_context, team_identifier, all_opinions)
            for agent in self.agents
        ]
        return await asyncio.gather(*tasks)
    
    def _determine_majority(self, votes: List[AgentResponse]) -> tuple[Choice, dict, bool]:
        """Determine the majority choice from votes.
        
        Args:
            votes: List of agent votes
            
        Returns:
            Tuple of (winning_choice, vote_counts, was_unanimous)
        """
        choices = [v.choice for v in votes]
        vote_counts = Counter(choices)
        
        # Get the choice with the most votes
        most_common = vote_counts.most_common(1)[0]
        winning_choice = most_common[0]
        
        # Check if unanimous
        was_unanimous = len(vote_counts) == 1
        
        # Convert Counter to regular dict for serialization
        counts_dict = {choice: count for choice, count in vote_counts.items()}
        
        return winning_choice, counts_dict, was_unanimous
    
    async def deliberate(
        self,
        round_context: dict,
        team_identifier: str,
    ) -> DeliberationResult:
        """Run the full deliberation process.
        
        Process:
        1. All agents share their initial opinions concurrently
        2. All opinions are shared with all agents
        3. All agents cast their final votes concurrently
        4. Majority vote determines team choice
        
        Args:
            round_context: Current game state context
            team_identifier: 'A' or 'B' indicating which team
            
        Returns:
            DeliberationResult with the team's final choice and all votes
        """
        # Phase 1: Gather initial opinions
        initial_opinions = await self._gather_initial_opinions(
            round_context, team_identifier
        )
        
        # Phase 2: Share opinions and gather final votes
        final_votes = await self._gather_final_votes(
            round_context, team_identifier, initial_opinions
        )
        
        # Determine majority
        final_choice, vote_counts, was_unanimous = self._determine_majority(final_votes)
        
        return DeliberationResult(
            final_choice=final_choice,
            initial_opinions=initial_opinions,
            final_votes=final_votes,
            vote_counts=vote_counts,
            was_unanimous=was_unanimous,
        )

