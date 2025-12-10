"""LLM-powered agent implementation for RedBlackBench."""

import re
from typing import List, Optional, TYPE_CHECKING

from redblackbench.agents.base import BaseAgent, AgentResponse
from redblackbench.agents.prompts import (
    build_system_prompt,
    build_initial_opinion_prompt,
    build_final_vote_prompt,
    PromptTemplate,
    DEFAULT_PROMPTS,
)
from redblackbench.game.scoring import Choice

if TYPE_CHECKING:
    from redblackbench.providers.base import BaseLLMProvider


class LLMAgent(BaseAgent):
    """An agent powered by a Large Language Model.
    
    Uses an LLM provider to generate responses during deliberation and voting.
    """
    
    def __init__(
        self,
        agent_id: str,
        team_name: str,
        provider: "BaseLLMProvider",
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """Initialize the LLM agent.
        
        Args:
            agent_id: Unique identifier for this agent
            team_name: Name of the team this agent belongs to
            provider: LLM provider for generating responses
            prompt_template: Optional custom prompt template
        """
        super().__init__(agent_id, team_name)
        self.provider = provider
        self.prompt_template = prompt_template or DEFAULT_PROMPTS
        self._system_prompt = build_system_prompt(
            agent_id, team_name, self.prompt_template
        )
    
    def _parse_choice(self, response: str) -> Choice:
        """Parse the choice from an LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed Choice (RED or BLACK)
            
        Raises:
            ValueError: If no valid choice found in response
        """
        response_upper = response.upper()
        
        # Look for explicit RECOMMENDATION: or VOTE: patterns
        recommendation_match = re.search(r'RECOMMENDATION:\s*(RED|BLACK)', response_upper)
        if recommendation_match:
            return Choice.RED if recommendation_match.group(1) == "RED" else Choice.BLACK
        
        vote_match = re.search(r'VOTE:\s*(RED|BLACK)', response_upper)
        if vote_match:
            return Choice.RED if vote_match.group(1) == "RED" else Choice.BLACK
        
        # Fallback: look for the last occurrence of RED or BLACK
        red_pos = response_upper.rfind("RED")
        black_pos = response_upper.rfind("BLACK")
        
        if red_pos == -1 and black_pos == -1:
            raise ValueError(f"Could not parse choice from response: {response[:200]}...")
        
        # Return whichever appears last (most likely the final decision)
        if red_pos > black_pos:
            return Choice.RED
        return Choice.BLACK
    
    def _parse_reasoning(self, response: str) -> str:
        """Parse the reasoning from an LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted reasoning text
        """
        # Look for REASONING: pattern
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=\n\n|\Z)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            return reasoning_match.group(1).strip()
        
        # Fallback: return everything after the choice
        for marker in ["RECOMMENDATION:", "VOTE:"]:
            if marker in response.upper():
                idx = response.upper().find(marker)
                remaining = response[idx:].split("\n", 1)
                if len(remaining) > 1:
                    return remaining[1].strip()
        
        return response.strip()
    
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
        user_prompt = build_initial_opinion_prompt(
            round_context, team_identifier, self.prompt_template
        )
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })
        
        # Get LLM response
        response_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=self.conversation_history,
        )
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant", 
            "content": response_text,
        })
        
        # Parse response
        try:
            choice = self._parse_choice(response_text)
        except ValueError:
            # Default to BLACK if parsing fails (cooperative default)
            choice = Choice.BLACK
        
        reasoning = self._parse_reasoning(response_text)
        
        return AgentResponse(
            choice=choice,
            reasoning=reasoning,
            raw_response=response_text,
        )
    
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
        user_prompt = build_final_vote_prompt(
            round_context, team_identifier, teammate_opinions, self.prompt_template
        )
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt,
        })
        
        # Get LLM response
        response_text = await self.provider.generate(
            system_prompt=self._system_prompt,
            messages=self.conversation_history,
        )
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })
        
        # Parse response
        try:
            choice = self._parse_choice(response_text)
        except ValueError:
            # Default to BLACK if parsing fails
            choice = Choice.BLACK
        
        reasoning = self._parse_reasoning(response_text)
        
        return AgentResponse(
            choice=choice,
            reasoning=reasoning,
            raw_response=response_text,
        )

