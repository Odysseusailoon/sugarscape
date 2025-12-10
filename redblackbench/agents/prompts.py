"""Prompt templates for LLM agents in the Red-Black game."""

from dataclasses import dataclass
from typing import List, Optional

from redblackbench.agents.base import AgentResponse


@dataclass
class PromptTemplate:
    """Templates for agent prompts.
    
    Attributes:
        system_prompt: The system message explaining the game
        initial_opinion_prompt: Template for getting initial opinion
        final_vote_prompt: Template for getting final vote after deliberation
    """
    system_prompt: str
    initial_opinion_prompt: str
    final_vote_prompt: str


def format_history(history: List[dict]) -> str:
    """Format game history for inclusion in prompts.
    
    Args:
        history: List of round history dictionaries
        
    Returns:
        Formatted string representation of history
    """
    if not history:
        return "No rounds played yet."
    
    lines = []
    for h in history:
        lines.append(
            f"Round {h['round']} (x{h['multiplier']}): "
            f"Team A chose {h['team_a_choice']} ({h['team_a_score']:+d}), "
            f"Team B chose {h['team_b_choice']} ({h['team_b_score']:+d})"
        )
    return "\n".join(lines)


def format_teammate_opinions(opinions: List[AgentResponse]) -> str:
    """Format teammate opinions for the final vote prompt.
    
    Args:
        opinions: List of teammate AgentResponse objects
        
    Returns:
        Formatted string of all opinions
    """
    if not opinions:
        return "No teammate opinions available."
    
    lines = []
    for i, op in enumerate(opinions, 1):
        lines.append(f"Teammate {i}: Recommends {op.choice.value}")
        lines.append(f"  Reasoning: {op.reasoning}")
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are playing a game called the Red-Black Game.

## GAME OBJECTIVE
Your objective is to achieve the MAXIMUM TOTAL POINTS possible.

## SCORING RULES
Each round, both teams simultaneously choose either RED or BLACK:
- Both teams choose BLACK: Each team gets +3 points (BEST collective outcome: +6 total)
- Both teams choose RED: Each team gets -3 points (WORST collective outcome: -6 total)
- One team chooses RED, other chooses BLACK: RED gets +6, BLACK gets -6 (Zero-sum: 0 total)

## BONUS MULTIPLIERS
Some rounds have score multipliers:
- Round 5: 3x multiplier
- Round 8: 5x multiplier  
- Round 10: 10x multiplier

## YOUR ROLE
You are Agent {agent_id} on Team {team_name}. You will:
1. Share your initial opinion and reasoning with your teammates
2. Consider your teammates' opinions
3. Cast a final vote (RED or BLACK)

Your team's choice will be determined by majority vote among all team members.

Remember: The objective is MAXIMUM TOTAL POINTS."""

INITIAL_OPINION_PROMPT = """## CURRENT GAME STATE
- Round: {current_round} of {total_rounds}
- This round's multiplier: {multiplier}x
- Your team (Team {team_id}) score: {own_score}
- Other team score: {opponent_score}
- Combined total: {total_score}
- Maximum possible total: {max_possible}

## ROUND HISTORY
{history}

## YOUR TASK
Share your INITIAL OPINION on what choice our team should make this round.

Think carefully about:
1. The game's objective (maximize TOTAL points, not just our team's)
2. The scoring matrix and what outcomes are possible
3. Any patterns in the history
4. The multiplier for this round

Respond with your recommendation and detailed reasoning. Format your response as:

RECOMMENDATION: [RED or BLACK]
REASONING: [Your detailed reasoning for this choice]"""


FINAL_VOTE_PROMPT = """## TEAMMATE OPINIONS
Your teammates have shared their initial opinions:

{teammate_opinions}

## CURRENT GAME STATE
- Round: {current_round} of {total_rounds}
- This round's multiplier: {multiplier}x
- Your team (Team {team_id}) score: {own_score}
- Other team score: {opponent_score}
- Combined total: {total_score}

## ROUND HISTORY
{history}

## YOUR FINAL VOTE
After considering your teammates' opinions, cast your FINAL VOTE.

Remember: The team's choice is determined by majority vote. Your vote matters!

Respond with your final decision. Format your response as:

VOTE: [RED or BLACK]
REASONING: [Brief explanation for your final decision]"""


DEFAULT_PROMPTS = PromptTemplate(
    system_prompt=SYSTEM_PROMPT,
    initial_opinion_prompt=INITIAL_OPINION_PROMPT,
    final_vote_prompt=FINAL_VOTE_PROMPT,
)


def build_system_prompt(agent_id: str, team_name: str, template: Optional[PromptTemplate] = None) -> str:
    """Build the system prompt for an agent.
    
    Args:
        agent_id: The agent's identifier
        team_name: The team name
        template: Optional custom prompt template
        
    Returns:
        Formatted system prompt
    """
    prompts = template or DEFAULT_PROMPTS
    return prompts.system_prompt.format(agent_id=agent_id, team_name=team_name)


def build_initial_opinion_prompt(
    round_context: dict,
    team_identifier: str,
    template: Optional[PromptTemplate] = None,
) -> str:
    """Build the initial opinion prompt.
    
    Args:
        round_context: Current game state context
        team_identifier: 'A' or 'B'
        template: Optional custom prompt template
        
    Returns:
        Formatted initial opinion prompt
    """
    prompts = template or DEFAULT_PROMPTS
    
    own_score = round_context["team_a_score"] if team_identifier == "A" else round_context["team_b_score"]
    opponent_score = round_context["team_b_score"] if team_identifier == "A" else round_context["team_a_score"]
    
    return prompts.initial_opinion_prompt.format(
        current_round=round_context["current_round"],
        total_rounds=round_context["total_rounds"],
        multiplier=round_context["multiplier"],
        team_id=team_identifier,
        own_score=own_score,
        opponent_score=opponent_score,
        total_score=round_context["total_score"],
        max_possible=round_context["max_possible"],
        history=format_history(round_context["history"]),
    )


def build_final_vote_prompt(
    round_context: dict,
    team_identifier: str,
    teammate_opinions: List[AgentResponse],
    template: Optional[PromptTemplate] = None,
) -> str:
    """Build the final vote prompt.
    
    Args:
        round_context: Current game state context
        team_identifier: 'A' or 'B'
        teammate_opinions: List of teammate opinions
        template: Optional custom prompt template
        
    Returns:
        Formatted final vote prompt
    """
    prompts = template or DEFAULT_PROMPTS
    
    own_score = round_context["team_a_score"] if team_identifier == "A" else round_context["team_b_score"]
    opponent_score = round_context["team_b_score"] if team_identifier == "A" else round_context["team_a_score"]
    
    return prompts.final_vote_prompt.format(
        current_round=round_context["current_round"],
        total_rounds=round_context["total_rounds"],
        multiplier=round_context["multiplier"],
        team_id=team_identifier,
        own_score=own_score,
        opponent_score=opponent_score,
        total_score=round_context["total_score"],
        teammate_opinions=format_teammate_opinions(teammate_opinions),
        history=format_history(round_context["history"]),
    )

