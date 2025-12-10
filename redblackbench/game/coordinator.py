"""Game coordinator for orchestrating Red-Black game sessions."""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TYPE_CHECKING, Callable, Any

from redblackbench.game.config import GameConfig, DEFAULT_CONFIG
from redblackbench.game.scoring import Choice, ScoringMatrix, RoundResult

if TYPE_CHECKING:
    from redblackbench.teams.team import Team
    from redblackbench.logging.game_logger import GameLogger
    from redblackbench.trajectory.collector import TrajectoryCollector
    from redblackbench.trajectory.trajectory import GameTrajectory


@dataclass
class GameState:
    """Current state of the game.
    
    Attributes:
        config: Game configuration
        current_round: Current round number (1-indexed)
        team_a_total: Team A's cumulative score
        team_b_total: Team B's cumulative score
        history: List of all round results
        is_complete: Whether the game has finished
    """
    config: GameConfig
    current_round: int = 1
    team_a_total: int = 0
    team_b_total: int = 0
    history: List[RoundResult] = field(default_factory=list)
    is_complete: bool = False
    
    @property
    def total_score(self) -> int:
        """Combined score for both teams."""
        return self.team_a_total + self.team_b_total
    
    @property
    def cooperation_rate(self) -> float:
        """Percentage of choices that were BLACK (cooperative)."""
        if not self.history:
            return 0.0
        total_choices = len(self.history) * 2
        black_choices = sum(
            (1 if r.team_a_choice == Choice.BLACK else 0) +
            (1 if r.team_b_choice == Choice.BLACK else 0)
            for r in self.history
        )
        return black_choices / total_choices
    
    @property
    def max_possible_score(self) -> int:
        """Maximum possible combined score."""
        return self.config.calculate_max_possible_score()
    
    @property 
    def efficiency(self) -> float:
        """Score achieved as percentage of maximum possible."""
        max_score = self.max_possible_score
        if max_score == 0:
            return 0.0
        # Normalize to 0-1 range (min possible is negative)
        min_possible = -max_score  # If both always defect
        return (self.total_score - min_possible) / (max_score - min_possible)
    
    def get_round_context(self) -> dict:
        """Get context information for the current round.
        
        Returns:
            Dictionary with round context for agents
        """
        return {
            "current_round": self.current_round,
            "total_rounds": self.config.num_rounds,
            "multiplier": self.config.get_multiplier(self.current_round),
            "team_a_score": self.team_a_total,
            "team_b_score": self.team_b_total,
            "total_score": self.total_score,
            "max_possible": self.max_possible_score,
            "history": [
                {
                    "round": r.round_num,
                    "team_a_choice": str(r.team_a_choice),
                    "team_b_choice": str(r.team_b_choice),
                    "team_a_score": r.team_a_score,
                    "team_b_score": r.team_b_score,
                    "multiplier": r.multiplier,
                }
                for r in self.history
            ],
        }


class TeamProtocol(Protocol):
    """Protocol defining the interface teams must implement."""
    
    @property
    def name(self) -> str:
        """Team identifier."""
        ...
    
    async def make_choice(self, game_state: GameState, team_identifier: str) -> Choice:
        """Make a choice for this round after deliberation.
        
        Args:
            game_state: Current game state
            team_identifier: 'A' or 'B' indicating which team this is
            
        Returns:
            The team's choice (RED or BLACK)
        """
        ...


class GameCoordinator:
    """Orchestrates the Red-Black game between two teams.
    
    Manages the game loop, score tracking, and round progression.
    Optionally collects full trajectory data for analysis.
    """
    
    def __init__(
        self,
        team_a: TeamProtocol,
        team_b: TeamProtocol,
        config: Optional[GameConfig] = None,
        logger: Optional["GameLogger"] = None,
        trajectory_collector: Optional["TrajectoryCollector"] = None,
    ):
        """Initialize the game coordinator.
        
        Args:
            team_a: First team
            team_b: Second team
            config: Game configuration (uses default if not provided)
            logger: Optional game logger for recording events
            trajectory_collector: Optional trajectory collector for full state capture
        """
        self.team_a = team_a
        self.team_b = team_b
        self.config = config or DEFAULT_CONFIG
        self.logger = logger
        self.trajectory_collector = trajectory_collector
        self.scoring = ScoringMatrix(self.config)
        self.state = GameState(config=self.config)
        self._trajectory: Optional["GameTrajectory"] = None
    
    async def play_round(self) -> RoundResult:
        """Play a single round of the game.
        
        Returns:
            The result of the round
            
        Raises:
            RuntimeError: If game is already complete
        """
        if self.state.is_complete:
            raise RuntimeError("Game is already complete")
        
        round_num = self.state.current_round
        multiplier = self.config.get_multiplier(round_num)
        
        if self.logger:
            await self.logger.log_round_start(round_num, multiplier)
        
        if self.trajectory_collector:
            self.trajectory_collector.record_round_start(round_num, multiplier)
        
        # Both teams make their choices concurrently
        # For trajectory collection, we need to capture deliberation details
        team_a_choice, team_b_choice = await asyncio.gather(
            self._team_make_choice(self.team_a, "A"),
            self._team_make_choice(self.team_b, "B"),
        )
        
        # Calculate scores
        result = self.scoring.create_round_result(
            round_num=round_num,
            team_a_choice=team_a_choice,
            team_b_choice=team_b_choice,
        )
        
        # Update state
        self.state.team_a_total += result.team_a_score
        self.state.team_b_total += result.team_b_score
        self.state.history.append(result)
        
        if self.logger:
            await self.logger.log_round_result(result)
        
        if self.trajectory_collector:
            self.trajectory_collector.record_round_end(result, self.state)
        
        # Advance to next round or end game
        if self.state.current_round >= self.config.num_rounds:
            self.state.is_complete = True
            if self.logger:
                await self.logger.log_game_end(self.state)
        else:
            self.state.current_round += 1
        
        return result
    
    async def _team_make_choice(self, team: TeamProtocol, team_identifier: str) -> Choice:
        """Have a team make their choice, with trajectory recording.
        
        Args:
            team: The team making a choice
            team_identifier: 'A' or 'B'
            
        Returns:
            The team's choice
        """
        # Check if this is a Team with deliberation (not just the Protocol)
        if hasattr(team, 'deliberation') and self.trajectory_collector:
            # Use detailed deliberation with trajectory capture
            return await self._deliberate_with_trajectory(team, team_identifier)
        else:
            # Standard choice
            return await team.make_choice(self.state, team_identifier)
    
    async def _deliberate_with_trajectory(self, team: "Team", team_identifier: str) -> Choice:
        """Run team deliberation with trajectory capture.
        
        Args:
            team: The team
            team_identifier: 'A' or 'B'
            
        Returns:
            The team's choice
        """
        round_context = self.state.get_round_context()
        round_num = self.state.current_round
        
        # Record deliberation start
        self.trajectory_collector.record_deliberation_start(round_num, team, team_identifier)
        
        initial_pairs = await team.deliberation._gather_initial_opinions(round_context, team_identifier)
        self.trajectory_collector.record_initial_opinions(round_num, team, team_identifier, initial_pairs, self.state)
        
        # Phase 2: Final votes
        initial_opinions = [resp for _, resp in initial_pairs]
        final_votes = await team.deliberation._gather_final_votes(round_context, team_identifier, initial_opinions)
        
        # Determine result
        final_choice, vote_counts, was_unanimous = team.deliberation._determine_majority(final_votes)
        
        # Create deliberation result and store in team history
        from redblackbench.teams.deliberation import DeliberationResult
        result = DeliberationResult(
            final_choice=final_choice,
            initial_opinions=initial_opinions,
            final_votes=final_votes,
            vote_counts=vote_counts,
            was_unanimous=was_unanimous,
        )
        team.deliberation_history.append(result)
        
        # Record final votes
        self.trajectory_collector.record_final_votes(
            round_num, team, team_identifier, result, self.state
        )
        
        return final_choice
    
    async def play_game(self) -> GameState:
        """Play a complete game from start to finish.
        
        Returns:
            Final game state with all results
        """
        if self.logger:
            await self.logger.log_game_start(self.config, self.team_a.name, self.team_b.name)
        
        if self.trajectory_collector:
            self._trajectory = self.trajectory_collector.start_game(
                self.config, self.team_a, self.team_b
            )
        
        import random
        if self.config.seed is not None:
            random.seed(self.config.seed)
        while not self.state.is_complete:
            await self.play_round()
        
        if self.trajectory_collector:
            self._trajectory = self.trajectory_collector.end_game(
                self.state, self.team_a, self.team_b
            )
        
        return self.state
    
    def get_trajectory(self) -> Optional["GameTrajectory"]:
        """Get the game trajectory if trajectory collection was enabled.
        
        Returns:
            The GameTrajectory, or None if not collected
        """
        return self._trajectory
    
    def reset(self) -> None:
        """Reset the game to initial state for a new game."""
        self.state = GameState(config=self.config)
        self._trajectory = None
    
    def get_summary(self) -> dict:
        """Get a summary of the current/final game state.
        
        Returns:
            Dictionary with game summary statistics
        """
        summary = {
            "is_complete": self.state.is_complete,
            "rounds_played": len(self.state.history),
            "total_rounds": self.config.num_rounds,
            "team_a": {
                "name": self.team_a.name,
                "score": self.state.team_a_total,
            },
            "team_b": {
                "name": self.team_b.name,
                "score": self.state.team_b_total,
            },
            "total_score": self.state.total_score,
            "max_possible_score": self.state.max_possible_score,
            "efficiency": self.state.efficiency,
            "cooperation_rate": self.state.cooperation_rate,
        }
        
        # Add trajectory summary if available
        if self._trajectory:
            summary["trajectory_summary"] = self._trajectory.get_summary()
        
        return summary
