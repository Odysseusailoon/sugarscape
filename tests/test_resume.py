
import asyncio
import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator
from redblackbench.game.scoring import Choice
from redblackbench.trajectory.collector import TrajectoryCollector
from redblackbench.trajectory.trajectory import GameTrajectory
from redblackbench.teams.team import Team
from redblackbench.agents.base import BaseAgent, AgentResponse

class MockProvider:
    def __init__(self):
        self.provider_name = "mock"
        
    async def generate(self, system_prompt, messages):
        return "BLACK"

class MockAgent(BaseAgent):
    async def get_initial_opinion(self, context, team_identifier):
        return AgentResponse(choice=Choice.BLACK, reasoning="Always cooperate")
        
    async def get_final_vote(self, context, team_identifier, discussion):
        return AgentResponse(choice=Choice.BLACK, reasoning="Vote BLACK")

    async def get_willingness_to_speak(self, context, team_identifier, seen_messages):
        return 3 # Always willing to speak


async def test_resume_functionality():
    # Setup directories
    test_dir = Path("tests/temp_resume_test")
    test_dir.mkdir(exist_ok=True)
    traj_path = test_dir / "test_traj.json"
    
    # Clean up previous run
    if traj_path.exists():
        os.remove(traj_path)
        
    # Create configuration
    config = GameConfig(num_rounds=5, team_size=2, multipliers={5: 3})
    
    # Create teams
    team_a = Team(name="A", agents=[MockAgent(f"a{i}", "A") for i in range(2)])
    team_b = Team(name="B", agents=[MockAgent(f"b{i}", "B") for i in range(2)])
    
    # Run partial game (3 rounds)
    print("Running initial partial game...")
    collector = TrajectoryCollector(trajectory_id="test_traj")
    coordinator = GameCoordinator(team_a, team_b, config, trajectory_collector=collector)
    
    # Manually play 3 rounds
    coordinator._trajectory = collector.start_game(config, team_a, team_b)
    for _ in range(3):
        await coordinator.play_round()
    
    # Save incomplete trajectory
    print(f"Saving trajectory to {traj_path}")
    coordinator.get_trajectory().save(str(traj_path))
    
    # Verify file exists
    assert traj_path.exists()
    
    # Verify state before resume
    with open(traj_path) as f:
        data = json.load(f)
        assert len(data["timesteps"]) > 0
        last_round = data["timesteps"][-1]["round_num"]
        print(f"Last recorded round in file: {last_round}")
        assert last_round == 3
        
    # Resume game
    print("Resuming game...")
    
    # Create NEW coordinator and teams for resume
    team_a_new = Team(name="A", agents=[MockAgent(f"a{i}", "A") for i in range(2)])
    team_b_new = Team(name="B", agents=[MockAgent(f"b{i}", "B") for i in range(2)])
    collector_new = TrajectoryCollector(trajectory_id="test_traj") # ID matches
    
    coordinator_new = GameCoordinator(
        team_a=team_a_new, 
        team_b=team_b_new, 
        config=config, 
        trajectory_collector=collector_new
    )
    
    # Play game with resume
    final_state = await coordinator_new.play_game(resume_from=str(traj_path))
    
    # Verify results
    print(f"Final state rounds: {len(final_state.history)}")
    assert len(final_state.history) == 5
    assert final_state.is_complete
    
    # Verify trajectory has all rounds
    full_traj = coordinator_new.get_trajectory()
    outcomes = full_traj.get_outcomes()
    round_outcomes = [o for o in outcomes if o.outcome_type == "round"]
    print(f"Total round outcomes in trajectory: {len(round_outcomes)}")
    assert len(round_outcomes) == 5
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("Test passed!")

if __name__ == "__main__":
    asyncio.run(test_resume_functionality())
