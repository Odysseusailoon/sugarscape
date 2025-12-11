"""Command-line interface for RedBlackBench."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
import yaml

from redblackbench.game.config import GameConfig
from redblackbench.game.coordinator import GameCoordinator
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.teams.team import Team
from redblackbench.logging.game_logger import GameLogger
from redblackbench.logging.metrics import MetricsCollector
from redblackbench.trajectory.collector import TrajectoryCollector


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_provider(provider_config: dict):
    """Create an LLM provider from configuration.
    
    Args:
        provider_config: Provider configuration dictionary
        
    Returns:
        Configured LLM provider
    """
    provider_type = provider_config.get("type", "openai")
    model = provider_config.get("model")
    temperature = provider_config.get("temperature", 0.7)
    api_key = provider_config.get("api_key")
    # Enable reasoning capture for OpenRouter by default
    include_reasoning = provider_config.get("include_reasoning", True)
    
    if provider_type == "openai":
        from redblackbench.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=model or "gpt-4",
            temperature=temperature,
            api_key=api_key,
        )
    elif provider_type == "anthropic":
        from redblackbench.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            model=model or "claude-3-opus-20240229",
            temperature=temperature,
            api_key=api_key,
        )
    elif provider_type == "openrouter":
        from redblackbench.providers.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(
            model=model,
            temperature=temperature,
            api_key=api_key,
            include_reasoning=include_reasoning,
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def create_team(
    team_config: dict,
    team_name: str,
    default_provider_config: dict,
) -> Team:
    """Create a team from configuration.
    
    Args:
        team_config: Team configuration dictionary
        team_name: Name for the team
        default_provider_config: Default provider config if not specified
        
    Returns:
        Configured Team
    """
    team_size = team_config.get("size", 5)
    provider_config = team_config.get("provider", default_provider_config)
    
    agents = []
    for i in range(team_size):
        agent_id = f"{team_name}_agent_{i+1}"
        provider = create_provider(provider_config)
        agent = LLMAgent(
            agent_id=agent_id,
            team_name=team_name,
            provider=provider,
        )
        agents.append(agent)
    
    return Team(name=team_name, agents=agents)


async def run_experiment(config: dict, save_trajectory: bool = True, resume: bool = False) -> None:
    """Run an experiment based on configuration.
    
    Args:
        config: Experiment configuration dictionary
        save_trajectory: Whether to save full trajectory data
        resume: Whether to resume from existing trajectory if found
    """
    # Extract configurations
    game_config_dict = config.get("game", {})
    team_a_config = config.get("team_a", {})
    team_b_config = config.get("team_b", {})
    default_provider = config.get("default_provider", {"type": "openai", "model": "gpt-4"})
    output_dir = config.get("output_dir", "results")
    experiment_name = config.get("experiment_name", "experiment")
    num_games = config.get("num_games", 1)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_path / "trajectories"
    if save_trajectory:
        trajectories_path.mkdir(parents=True, exist_ok=True)
    
    # Create game configuration
    game_config = GameConfig(
        num_rounds=game_config_dict.get("num_rounds", 10),
        team_size=game_config_dict.get("team_size", 5),
        multipliers=game_config_dict.get("multipliers", {5: 3, 8: 5, 10: 10}),
    )
    
    # Create logger
    logger = GameLogger(output_dir=output_dir, experiment_name=experiment_name)
    
    # Run games
    for game_num in range(1, num_games + 1):
        print(f"\n{'='*60}")
        print(f"Starting Game {game_num} of {num_games}")
        print(f"{'='*60}")
        
        # Create fresh teams for each game
        team_a = create_team(team_a_config, "Team_A", default_provider)
        team_b = create_team(team_b_config, "Team_B", default_provider)
        
        # Create trajectory collector if enabled
        trajectory_collector = None
        resume_from_path = None
        
        if save_trajectory:
            trajectory_id = f"{experiment_name}_game_{game_num}"
            trajectory_collector = TrajectoryCollector(trajectory_id=trajectory_id)
            
            # Check for existing trajectory to resume
            if resume:
                traj_file = trajectories_path / f"{trajectory_id}.json"
                if traj_file.exists():
                    print(f"Found existing trajectory: {traj_file}")
                    resume_from_path = str(traj_file)
        
        # Create coordinator
        coordinator = GameCoordinator(
            team_a=team_a,
            team_b=team_b,
            config=game_config,
            logger=logger,
            trajectory_collector=trajectory_collector,
        )
        
        # Play the game
        try:
            final_state = await coordinator.play_game(resume_from=resume_from_path)
            
            # Print results
            summary = coordinator.get_summary()
            print(f"\nGame {game_num} Complete!")
            print(f"Team A ({summary['team_a']['name']}): {summary['team_a']['score']} points")
            print(f"Team B ({summary['team_b']['name']}): {summary['team_b']['score']} points")
            print(f"Combined Total: {summary['total_score']} / {summary['max_possible_score']}")
            print(f"Efficiency: {summary['efficiency']:.1%}")
            print(f"Cooperation Rate: {summary['cooperation_rate']:.1%}")
            
            # Save trajectory if enabled
            if save_trajectory and coordinator.get_trajectory():
                trajectory = coordinator.get_trajectory()
                trajectory_file = trajectories_path / f"{trajectory.trajectory_id}.json"
                trajectory.save(str(trajectory_file))
                print(f"Trajectory saved to: {trajectory_file}")
                
                # Print trajectory summary
                traj_summary = trajectory.get_summary()
                print(f"  Timesteps: {traj_summary.get('total_timesteps', 0)}")
                print(f"  Total Actions: {traj_summary.get('total_actions', 0)}")
                print(f"  Dialogue Exchanges: {traj_summary.get('total_dialogue_exchanges', 0)}")
            
        except Exception as e:
            print(f"Error during game {game_num}: {e}")
            raise
    
    print(f"\n{'='*60}")
    print(f"Experiment Complete! Results saved to: {output_dir}")
    if save_trajectory:
        print(f"Trajectories saved to: {trajectories_path}")
    print(f"{'='*60}")


async def analyze_results(results_dir: str) -> None:
    """Analyze results from a directory of game logs.
    
    Args:
        results_dir: Path to directory containing game logs
    """
    collector = MetricsCollector()
    collector.load_from_directory(results_dir)
    
    print(collector.generate_summary())


async def analyze_trajectory(trajectory_path: str) -> None:
    """Analyze a single trajectory file.
    
    Args:
        trajectory_path: Path to trajectory JSON file
    """
    from redblackbench.trajectory import GameTrajectory
    
    trajectory = GameTrajectory.load(trajectory_path)
    
    print(f"\n{'='*60}")
    print(f"Trajectory Analysis: {trajectory.trajectory_id}")
    print(f"{'='*60}")
    
    print(f"\nGame Configuration:")
    for key, value in trajectory.game_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nTeams: {trajectory.team_a_name} vs {trajectory.team_b_name}")
    print(f"Start Time: {trajectory.start_time}")
    print(f"End Time: {trajectory.end_time}")
    
    # Summary stats
    summary = trajectory.get_summary()
    print(f"\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "rate" in key or "efficiency" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Action sequence
    actions = trajectory.get_action_sequence()
    print(f"\nAction Sequence ({len(actions)} total actions):")
    team_choices = [a for a in actions if a.action_type == "team_choice"]
    for action in team_choices:
        print(f"  Round {action.round_num}: {action.actor} chose {action.choice}")
    
    # Outcomes
    outcomes = trajectory.get_outcomes()
    print(f"\nRound Outcomes:")
    for outcome in outcomes:
        if outcome.outcome_type == "round":
            coop = "✓" if outcome.both_cooperated else ("✗" if outcome.both_defected else "~")
            print(f"  Round {outcome.round_num} ({outcome.multiplier}x): "
                  f"A={outcome.team_a_choice} ({outcome.team_a_score:+d}), "
                  f"B={outcome.team_b_choice} ({outcome.team_b_score:+d}) [{coop}]")
    
    if trajectory.final_outcome:
        fo = trajectory.final_outcome
        print(f"\nFinal Outcome:")
        print(f"  Team A Score: {fo.team_a_score}")
        print(f"  Team B Score: {fo.team_b_score}")
        print(f"  Total Score: {fo.total_score} / {fo.max_possible_score}")
        print(f"  Efficiency: {fo.efficiency:.1%}")
    print(f"  Cooperation Rate: {fo.cooperation_rate:.1%}")


async def provider_check(provider: str, model: str, api_key: Optional[str], max_tokens: int = 64) -> None:
    """Check provider connectivity and attempt minimal completion."""
    if provider == "openrouter":
        from redblackbench.providers.openrouter_provider import OpenRouterProvider
        # Enable reasoning for check to verify it works
        prov = OpenRouterProvider(model=model, api_key=api_key, temperature=0.0, max_tokens=max_tokens, include_reasoning=True)
        # List a few models
        try:
            models = await prov._client.models.list()
            print(f"Models available (first 5): {[m.id for m in models.data[:5]]}")
        except Exception as e:
            print(f"Model listing failed: {e}")
        # Retrieve target model
        try:
            m = await prov._client.models.retrieve(model)
            print(f"Model retrieve OK: {m.id}")
        except Exception as e:
            print(f"Model retrieve failed for '{model}': {e}")
        try:
            resp = await prov.generate(system_prompt="ping", messages=[{"role":"user","content":"ping"}])
            print(f"Chat completion succeeded!")
            # Check for hidden thinking delimiters
            if "__THINKING_START__" in resp:
                print("✓ Reasoning/Thinking tokens captured successfully (hidden from final output)")
                print(f"Raw output preview: {resp[:100]}...")
            else:
                print("⚠ No reasoning tokens found in response (Model might not support it or didn't think)")
                print(f"Response preview: {resp[:80]}...")
        except Exception as e:
            print(f"Chat completion failed: {e}")
            print("If error code is 402, you need OpenRouter credits: https://openrouter.ai/settings/credits")
    else:
        print(f"Unsupported provider: {provider}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RedBlackBench: Multi-Agent Game Theory Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment from config file
  redblackbench run --config experiments/configs/example.yaml
  
  # Run with trajectory collection disabled
  redblackbench run --config experiments/configs/example.yaml --no-trajectory
  
  # Analyze results
  redblackbench analyze --results-dir results/
  
  # Analyze a specific trajectory
  redblackbench trajectory --file results/trajectories/game_1.json
  
  # Quick test with default settings
  redblackbench run --quick-test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test game with default settings"
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory collection (saves memory/disk)"
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing trajectory if found"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment results")
    analyze_parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory containing game log files"
    )
    
    # Trajectory command
    traj_parser = subparsers.add_parser("trajectory", help="Analyze a trajectory file")
    traj_parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to trajectory JSON file"
    )
    # Provider check command
    pc_parser = subparsers.add_parser("provider-check", help="Check provider connectivity")
    pc_parser.add_argument("--provider", required=True, type=str, help="Provider type (e.g., openrouter)")
    pc_parser.add_argument("--model", required=True, type=str, help="Model ID to test")
    pc_parser.add_argument("--api-key", required=False, type=str, help="API key override")
    pc_parser.add_argument("--max-tokens", required=False, type=int, default=64, help="Max output tokens for the check (default: 64)")
    
    args = parser.parse_args()
    
    if args.command == "run":
        if args.config:
            config = load_config(args.config)
        elif args.quick_test:
            # Default config for quick testing
            config = {
                "experiment_name": "quick_test",
                "output_dir": args.output_dir,
                "num_games": 1,
                "default_provider": {
                    "type": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                },
                "game": {
                    "num_rounds": 10,
                    "team_size": 5,
                },
            }
        else:
            print("Error: Either --config or --quick-test is required")
            sys.exit(1)
        
        save_trajectory = not getattr(args, 'no_trajectory', False)
        resume = getattr(args, 'resume', False)
        asyncio.run(run_experiment(config, save_trajectory=save_trajectory, resume=resume))
        
    elif args.command == "analyze":
        asyncio.run(analyze_results(args.results_dir))
    
    elif args.command == "trajectory":
        asyncio.run(analyze_trajectory(args.file))
    elif args.command == "provider-check":
        asyncio.run(provider_check(args.provider, args.model, args.api_key, args.max_tokens))
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
