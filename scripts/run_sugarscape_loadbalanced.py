#!/usr/bin/env python3
"""Run Sugarscape with load-balanced LLM providers.

Uses both OpenRouter and AIHubMix to handle high agent concurrency.
"""

import sys
import os
import argparse
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.providers import (
    OpenRouterProvider,
    AIHubMixProvider,
    LoadBalancedProvider,
    LoadBalancerConfig,
    LoadBalanceStrategy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sugarscape with Load-Balanced LLM Providers")
    
    # Experiment parameters
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of simulation ticks")
    parser.add_argument("--population", type=int, default=100,
                        help="Initial population size")
    parser.add_argument("--grid-size", type=int, default=50,
                        help="Grid size (width and height)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="qwen/qwen3-30b-a3b-thinking-2507",
                        help="Model identifier (used for both providers)")
    parser.add_argument("--openrouter-model", type=str, default=None,
                        help="OpenRouter model (overrides --model)")
    parser.add_argument("--aihubmix-model", type=str, default=None,
                        help="AIHubMix model (overrides --model)")
    
    # Load balancing
    parser.add_argument("--strategy", type=str, 
                        choices=["round_robin", "weighted", "least_pending", "random", "failover"],
                        default="round_robin",
                        help="Load balancing strategy")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Max concurrent requests per provider")
    parser.add_argument("--openrouter-weight", type=float, default=0.5,
                        help="OpenRouter weight for weighted strategy")
    parser.add_argument("--aihubmix-weight", type=float, default=0.5,
                        help="AIHubMix weight for weighted strategy")
    
    # Provider selection
    parser.add_argument("--providers", type=str, default="both",
                        choices=["openrouter", "aihubmix", "both"],
                        help="Which providers to use")
    
    # Goal configuration
    parser.add_argument("--goal-preset", type=str,
                        choices=["survival", "wealth", "egalitarian", "utilitarian"],
                        default="survival",
                        help="Goal preset for LLM agents")
    
    # Difficulty
    parser.add_argument("--difficulty", type=str,
                        choices=["standard", "easy", "harsh", "desert"],
                        default="standard",
                        help="Difficulty preset")
    
    # Token limits
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens per LLM response (default: 512, good for thinking models)")
    
    # Trade
    parser.add_argument("--enable-trade", action="store_true",
                        help="Enable trading between agents")
    parser.add_argument("--trade-rounds", type=int, default=4,
                        help="Maximum trade dialogue rounds")
    parser.add_argument("--trade-mode", type=str, default="dialogue",
                        choices=["mrs", "dialogue"],
                        help="Trade mode: mrs (automatic) or dialogue (LLM negotiation)")
    
    return parser.parse_args()


def create_load_balanced_provider(args) -> LoadBalancedProvider:
    """Create load-balanced provider based on args."""
    
    providers = []
    weights = {}
    
    openrouter_model = args.openrouter_model or args.model
    aihubmix_model = args.aihubmix_model or args.model
    
    # Map model name for AIHubMix (they may use different naming)
    # Common mappings:
    aihubmix_model_map = {
        "qwen/qwen3-30b-a3b-thinking-2507": "Qwen/Qwen3-30B-A3B",
        "moonshotai/kimi-k2-thinking": "kimi-k2-thinking",
    }
    aihubmix_model = aihubmix_model_map.get(aihubmix_model, aihubmix_model)
    
    if args.providers in ["openrouter", "both"]:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Export it or create a .env file with OPENROUTER_API_KEY=..."
            )
        
        openrouter = OpenRouterProvider(
            model=openrouter_model,
            api_key=openrouter_key,
            max_tokens=args.max_tokens,
            rate_limit_concurrent=args.max_concurrent,
            enable_retry=True,
            enable_circuit_breaker=True,
        )
        providers.append(openrouter)
        weights["openrouter"] = args.openrouter_weight
        print(f"✓ OpenRouter provider initialized: {openrouter_model}")
    
    if args.providers in ["aihubmix", "both"]:
        aihubmix_key = os.environ.get("AIHUBMIX_API_KEY")
        if not aihubmix_key:
            print("WARNING: AIHUBMIX_API_KEY not set")
        
        aihubmix = AIHubMixProvider(
            model=aihubmix_model,
            api_key=aihubmix_key,
            max_tokens=args.max_tokens,
            rate_limit_concurrent=args.max_concurrent,
            enable_retry=True,
            enable_circuit_breaker=True,
        )
        providers.append(aihubmix)
        weights["aihubmix"] = args.aihubmix_weight
        print(f"✓ AIHubMix provider initialized: {aihubmix_model}")
    
    if not providers:
        raise ValueError("No providers configured!")
    
    # Map strategy string to enum
    strategy_map = {
        "round_robin": LoadBalanceStrategy.ROUND_ROBIN,
        "weighted": LoadBalanceStrategy.WEIGHTED,
        "least_pending": LoadBalanceStrategy.LEAST_PENDING,
        "random": LoadBalanceStrategy.RANDOM,
        "failover": LoadBalanceStrategy.FAILOVER,
    }
    
    config = LoadBalancerConfig(
        strategy=strategy_map[args.strategy],
        max_concurrent_per_provider=args.max_concurrent,
        weights=weights,
        failure_threshold=15,
        recovery_timeout=30.0,
    )
    
    lb = LoadBalancedProvider(providers, config)
    print(f"✓ Load balancer initialized: strategy={args.strategy}, providers={len(providers)}")
    
    return lb


def main():
    args = parse_args()
    
    print("=" * 60)
    print("SUGARSCAPE WITH LOAD-BALANCED LLM PROVIDERS")
    print("=" * 60)
    print(f"Population: {args.population} agents")
    print(f"Ticks: {args.ticks}")
    print(f"Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Strategy: {args.strategy}")
    print(f"Goal: {args.goal_preset}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Trade: {'ENABLED (' + args.trade_mode + ')' if args.enable_trade else 'DISABLED'}")
    print("=" * 60)
    
    # Create load-balanced provider
    lb_provider = create_load_balanced_provider(args)
    
    # Configure simulation
    config = SugarscapeConfig(
        width=args.grid_size,
        height=args.grid_size,
        initial_population=args.population,
        max_ticks=args.ticks,
        seed=args.seed,
        enable_spice=True,  # Always enable spice for Sugarscape 2
        # Trade settings
        enable_trade=args.enable_trade,
        trade_mode=args.trade_mode,
        trade_dialogue_rounds=args.trade_rounds,
    )
    
    # Apply difficulty
    if args.difficulty == "easy":
        config.sugar_growback_rate = 2
        config.max_sugar_capacity = 6
        config.spice_growback_rate = 2
        config.max_spice_capacity = 6
    elif args.difficulty == "harsh":
        config.sugar_growback_rate = 1
        config.max_sugar_capacity = 2
        config.spice_growback_rate = 1
        config.max_spice_capacity = 2
    elif args.difficulty == "desert":
        config.sugar_growback_rate = 0
        config.max_sugar_capacity = 4
        config.spice_growback_rate = 0
        config.max_spice_capacity = 4
    
    # Configure LLM agents with load-balanced provider
    config.enable_llm_agents = True
    config.llm_agent_ratio = 1.0  # All agents are LLM
    config.llm_provider_model = args.model  # For logging
    config.llm_goal_preset = args.goal_preset
    
    # Create simulation with custom provider
    from redblackbench.sugarscape.llm_agent import LLMSugarAgent
    
    # Create experiment name
    trade_suffix = f"_trade_{args.trade_mode}" if args.enable_trade else "_notrade"
    experiment_name = f"loadbalanced_{args.goal_preset}{trade_suffix}"
    
    # Override the provider in simulation
    sim = SugarSimulation(config=config, experiment_name=experiment_name)
    
    # Inject load-balanced provider into all LLM agents
    for agent in sim.agents:
        if isinstance(agent, LLMSugarAgent):
            agent.provider = lb_provider
    
    print("\nInitial Stats:")
    stats = sim.get_stats()
    print(f"Population: {stats['population']}, Mean Wealth: {stats['mean_wealth']:.2f}")
    
    print(f"\nRunning for {args.ticks} ticks...")
    
    try:
        for i in range(args.ticks):
            sim.step()
            
            # Progress update every 10 ticks
            if (i + 1) % 10 == 0:
                stats = sim.get_stats()
                lb_stats = lb_provider.get_stats()
                
                # Provider health status
                health_str = ", ".join(
                    f"{name}: {'✓' if p['is_healthy'] else '✗'}"
                    for name, p in lb_stats['providers'].items()
                )
                
                print(f"Tick {i+1}: Pop={stats['population']}, "
                      f"Wealth={stats['mean_wealth']:.1f}, "
                      f"Gini={stats['gini']:.2f} | "
                      f"Providers: {health_str}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    # Final stats
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    final_stats = sim.get_stats()
    print(f"Final Population: {final_stats['population']}")
    print(f"Mean Wealth: {final_stats['mean_wealth']:.2f}")
    print(f"Gini Coefficient: {final_stats['gini']:.3f}")
    print(f"Survival Rate: {final_stats['survival_rate']:.2%}")
    
    # Load balancer stats
    print("\n--- Load Balancer Statistics ---")
    lb_stats = lb_provider.get_stats()
    for name, p_stats in lb_stats['providers'].items():
        print(f"\n{name}:")
        print(f"  Total Requests: {p_stats['total_requests']}")
        print(f"  Success Rate: {p_stats['success_rate']:.2%}")
        print(f"  Avg Latency: {p_stats['avg_latency_ms']:.0f}ms")
        print(f"  Healthy: {p_stats['is_healthy']}")
    
    print(f"\nQueue: {lb_stats['queue']}")
    print(f"Healthy Providers: {lb_stats['healthy_providers']}/{lb_stats['total_providers']}")
    
    # Save trajectory data (conversations, actions, etc.)
    trajectory_path = sim.logger.get_log_path("trajectory.json")
    sim.trajectory.save(trajectory_path)
    print(f"\n✓ Trajectory saved: {trajectory_path}")
    
    # Save detailed agent data including trade memory
    agent_data = []
    for agent in sim.agents:
        agent_info = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "alive": agent.alive,
            "wealth": agent.wealth,
            "spice": agent.spice,
            "age": agent.age,
            "pos": agent.pos,
        }
        
        # Add trade memory if exists
        if hasattr(agent, 'trade_memory') and agent.trade_memory:
            agent_info["trade_memory"] = {
                str(partner_id): list(logs) 
                for partner_id, logs in agent.trade_memory.items()
            }
        
        # Add conversation history for LLM agents
        if hasattr(agent, 'conversation_history') and agent.conversation_history:
            agent_info["conversation_history"] = list(agent.conversation_history)
        
        # Add move history for LLM agents
        if hasattr(agent, 'move_history') and agent.move_history:
            agent_info["move_history"] = list(agent.move_history)
            
        agent_data.append(agent_info)
    
    import json
    agents_path = sim.logger.get_log_path("agents_data.json")
    with open(agents_path, 'w') as f:
        json.dump(agent_data, f, indent=2, default=str)
    print(f"✓ Agent data saved: {agents_path}")


if __name__ == "__main__":
    main()


