import asyncio
import websockets
import json
import argparse
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from redblackbench.sugarscape.simulation import SugarSimulation
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.llm_agent import LLMSugarAgent

class SimulationServer:
    def __init__(self, host="localhost", port=8765, config=None):
        self.host = host
        self.port = port
        self.config = config or SugarscapeConfig()
        self.sim = SugarSimulation(config=self.config, experiment_name="web_demo")
        self.connected_clients = set()
        self.is_running = False
        self.tick_rate = 10  # Ticks per second target
        
        print(f"Simulation initialized. Grid: {self.config.width}x{self.config.height}")

    async def register(self, websocket):
        self.connected_clients.add(websocket)
        print(f"Client connected. Total: {len(self.connected_clients)}")
        # Send initial config/metadata
        await websocket.send(json.dumps({
            "type": "init",
            "width": self.config.width,
            "height": self.config.height,
            "config": self.config.__dict__
        }))

    async def unregister(self, websocket):
        self.connected_clients.remove(websocket)
        print(f"Client disconnected. Total: {len(self.connected_clients)}")

    async def broadcast_state(self):
        if not self.connected_clients:
            return

        # 1. Serialize State
        # Sugar Grid (flattened or 2D)
        sugar_map = self.sim.env.sugar_amount.tolist() # 2D array
        
        # Agents
        agents_data = []
        for agent in self.sim.agents:
            if not agent.alive: continue
            
            agent_type = "LLM" if isinstance(agent, LLMSugarAgent) else "Rule"
            
            # Identity/Leaning
            leaning = 0.0
            if hasattr(agent, "self_identity_leaning"):
                leaning = float(agent.self_identity_leaning)
            
            # Persona
            persona = getattr(agent, "persona", "A")

            agents_data.append({
                "id": agent.agent_id,
                "x": agent.pos[0],
                "y": agent.pos[1],
                "w": int(agent.wealth),
                "s": int(agent.spice),
                "type": agent_type,
                "p": persona, # Persona
                "l": leaning  # Leaning (-1 to 1)
            })

        state = {
            "type": "update",
            "tick": self.sim.tick,
            "pop": len(agents_data),
            "grid": sugar_map,
            "agents": agents_data
        }

        # 2. Broadcast
        message = json.dumps(state)
        # Create tasks for sending to all clients
        tasks = [client.send(message) for client in self.connected_clients]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def simulation_loop(self):
        print("Starting simulation loop...")
        while True:
            start_time = time.time()
            
            if self.is_running and self.connected_clients:
                # Step Simulation
                # Note: This is blocking. For LLM agents, this might take time.
                # In a production app, we'd offload this to a thread, 
                # but SugarSimulation isn't fully thread-safe.
                try:
                    self.sim.step()
                    await self.broadcast_state()
                except Exception as e:
                    print(f"Error in simulation step: {e}")
                    import traceback
                    traceback.print_exc()
                    self.is_running = False
            
            # Rate Limiting
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / self.tick_rate) - elapsed)
            await asyncio.sleep(sleep_time)

    async def handler(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                command = data.get("command")
                
                if command == "start":
                    self.is_running = True
                    print("Received START command")
                elif command == "pause":
                    self.is_running = False
                    print("Received PAUSE command")
                elif command == "step":
                    self.is_running = False
                    self.sim.step()
                    await self.broadcast_state()
                    print("Received STEP command")
                elif command == "reset":
                    self.is_running = False
                    print("Received RESET command - Re-initializing...")
                    self.sim = SugarSimulation(config=self.config, experiment_name="web_demo")
                    await self.broadcast_state()
                elif command == "speed":
                    self.tick_rate = int(data.get("value", 10))
                    print(f"Speed set to {self.tick_rate} tps")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def main(self):
        print(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handler, self.host, self.port):
            await self.simulation_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Enable LLM agents")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--port", type=int, default=8765)
    
    parser.add_argument("--test-mode", action="store_true", help="Run briefly to test server start")
    
    args = parser.parse_args()
    
    config = SugarscapeConfig()
    config.width = args.width
    config.height = args.height
    config.initial_population = args.population
    
    if args.with_llm:
        config.enable_llm_agents = True
        config.llm_agent_ratio = 0.1
    else:
        config.enable_llm_agents = False
        
    server = SimulationServer(port=args.port, config=config)
    
    if args.test_mode:
        print("Test mode: Server initialized successfully.")
        sys.exit(0)

    try:
        asyncio.run(server.main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
