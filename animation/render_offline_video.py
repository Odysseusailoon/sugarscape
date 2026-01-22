import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import Normalize

def render_animation(input_file, output_file, fps=10):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    frames = data['frames']
    metadata = data['metadata']
    width = metadata['width']
    height = metadata['height']
    
    print(f"Loaded {len(frames)} frames. Grid size: {width}x{height}")
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Sugar Map Layer
    # Initial empty map
    sugar_map_img = ax.imshow(np.zeros((height, width)), origin='lower', cmap='YlOrBr', vmin=0, vmax=4, alpha=0.6)
    
    # Agents Layer
    # We'll use a scatter plot
    # Initialize with empty data
    scat = ax.scatter([], [], c=[], s=[], cmap='viridis', edgecolors='black', linewidth=0.5, alpha=0.9, vmin=0, vmax=100)
    
    # Text info
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', weight='bold')
    pop_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, color='black')
    
    # Colorbar for agents (Wealth)
    cbar = plt.colorbar(scat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Agent Wealth')
    
    def update(frame_idx):
        frame = frames[frame_idx]
        tick = frame['tick']
        
        # Update Sugar Map
        # Note: imshow expects (height, width), and often the simulation stores it as (width, height) or similar.
        # We might need to transpose depending on how it was saved.
        # In generate_animation_data, we did `sim.env.sugar_amount.tolist()`.
        # Assuming numpy conventions, check if we need transpose.
        grid = np.array(frame['sugar_map']).T 
        sugar_map_img.set_data(grid)
        
        # Update Agents
        agents = frame['agents']
        if agents:
            x = [a['pos'][0] + 0.5 for a in agents] # Center in cell
            y = [a['pos'][1] + 0.5 for a in agents]
            c = [min(a['wealth'], 100) for a in agents] # Cap color at 100 wealth
            s = [30 + min(a['wealth'], 200) * 0.5 for a in agents] # Size based on wealth
            
            # Colors based on type?
            # Let's stick to Wealth for now, it's classic Sugarscape.
            # Or we could color by 'leaning' if it exists for Identity demo.
            
            scat.set_offsets(np.c_[x, y])
            scat.set_array(np.array(c))
            scat.set_sizes(s)
        else:
            scat.set_offsets(np.zeros((0, 2)))
            
        time_text.set_text(f'Tick: {tick}')
        pop_text.set_text(f'Population: {len(agents)}')
        
        return sugar_map_img, scat, time_text, pop_text

    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    print(f"Saving to {output_file}...")
    if output_file.endswith('.gif'):
        ani.save(output_file, writer='pillow', fps=fps)
    else:
        # Try default (likely ffmpeg), fall back or warn if fails
        try:
            ani.save(output_file, fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            print(f"Error saving video (ffmpeg might be missing): {e}")
            print("Trying to save as .gif instead...")
            ani.save(output_file.replace('.mp4', '.gif'), writer='pillow', fps=fps)
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="animation_data.json")
    parser.add_argument("--output", type=str, default="sugarscape_demo.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    
    render_animation(args.input, args.output, args.fps)
