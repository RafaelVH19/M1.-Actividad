from mesa import Agent, Model # Provides the base classes for the agents and the model
from mesa.space import MultiGrid # Provides the grid for the agents to move in
import numpy as np # Handles the position arrays and general required math operations
import matplotlib.pyplot as plt # Creates the visual ouput for the animation
from matplotlib.animation import FuncAnimation # Supports the process for the animation creation
import random # Adds randomness to agent movements

# Set board size constants
M = 6  # Width
N = 6  # Height
max_steps = 100
agent_count = 3
dirty_count = 12

# Agent for the board
class DummyAgent(Agent):

    # Initialization for the agent
    def __init__(self, unique_id, model):
        super().__init__(model) # Initializing the base Agent class Mesa offers requires an existing model
        self.unique_id = unique_id
        self.next_pos = None #use intent to check  for next position on the board
        self.moves_made = 0  # Track actual moves

    # To get an agent's current position on the baord
    def get_position(self): 
        return np.array(self.pos)

    # To take a step and possibly move on the board
    def step(self):
        self.intent()

    #To check what is and isn't the border
    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height # Returns true if the position is in bounds

    # How the agent plans to move, decided at random
    def intent(self):
        # Possible moves: left, right, up, down, diagonals
        moves = [
            np.array([-1, 0]),  # left
            np.array([1, 0]),   # right
            np.array([0, 1]),   # up
            np.array([0, -1]),  # down
            np.array([-1, -1]), # left down
            np.array([1, 1]),   # right up
            np.array([-1, 1]),  # left up
            np.array([1, -1])   # right down
        ]
        move = self.random.choice(moves)
        new_pos = tuple(self.get_position() + move) # Gets the coordinates for the new position
        if self.in_bounds(new_pos):
            self.next_pos = new_pos
        else:
            self.next_pos = tuple(self.get_position()) # Stays in place if moved out of bounds

    # Make the move on the baord
    def advance(self):
        if self.next_pos and self.in_bounds(self.next_pos): # In case it was moved elsewhere, checks again
            if self.next_pos != self.pos:
                self.model.grid.move_agent(self, self.next_pos)
                self.moves_made += 1  # Count only actual moves

# Model for the board 
class DummyModel(Model):

    # Initializes by setting itself up with every constant
    def __init__(self, height=N, width=M, agent_count=agent_count, dirty_count=dirty_count, max_steps=max_steps):
        super().__init__() # Initializes the base model class of Mesa
        self.grid = MultiGrid(width, height, torus=False) # torus=False ensure there is no wrap-around
        self._agents = {}
        for i in range(agent_count):
            agent = DummyAgent(i, self) # Creates an agent based on the DummyAgent class and this model
            self.grid.place_agent(agent, (0, N-1)) # Top left corner
            self._agents[i] = agent
        self.dirty_tiles = []
        self.dirty_count = dirty_count
        self.spawn_dirty_tiles()  # Only spawn at the beginning
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.cleaned_count = 0
        self.last_cleaned_step = None

    # Read-only, gets the agents and ensures they can't be changed later
    @property
    def agents(self):
        return [agent for agent in self._agents.values() if agent is not None]

    # Spawns every dirty tile
    def spawn_dirty_tiles(self):
        # Only called once at init
        needed = self.dirty_count - len(self.dirty_tiles) # In case it's ever needed to call for new dirty tiles mid-simulation
        free_cells = [
            (x, y)
            for x in range(self.grid.width)
            for y in range(self.grid.height)
            if self.grid.is_cell_empty((x, y)) and (x, y) not in self.dirty_tiles # Function given by Mesa to check for empty cells
        ]
        for _ in range(min(needed, len(free_cells))):
            self.dirty_tiles.append(random.choice(free_cells))
            free_cells.remove(self.dirty_tiles[-1]) # Removes the cell that was just chosen so it cannot be picked again

    # To make the agents take a step and advance
    def step(self):
        if self.done:
            return
        for agent in self.agents:
            agent.step()
        for agent in self.agents:
            agent.advance()

        # Mark tiles for deletion if an agent is on them
        to_remove = []
        for tile in self.dirty_tiles:
            for agent in self.agents:
                if tuple(agent.pos) == tile:
                    to_remove.append(tile)
                    break
        # Actually remove them from the logic
        for tile in to_remove:
            self.dirty_tiles.remove(tile)
            self.cleaned_count += 1
            self.last_cleaned_step = self.current_step + 1  # + 1 for ease of reading

        # Advance the step on the boared
        self.current_step += 1
        if not self.dirty_tiles or self.current_step >= self.max_steps: # If a finish condition is met, conclude the board
            self.done = True
            self.print_summary()

    # Summary after baord is done, contains the data for analysis
    def print_summary(self):
        total_moves = sum(agent.moves_made for agent in self.agents)
        if not self.dirty_tiles:
            print(f"All dirty tiles cleaned at step {self.last_cleaned_step}.")
            print(f"Total tiles cleaned: {self.cleaned_count}")
            print(f"Total agent moves: {total_moves}")
        else:
            print(f"Max steps ({self.max_steps}) reached at step {self.current_step}.")
            print(f"Total tiles cleaned: {self.cleaned_count}")
            print(f"Total agent moves: {total_moves}")

# Sets up all of the visuals for the animation
def animate_agents(model):
    fig, ax = plt.subplots(figsize=(5, 3)) # Creates the screen
    
    # Makes the visual for the board
    ax.set_xlim(0, model.grid.width)
    ax.set_ylim(0, model.grid.height)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, model.grid.width, 1))
    ax.set_yticks(np.arange(0, model.grid.height, 1))
    ax.grid(True, which='both') # Turns on grid lines
    scat = ax.scatter([], [], s=200) # Creates the scatter plot to make the dots for the agents
    texts = [] # Initializes the list for the agent IDs
    dirty_patches = [] # Initializes the list for the dirty tiles

    # Colours for up to 10 agents, repeats last colour if more
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'purple', 'brown', 'black', 'grey'] 

    # Initializes the animations by ensuring everything is properly reset
    def init():
        scat.set_offsets(np.empty((0, 2))) # Sets up the offsets to show agents on the proper place
        for t in texts:
            t.remove()
        texts.clear()
        for patch in dirty_patches:
            patch.remove()
        dirty_patches.clear()
        return scat, *texts

    # Function called every step to show updates on the board as agents move
    def update(frame):
        if frame > 0 and not model.done:
            model.step()
        # Remove previous patches to avoid drawing over with new ones
        for patch in dirty_patches:
            patch.remove()
        dirty_patches.clear()
        # Draw dirty tiles currently remaining
        for x, y in model.dirty_tiles:
            dirty_patch = plt.Rectangle((x, y), 1, 1, color='yellow', alpha=0.3)
            ax.add_patch(dirty_patch)
            dirty_patches.append(dirty_patch)
        # Draw agents (draw after tiles)
        positions = np.array([[agent.pos[0]+0.5, agent.pos[1]+0.5] for agent in model.agents]) # Offset so they are on the center of the square
        if positions.size == 0:
            scat.set_offsets(np.empty((0, 2)))
        else:
            scat.set_offsets(positions) # Shows the agents in their current positions
        scat.set_color(colors[:len(positions)])
        # Removes previous ID text then rewrites it in the correct position
        for t in texts:
            t.remove()
        texts.clear()
        for idx, agent in enumerate(model.agents):
            t = ax.text(agent.pos[0]+0.5, agent.pos[1]+0.5, str(agent.unique_id),
                        color='white', ha='center', va='center', fontsize=8, fontweight='bold')
            texts.append(t)
        return scat, *texts

    # Run until either all dirty tiles are gone or max_steps is reached
    frames = model.max_steps + 1
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False) # Calls from a function from the Mesa library
    ani.save('animation.gif', writer='pillow', fps=2)  # Save as GIF using Pillow
    plt.show() # Displays it
    return

# Board is set up and animation starts
if __name__ == "__main__":
    model = DummyModel()
    animate_agents(model)