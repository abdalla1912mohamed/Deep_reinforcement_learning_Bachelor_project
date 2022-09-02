

import numpy as np
import gym

from gym_pathfinding.envs.pathfinding_env import PathFindingEnv


class PartiallyObservablePathFindingEnv(gym.Env):
    """ PartiallyObservableEnv
        -1 = unknown
    """

    def __init__(self, lines, columns, observable_depth, *, grid_type="free", screen_size=(640, 640)):
        self.env = PathFindingEnv(lines, columns, 
            grid_type=grid_type, 
            screen_size=screen_size
        )
        self.observable_depth = observable_depth

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        state = self.env.reset()
        return self.partial_state(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.partial_state(state), reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def render(self, mode='human'):
        grid = self.env.game.get_state()
        grid = self.partial_state(grid)

        if (mode == 'human'):
            self.env.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.env.close()

    def partial_state(self, state):
        return partial_grid(state, self.env.game.player, self.observable_depth)

    
def partial_grid(grid, center, observable_depth):
    """return the centered partial state, place -1 to non-visible cells"""
    
    i, j = center
    offset = observable_depth

    mask = np.ones_like(grid, dtype=bool)
    mask[max(0, i - offset): i + offset + 1, max(0, j - offset): j + offset + 1] = False

    _grid = np.array(grid, copy=True)
    _grid[mask] = -1
    return _grid

def create_partially_observable_pathfinding_env(id, name, lines, columns, observable_depth, *, grid_type="free"):

    def constructor(self):
        PartiallyObservablePathFindingEnv.__init__(self, lines, columns, observable_depth, grid_type=grid_type)
    
    env_class = type(name, (PartiallyObservablePathFindingEnv,), {
        "id" : id,
        "__init__": constructor
    })
    return env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [
    create_partially_observable_pathfinding_env(
        id="partially-observable-pathfinding-{type}-{n}x{n}-d{obs}-v0".format(
            type=grid_type, n=size, obs=obs_depth
        ),
        name="PartiallyObservablePathFinding{type}{n}x{n}d{obs}Env".format(
            type=grid_type.capitalize(), n=size, obs=obs_depth
        ),
        grid_type=grid_type,
        lines=size, 
        columns=size, 
        observable_depth=obs_depth
    ) 
    for grid_type in ["free", "obstacle", "maze"]
    for obs_depth in range(2, 10 + 1)
    for size in sizes 
]

for env_class in envs:
    globals()[env_class.__name__] = env_class

def get_env_classes():
    return envs


        
