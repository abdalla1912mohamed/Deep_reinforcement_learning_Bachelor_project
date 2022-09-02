import pygame
import numpy as np
import gym
from gym import error, spaces, utils

from gym_pathfinding.games.pathfinding import PathFindingGame
from gym_pathfinding.rendering import GridViewer


class PathFindingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self, lines, columns, *, grid_type="free", screen_size=(640, 640)):
        self.game = PathFindingGame(lines, columns, grid_type=grid_type)
        self.game.reset()
        
        self.viewer = GridViewer(screen_size[0], screen_size[1], lines, columns)

        shape = self.game.get_state().shape
        self.observation_space = spaces.Box(low=0, high=3, shape=shape, dtype=np.int8)
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        return self.game.reset()

    def step(self, action):
        return self.game.step(action)

    def seed(self, seed=None):
        self.game.seed(generation_seed=seed, spawn_seed=seed)

    def getPlayer(self):
        return self.game.getPlayer()
    
    def getLines(self):
        return self.game.getLines()

    def getColumns(self):
        return self.game.getColumns()

    def getTerminal(self):
        return self.game.getTerminal()

    def render(self, mode='human'):
        grid = self.game.get_state()

        if (mode == 'human'):
            self.viewer.draw(grid)
        elif (mode == 'array'):
            return grid

    def close(self):
        self.viewer.stop()

    # For image representation !

    # @staticmethod
    # def rgb2gray(rgb):
    #     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    # def get_state(self):

    #     if self.state_representation == "image":
    #         arr = pygame.surfarray.array3d(self.viewer.screen)
    #         arr = scipy.misc.imresize(arr, self.image_state_size)
    #         arr = arr / 255
    #         arr = np.expand_dims(arr, axis=0)
    #         return arr

    #     elif self.state_representation == "image_gray":
    #         arr = pygame.surfarray.array3d(self.screen)
    #         arr = scipy.misc.imresize(arr, self.image_state_size)
    #         arr = MazeGame.rgb2gray(arr)
    #         arr = arr.reshape(*arr.shape, 1)
    #         arr = np.expand_dims(arr, axis=0)
    #         return arr

    #     elif self.state_representation == "array":
    #         return self.game.get_state()
            

def create_pathfinding_env(id, name, lines, columns, grid_type="free"):

    def constructor(self):
        PathFindingEnv.__init__(self, lines, columns, grid_type=grid_type)
    
    pathfinding_env_class = type(name, (PathFindingEnv,), {
        "id" : id,
        "__init__": constructor
    })
    return pathfinding_env_class


# Create classes 

sizes = list(range(9, 20, 2)) + [25, 35, 55]
envs = [
    create_pathfinding_env(
        id="pathfinding-{type}-{n}x{n}-v0".format(
            type=grid_type, n=size
        ),
        name="PathFinding{type}{n}x{n}Env".format(
            type=grid_type.capitalize(), n=size
        ),
        grid_type=grid_type,
        lines=size, 
        columns=size
    ) 
    for grid_type in ["free", "obstacle", "maze"]
    for size in sizes 
]

for env_class in envs:
    globals()[env_class.__name__] = env_class


def get_env_classes():
    return envs
