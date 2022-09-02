# -*- coding: utf-8 -*-
import random
from queue import PriorityQueue

import numpy as np
import scipy.misc
import math

from gym_pathfinding.games.gridworld import generate_grid, is_legal, MOUVEMENT

class PathFindingGame(object):
    """
    A PathFinding games
    state : 
        0 = nothing
        1 = wall
        2 = player
        3 = goal
    """

    def __init__(self, lines=15, columns=15, *, grid_type="free"):
        self.lines = lines
        self.columns = columns
        self.shape = (lines, columns)

        self.grid_type = grid_type
        
        self.terminal = True

        self.generation_seed = self.spawn_seed = None
        self.grid = self.player = self.target = None

    def seed(self, generation_seed=None, spawn_seed=None):
        self.generation_seed = generation_seed
        self.spawn_seed = spawn_seed

    def reset(self):

        self.terminal = False

        self.grid, self.player, self.target = generate_grid(
            self.shape, 
            grid_type=self.grid_type,
            generation_seed=self.generation_seed, 
            spawn_seed=self.spawn_seed
        )

        return self.get_state()

    def getPlayer(self):
        return self.player
    
    def getLines(self):
        return self.lines

    def getColumns(self):
        return self.columns

    def getTerminal(self):
        return self.terminal

    def get_state(self):
        """ return a (n, n) grid """
        state = np.array(self.grid, copy=True)
        state[self.player[0], self.player[1]] = 2
        state[self.target[0], self.target[1]] = 3
        return state

    def get_stateSSMR(self):
        """ return a (n, n) grid """
        state = np.array(self.grid, copy=True)
        state[self.player[1], self.player[0]] = 2
        state[self.target[0], self.target[1]] = 3
        return state

    def step(self, a):
        if self.terminal:
            return self.step_return(1)
        
        assert 0 <= a and a < 4

        di, dj = MOUVEMENT[a]
        pi, pj = self.player

        next_i, next_j = pi + di, pj + dj

        if is_legal(self.grid, next_i, next_j):
            self.player = (next_i, next_j)
        else:                                           # edited by M
            return self.step_return(-1)

        if self.player == self.target:
            self.terminal = True
            return self.step_return(1)

        # x_t= self.target[0]
        # y_t= self.target[1]
        # e= math.sqrt((x_t-self.player[0])**2+(y_t-self.player[1])**2)
        # d=-0.1*e

        return self.step_return(-0.01)

    def step_return(self, reward):
        return self.get_state(), reward, self.terminal, ""



