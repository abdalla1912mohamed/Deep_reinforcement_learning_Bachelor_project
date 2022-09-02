
import pytest

import numpy as np
from gym_pathfinding.games.gridworld import path_exists

def test_path_exists():
    grid = np.array([
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1],
    ])

    assert path_exists(grid, (1, 1), (1, 5))
    assert not path_exists(grid, (1, 1), (3, 5))


from gym_pathfinding.games.gridworld import add_rectangle

def test_add_rectangle():
    grid = np.array([
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
    ])

    expected = np.array([
        [0,0,0,0,0,0,0],
        [0,1,1,1,0,0,0],
        [0,1,1,1,0,0,0],
        [0,0,0,1,1,0,0],
        [0,0,0,1,1,0,0],
    ])
    
    add_rectangle(grid, (1, 1, 3, 2))
    add_rectangle(grid, (3, 3, 2, 2))
    
    assert np.all(grid == expected)
    





