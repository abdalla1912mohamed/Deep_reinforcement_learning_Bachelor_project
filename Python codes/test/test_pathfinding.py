
import pytest

import numpy as np

from gym_pathfinding.envs.partially_observable_env import partial_grid

def test_partial_state():
	state = np.array([
		[1,1,1,1,1,1,1,1],
		[1,0,2,0,0,0,0,1],
		[1,0,0,0,0,0,3,1],
		[1,0,0,0,0,0,0,1],
		[1,1,1,1,1,1,1,1]
	])

	partial_state = partial_grid(state, (1, 2), 2)

	assert np.all(partial_state == np.array([
		[ 1, 1, 1, 1, 1,-1,-1,-1],
		[ 1, 0, 2, 0, 0,-1,-1,-1],
		[ 1, 0, 0, 0, 0,-1,-1,-1],
		[ 1, 0, 0, 0, 0,-1,-1,-1],
		[-1,-1,-1,-1,-1,-1,-1,-1]
	]))

