# gym-pathfinding

An OpenAI Gym implementation for the pathfinding problem.

<p align="center">
   <img src="./env_examples.png" alt="env-examples" width="70%">
</p>

## Installation
```bash
git clone [url_repo]
cd gym-pathfinding
pip install -e .
```

## Basic Usage
```python
import gym
import gym_pathfinding
from time import sleep

env = gym.make('pathfinding-obstacle-25x25-v0')
env.seed(1) # full deterministic env

for episode in range(10):
    s = env.reset()
    
    for timestep in range(50):
        env.render()
        sleep(0.05)
        
        s, r, done, _ = env.step(env.action_space.sample())

        if done:
            break

env.close()
```

## Environnements

#### Examples

```
pathfinding-free-9x9-v0
pathfinding-maze-25x25-v0
pathfinding-obstacle-25x25-v0

partially-observable-pathfinding-free-9x9-d3-v0
partially-observable-pathfinding-obstacle-25x25-d5-v0
```

#### Naming
```
pathfinding-[grid_type]-[size_w]x[size_h]-v0 
partially-observable-pathfinding-[grid_type]-[size_w]x[size_h]-d[observation_depth]-v0
```

- grid_type : either "free", "obstacle" or "maze"
- size_w and size_h : 9, 11, 13, 15, 17, 19, 25, 35, 55
- observation_depth : between 2 and 10

## Information
The initiale project come from https://github.com/cair/gym-maze

## Licence
Copyright 2017 Per-Arne Andersen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
