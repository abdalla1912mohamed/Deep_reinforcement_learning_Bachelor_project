import math
from MCs.MobileRobot_class import MobileRobot
import gym
import gym_pathfinding
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

Q_values = ''
with open(r'training_robot4.txt','r') as f:
    for i in f.readlines():
        Q_values = i
Q_values = eval(Q_values)

def chooseAction(state):
    mx_nxt_reward = -999
    action = ""
    # print(state)
    if len(set(Q_values[state].values())) == 1:
        action = np.random.choice(actions)
    else:
        for a in actions:
            nxt_reward = Q_values[state][a]
            if nxt_reward >= mx_nxt_reward:
                action = a
                mx_nxt_reward = nxt_reward
    return action

def step(a) :
    if env.getTerminal() :
        return step_return (1)

    assert 0 <= a and a < 4

    robot.optimal_action(a)
    robot.take_step()

    if a == 0 or a == 1 :
        t = 40
    else:
        t = 10
        
    for i in range (0 ,t) :
        robot.take_step ( )
        next_i, next_j, robot.orientation = robot.get_discretized_state()

        if env.game.grid[next_i, next_j] == 1 :
            break

    if env.game.grid[next_i, next_j] == 0 : # is legal ?
        env.game.player = (next_i,next_j)
        # print(robot.env.game.player)

    if env.game.grid[next_i, next_j] == 1 :
        robot . assign_discretized_state ( env.game.player [ 0 ] , env.game. player [ 1 ] )
        return step_return(-1)

    if env.game.player == env.game.target :
        env.game.terminal = True
        return step_return (1)

    return step_return(-0.01)

def step_return (reward) :

    return get_state(), robot.orientation, reward , env.game.terminal

def get_state() :
# return a (n, n) grid #
    state = np.array(env.game.grid,copy=True)
    state[env.game.player[0], env.game.player[1]] = 2
    state[env.game.target[0], env.game.target[1]] = 3
    return state

def randomStart():
        # To start from a rondom position
        while True:    
            x = np.random.randint(0,env.getLines())
            y = np.random.randint(0,env.getColumns())
            if(env.game.grid[x,y] == 0):
                return (x,y)

def reset():
    env.reset()
    env.seed(1)
    # env.game.player = randomStart()
    robot.reset(env.game.player)      
    robot.state = robot.get_discretized_state()

robot = MobileRobot()
env = gym.make('pathfinding-obstacle-25x25-v0')
env.seed(1)
reset()
actions = [0,1,2,3]
done = False

plt.figure(1)
plt.xlim(0,4)
plt.ylim(4,0)
success = 0
robot.print_xya()
# robot.plot_robot()
plt.scatter(robot.pos_y, robot.pos_x, c= 'lime',edgecolors= 'lime' )
print(robot.get_discretized_state())
for ep in range(0,1):
    steps = 0
    while steps < 200:
        env.render()
        sleep(0.05)

        action = chooseAction(robot.get_discretized_state())
        state, orientation, r, done = step(action)
        print(action)
        plt.scatter(robot.pos_y, robot.pos_x, c= 'lime',edgecolors= 'lime' )
        robot.print_xya()
        print(robot.get_discretized_state())
        # robot.plot_robot()
        if done:
            robot.plot_xya()
            success += 1
            break
        steps += 1
    print("episode", ep)
    reset()

print(success)
plt.show()