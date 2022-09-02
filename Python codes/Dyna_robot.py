from MCs.MobileRobot_class import MobileRobot
import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from time import sleep

class DynaRobotAgent:

    def __init__(self, exp_rate=0.7, lr=0.9, gamma = 0.9, max_epochs = 800, n_steps=5, episodes=1):
        self.env = gym.make('pathfinding-obstacle-25x25-v0')
        self.robot = MobileRobot()
        self.state = self.robot.get_discretized_state() # (x,y,theta)
        self.actions = [0,1,2,3] # 0 F, 1 B, 2 L, 3 R
        self.state_actions = [] # state & action track
        self.exp_rate = exp_rate # Epsilon
        self.lr = lr # learning rate
        self.gamma = gamma                  
        self.max_epochs = max_epochs # maximum trials each episode
        self.success_episodes = 0           
        self.convergence_episode = 0               
        self.steps = n_steps # Planning steps
        self.episodes = episodes # number of episodes going to play
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []

        self.Q_values = {}
        # model function
        self.model = {}
        for row in range(self.env.getLines()):
            for col in range(self.env.getColumns()):
                for theta in range(-10,11):
                    self.Q_values[(row, col, theta)] = {}
                    for a in self.actions:
                        self.Q_values[(row, col, theta)][a] = 0

    def chooseAction(self):
        # epsilon-greedy
        mx_nxt_reward = -999
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.state
            # if all actions have same value, then select randomly
            if len(set(self.Q_values[current_position].values())) == 1:
                action = np.random.choice(self.actions)
            else:
                for a in self.actions:
                    nxt_reward = self.Q_values[current_position][a]
                    if nxt_reward >= mx_nxt_reward:
                        action = a
                        mx_nxt_reward = nxt_reward
        return action

    def randomStart(self):
        # To start from a rondom position
        while True:
            x = np.random.randint(0,self.env.getLines())
            y = np.random.randint(0,self.env.getColumns())
            if(self.env.game.grid[x,y] == 0):
                return (x,y)

    def reset(self):
        self.env.reset()
        self.env.seed(1)
        # self.env.game.player = self.randomStart()
        self.robot.reset(self.env.game.player)
        self.state = self.robot.get_discretized_state()
        self.state_actions = []

    def step(self, a) :
        if self.env.getTerminal() :
            return self.step_return (1)

        assert 0 <= a and a < 4
        
        self.robot.optimal_action(a)
        self.robot.take_step()

        if a == 0 or a == 1 :
            t = 40
        else:
            t = 10
        
        for i in range (0 ,t) :
            self.robot.take_step ( )
            next_i, next_j, self.robot.orientation = self.robot.get_discretized_state()

            if self.env.game.grid[next_i, next_j] == 1 :
                break

        next_i , next_j , self.robot.orientation = self.robot.get_discretized_state ( )

        if self.env.game.grid[next_i, next_j] == 0 : # is legal 
            self.env.game.player = (next_i,next_j)
        elif self.env.game.grid[next_i, next_j] == 1 :
            self.robot.assign_discretized_state(self.env.game.player[0], self.env.game.player[1])
            return self.step_return(-1)
             
        if self.env.getPlayer() == self.env.game.target :
            self.env.game.terminal = True
            return self.step_return (1) 
        
        return self.step_return(-0.01)

    def step_return (self, reward) :

        return self.env.game.get_state(), self.robot.orientation, reward , self.env.game.terminal

    def play(self):
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []
        self.reset()
        self.reset()
        for ep in range(self.episodes):
            epoches = 0
            cumulative_reward = 0
            while epoches < self.max_epochs:
                # self.env.render()
                # sleep(0.005)
                action = self.chooseAction()
                self.state_actions.append((self.state, action))
            
                nxtState2D, _ , reward, self.env.game.terminal = self.step(action) 
                nxtState = self.robot.get_discretized_state()

                cumulative_reward += reward
                
                # update Q-value
                self.Q_values[self.state][action] += self.lr*(reward + self.gamma*np.max(list(self.Q_values[nxtState].values())) - self.Q_values[self.state][action])

                # update model
                if self.state not in self.model.keys():
                    self.model[self.state] = {}
                self.model[self.state][action] = (reward, nxtState)
                self.state = nxtState
                # print(self.env.getTerminal())

                # loop n times to randomly update Q-value
                for _ in range(self.steps):
                    # randomly choose an state
                    rand_idx = np.random.choice(range(len(self.model.keys())))
                    _state = list(self.model)[rand_idx]
                    # randomly choose an action
                    rand_idx = np.random.choice(range(len(self.model[_state].keys())))
                    _action = list(self.model[_state])[rand_idx]

                    _reward, _nxtState = self.model[_state][_action]

                    self.Q_values[_state][_action] += self.lr*(_reward + self.gamma*np.max(list(self.Q_values[_nxtState].values())) - self.Q_values[_state][_action])       
                # print(self.env.getTerminal())

                epoches +=1
                if epoches % 100 == 0:
                    print(f"Epoches: {epoches}")
                if(self.env.getTerminal()):
                    if epoches < self.max_epochs :
                        self.max_epochs = (epoches + self.max_epochs)/2
                        # self.max_epochs = epoches
                        self.convergence_episode = ep
                    self.success_episodes += 1
                    break

            # end of game
            if(self.exp_rate > 0.00001):
                self.exp_rate = self.exp_rate*0.99
            if ep % 100 == 0:
                print("episode", ep)
            self.steps_per_episode.append(len(self.state_actions))
            self.cumulative_reward_per_episode.append(cumulative_reward)
            self.reset()

if __name__ == "__main__":
    N_EPISODES = 2500
    agent = DynaRobotAgent(n_steps=10, episodes=N_EPISODES)
    agent.play()

    steps_episode_0 = agent.steps_per_episode
    cumulative_r_0 = agent.cumulative_reward_per_episode

    # agent = DynaRobotAgent(n_steps=100, episodes=N_EPISODES)
    # agent.play()

    # steps_episode_100 = agent.steps_per_episode
    # cumulative_r_100 = agent.cumulative_reward_per_episode

    # Save the Q-table in a text file
    with open(r'training_robot.txt','w+') as f:
        f.write(str(agent.Q_values))
    
    noEpoches = agent.max_epochs
    conv = agent.convergence_episode
    successRate = (agent.success_episodes / N_EPISODES)*100
    
    plt.figure(1)
    plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
    # plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.legend()

    plt.figure(2)
    plt.plot(range(N_EPISODES), cumulative_r_0, label="step=0")
    # plt.plot(range(N_EPISODES), cumulative_r_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.legend()

    print(f"No. epoches: {noEpoches}")
    print(f"Success Rate: {successRate}")
    print(f"Convergence speed After {conv} episodes")
    
    plt.show()