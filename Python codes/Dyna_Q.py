import gym
import gym_pathfinding
import numpy as np
import random
import matplotlib.pyplot as plt
from time import sleep

class DynaAgent:

    def __init__(self, exp_rate=0.7, lr=0.9, gamma = 0.9, max_epochs = 500, n_steps=5, episodes=1):
        self.env = gym.make('pathfinding-obstacle-25x25-v0')
        self.state = self.env.getPlayer() # (x,y)
        self.actions = [0,1,2,3] # 0 up, 1 down, 2 left, 3 right
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
                self.Q_values[(row, col)] = {}
                for a in self.actions:
                    self.Q_values[(row, col)][a] = 0
        
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
        self.state = self.env.getPlayer()
        self.state_actions = []

    def play(self):
        self.steps_per_episode = []
        self.cumulative_reward_per_episode = []
        self.reset()
        self.reset()
        for ep in range(self.episodes):
            epoches = 0
            cumulative_reward = 0
            while epoches < self.max_epochs:

                self.env.render()
                sleep(0.005)
                action = self.chooseAction()
                self.state_actions.append((self.state, action))

                nxtState2D, reward, self.env.game.terminal, _ = self.env.step(action)
                nxtState = self.env.getPlayer()

                cumulative_reward += reward

                # update Q-value    
                self.Q_values[self.state][action] += self.lr*(reward + self.gamma*np.max(list(self.Q_values[nxtState].values())) - self.Q_values[self.state][action])

                # update model
                if self.state not in self.model.keys():
                    self.model[self.state] = {}
                self.model[self.state][action] = (reward, nxtState)
                self.state = nxtState

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
            
                epoches +=1
                if epoches % 100 == 0:
                    print(f"Epoches: {epoches}")
                if(self.env.getTerminal()):
                    if(epoches < self.max_epochs):
                        self.max_epochs = epoches
                        self.convergence_episode = ep
                    self.success_episodes += 1
                    break

            # end of game
            if(self.exp_rate > 0.01):
                self.exp_rate = self.exp_rate*0.95
            if ep % 100 == 0:
                print("episode", ep)
            self.steps_per_episode.append(len(self.state_actions))
            self.cumulative_reward_per_episode.append(cumulative_reward)
            self.reset()

if __name__ == "__main__":
    N_EPISODES = 500
    # agent = DynaAgent(n_steps=0, episodes=N_EPISODES)
    # agent.play()

    # steps_episode_0 = agent.steps_per_episode
    # cumulative_r_0 = agent.cumulative_reward_per_episode

    agent = DynaAgent(n_steps=100, episodes=N_EPISODES)
    agent.play()

    steps_episode_100 = agent.steps_per_episode
    cumulative_r_100 = agent.cumulative_reward_per_episode

    # Save the Q-table in a text file
    with open(r'training.txt','w+') as f:
        f.write(str(agent.Q_values))

    noEpoches = agent.max_epochs
    conv = agent.convergence_episode
    succ = agent.success_episodes
    successRate = (succ / N_EPISODES)*100
    
    plt.figure(1)
    # plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
    plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.legend()
    
    plt.figure(2)
    # plt.plot(range(N_EPISODES), cumulative_r_0, label="step=0")
    plt.plot(range(N_EPISODES), cumulative_r_100, label="step=100")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")
    plt.legend()

    print(f"No. epoches: {noEpoches}")
    print(f"Succsess episodes: {succ}")
    print(f"Success Rate: {successRate}")
    print(f"Convergence speed After {conv} episodes")
    
    plt.show()
    