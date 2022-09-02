import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import SimpleRNN
from keras.optimizers import Adam
import gym
import gym_pathfinding
import matplotlib.pyplot as plt
from time import sleep

EPISODES = 2500

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = list()
        self.max_mem = 20000
        self.gamma = 0.98   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.replay_cnt = 0 
        self.actions = [0,1,2,3]

        self.target_model = self._build_model()
        self.model = self._build_model()
        self.update_model = 10

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(625, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if(len(self.memory)>self.max_mem) :
            self.memory.pop(0)

    def act(self, state):
        if np.random.uniform(0,1) <= self.epsilon:
            return random.choice(range(self.action_size) )
        act_values = self.model.predict(state)[0]
        #print(self.model.predict(state))
        return np.argmax(act_values)  # returns action

    def replay(self, batch_size):
        self.replay_cnt+=1 

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # q value of the state  if it is the final 
            if not done:
                target = (reward + self.gamma *
                          np.max(self.target_model.predict(next_state)[0])) # q value of state is computed by bellman
            target_f = self.target_model.predict(state) # value of the current state q_value 
            target_f[0][action] = target #
            self.model.fit(state, target_f, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if (self.replay_cnt % agent.update_model ==0) :
                agent.target_train() 

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def load(self, name):   
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def reset(self):
        self.env.reset()
        self.env.seed(1)
        # self.env.game.player = self.randomStart()
        self.state = self.env.getPlayer()
        self.state_actions = []

    # def play(self):


if __name__ == "__main__":
    env = gym.make('pathfinding-obstacle-25x25-v0')
    steps_per_episode = []
    cumulative_reward_per_episode = []
    env.reset()
    env.reset()

    state_size= 625

    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 48

    print("START DQN")
    donecount=0 
    for e in range(EPISODES):

        reward_sum = 0

        state = env.reset()

        env.seed(1)
        state = [[0]*state_size]*1
        state[0][env.getPlayer()[0]*25 + env.getPlayer()[1]] = 1

        state = np.reshape(state, [1, state_size])

        steps = 0
        while steps < 2000:

            #env.render()
            #sleep(0.005)

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            next_state = [[0]*state_size]*1
            next_state[0][env.getPlayer()[0]*25 + env.getPlayer()[1]] = 1

            next_state = np.reshape(next_state, [1, state_size])
            if(done == True):
                donecount+=1
            reward_sum += reward
            
               # print("memory length", len(agent.memory))
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break
            steps += 1
            
        print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                  .format(e, EPISODES, reward_sum, agent.epsilon, steps))
        
        if len(agent.memory) > batch_size & donecount>0:
                agent.replay(batch_size)
        #if e % 1000 == 0:
        #     agent.save("./save/dqn" + str(e) + ".h5")
        steps_per_episode.append(steps)
        cumulative_reward_per_episode.append(reward_sum)
    
    plt.figure(1)
    plt.plot(range(EPISODES), steps_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Steps per episode")
    plt.ylim(0, 500)

    plt.figure(2)
    plt.plot(range(EPISODES), cumulative_reward_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward")

    plt.show()