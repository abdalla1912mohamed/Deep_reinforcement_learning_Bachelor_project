Abstract
Learning to navigate in an unknown environment is a substantial capability of mobile
robot. Conventional methods for robot navigation consists of three steps, involving localization, map building and path planning. However, such methods assume complete
knowledge of the environment dynamics, but in real-life applications the environment
parameters are usually stochastic and difficult to deduce so it becomes impractical to
rely on an explicit map of the environment. To adapt to map-less environments, learning
ability is a must to ensure obstacle avoidance and path planning flexibility. Recently, Reinforcement learning techniques were applied widely in the adaptive path planning for the
autonomous mobile robots. In this thesis, we propose different variations of Q-learning to
obtain an optimal navigation trajectory considering goal-oriented tasks of a skid-steering
mobile robot. Furthermore, we examine the performance of integrating neural network
approximators with Q-learning by applying deep Q Learning and comparing its results
with the standard Q-learning and dynamic Q-learning in maze-like environments. To
clarify, we simulate the paths taken by the skid-steering mobile robot in a grid world
environment containing static obstacles and a single static target. The goal of the robot
is to find the shortest path to the target while avoiding the obstacles without having any
access to the map. Deep Q learning, traditional Q-learning and Dyna-Q learning algorithms are implemented and the robot simulation parameters are captured and discussed
in this thesis.
