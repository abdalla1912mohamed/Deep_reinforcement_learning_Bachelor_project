README: Navigation in Unknown Environments for Mobile Robots
Abstract
Learning to navigate in an unknown environment is a crucial capability for mobile robots. Traditional navigation methods typically consist of three steps: localization, map building, and path planning. However, these approaches assume complete knowledge of environmental dynamics, which is often unrealistic, as real-life environments are usually stochastic and difficult to model explicitly. Therefore, relying on a predefined map becomes impractical.

To effectively adapt to map-less environments, robots must possess learning capabilities to ensure obstacle avoidance and flexible path planning. Recently, reinforcement learning techniques have been widely applied in adaptive path planning for autonomous mobile robots. This thesis proposes various adaptations of Q-learning to derive optimal navigation trajectories for a skid-steering mobile robot focused on goal-oriented tasks.

We further explore the integration of neural network approximators with Q-learning by implementing deep Q-learning and comparing its performance against standard Q-learning and dynamic Q-learning in maze-like environments. Our simulations depict the paths taken by the skid-steering mobile robot in a grid world containing static obstacles and a single target. The objective is to navigate toward the target while avoiding obstacles, all without access to a map.

The thesis presents the implementation of deep Q-learning, traditional Q-learning, and Dyna-Q algorithms, with a thorough discussion of the robot simulation parameters and the results obtained.

Key Components
Objective: Optimize navigation trajectories for skid-steering mobile robots using reinforcement learning techniques.
Methods: Comparison of Q-learning variations, including deep Q-learning, traditional Q-learning, and Dyna-Q.
Environment: Grid world simulations with static obstacles and targets.
Results: Performance analysis of the different algorithms in terms of efficiency and path optimization.
Usage
To replicate the experiments and simulations discussed in this thesis, ensure you have the necessary libraries and frameworks installed. Follow the provided instructions to set up the environment and run the simulations.

Conclusion
This work highlights the potential of reinforcement learning in enabling mobile robots to navigate unknown environments effectively, emphasizing the importance of adaptability and learning in real-world applications.
