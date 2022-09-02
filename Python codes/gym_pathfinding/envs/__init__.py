from gym.envs.registration import register
import gym_pathfinding.envs.pathfinding_env
import gym_pathfinding.envs.partially_observable_env


for env_class in pathfinding_env.get_env_classes():
    register(
        id=env_class.id,
        entry_point='gym_pathfinding.envs.pathfinding_env:{name}'.format(name=env_class.__name__)
    )

for env_class in partially_observable_env.get_env_classes():
    register(
        id=env_class.id,
        entry_point='gym_pathfinding.envs.partially_observable_env:{name}'.format(name=env_class.__name__)
    )
