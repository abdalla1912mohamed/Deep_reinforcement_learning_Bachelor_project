from setuptools import setup

setup(name='gym_pathfinding',
      description='Gym environnement for pathfinding',
      version='0.0.1',
      install_requires=['gym==0.9.7', 'cython', 'numpy', 'scipy', 'pygame'],
      packages=['gym_pathfinding', 'gym_pathfinding.envs', 'gym_pathfinding.games'],

      author='Adrien Turiot',
      url='https://github.com/DidiBear/gym-pathfinding'
)
