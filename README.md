# RL_Library

Nowadays, artificial intelligence is covers an important role in industry and
scientific research. Next to clustering, deep learning and neural networks;
reinforcement learning is becoming more and more popular. In the present
work, the performance of reinforcement learning algorithms has been tested.
Further more, two types of results have been gathered:
- A solo-agent version, in which algorithms are executed as usual in the
given environment.
- A cooperative version, in which two or more algorithms work together
in order to take decisions.

## Analysed algorithms

- Q-Learning
- SARSA
- DQN/DDQN
- AC (not fully tested)

## Ensembling strategies

- Major voting based
- Rank voting based
- Trust based

## OpenAI Gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning
algorithms written in python. It provides a set of environments ranging
from simple textual games to emulated Atari games and physics problems.
Each environment is shipped with a set of possible actions/moves with a
related reward. The user has the possibility to obtain a standardised set of
environments in order to feed the reinforcement learning algorithm. Moreover,
an optional rendering is provided in order to offer a clear view of what
is happening in background. There are different types of environments, characterised
by different features such as:
- Observation space domain: discrete or continuous.
- Observation state type: memory representation or video frame.
- Reward range: finite or infinite set of values.
- Steps limitation.
- Maximum number of trials.

### Testing environments
- Frozen-Lake4x4
- Frozen-Lake8x8
- Taxi
- MountainCar
- Breakout (not fully tested)
- Pong (not fully tested)
- CartPole
