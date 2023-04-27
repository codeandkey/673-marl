import numpy as np

from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from pfrl import nn as pnn
from pfrl import agents, q_functions, replay_buffers, explorers, action_value


def rainbow_agent(env, agent, steps):

    obs_size = env.observation_space(agent).shape[0]
    n_actions = env.action_space(agent).n

    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size,
        n_actions,
        n_hidden_layers=2,
        n_hidden_channels=50,
    )
    # q_func = QNetwork(obs_size, n_actions)

    # Hyperparameter; the continuous range of possible Q-values is discretized into a fixed number of "atoms"
    # n_atoms = 51 
    # Hyperparameter; Lower and upper bounds of the support for the Q-value distribution. They have a direct impact on the accuracy of the learned distribution.
    # v_max = 10 
    # v_min = -10
    # q_func = q_functions.DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max)


    # Noisy nets; Sigma is the scaling factor of the initial weights of noise-scaling parameters
    pnn.to_factorized_noisy(q_func, sigma_scale=0.5)

    # Turn off explorer
    explorer = explorers.Greedy()

    # Optimizer; Use the same hyper parameters as https://arxiv.org/abs/1710.02298
    opt = optim.Adam(q_func.parameters(), eps=1e-2)

    # Prioritized Replay
    # Anneal beta from beta0 to 1 throughout training
    update_interval = 4
    betasteps = steps / update_interval

    # Capacity in terms of number of transitions, Exponent of errors to compute probabilities to sample
    # Initial value of beta, Steps to anneal beta to 1 
    # Divide by the maximum weight in the memory to normalize weights
    rbuf = replay_buffers.PrioritizedReplayBuffer(10**6)

    # Feature extractor
    def phi(x):
        return np.asarray(x, dtype=np.float32) / 255

    Agent = agents.DoubleDQN

    # target_update_interval: Determines how often the target Q-network is updated with the weights of the online Q-network
    # update_interval: This parameter determines how often the online Q-network is updated with the experience collected in the replay buffer
    agent = Agent(q_func, opt, rbuf, gpu=-1, gamma=0.99, explorer=explorer, target_update_interval=100, update_interval=1, phi=phi)

    return agent
