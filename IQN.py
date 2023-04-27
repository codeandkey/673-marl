import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"
from pfrl import agents, q_functions, replay_buffers, explorers, action_value

import gym  # NOQA:E402
import gym.wrappers  # NOQA:E402
import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402
import torch
import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import acer  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402
from pfrl.q_functions import DiscreteActionValueHead  # NOQA:E402
from pfrl.replay_buffers import EpisodicReplayBuffer  # NOQA:E402
from pfrl.wrappers import atari_wrappers 

def IQN_agent(env, agent, steps):

    obs_size = env.observation_space(agent).shape[0]
    n_actions = env.action_space(agent).n

    # Hyperparameter; the continuous range of possible Q-values is discretized into a fixed number of "atoms"
    # n_atoms = 51 
    # Hyperparameter; Lower and upper bounds of the support for the Q-value distribution. They have a direct impact on the accuracy of the learned distribution.
    # v_max = 10 
    # v_min = -10
    # q_func = q_functions.DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max)
    # model = nn.Sequential(
    #     nn.Linear(obs_size, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, 50),
    #     nn.ReLU(),
    #     pfrl.nn.Branched(
    #         nn.Sequential(
    #             nn.Linear(50, n_actions),
    #             SoftmaxCategoricalHead(),
    #         ),
    #         nn.Linear(50, 1),
    #     )
    #     )
 
    q_func = pfrl.agents.iqn.ImplicitQuantileQFunction(
        psi=nn.Sequential(
            nn.Linear(obs_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Flatten(),
        ),
        phi=nn.Sequential(
            pfrl.agents.iqn.CosineBasisLinear(50, 50),
            nn.ReLU(),
        ),
        f=nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        ),
    )

    # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
    opt = torch.optim.Adam(q_func.parameters(), lr=5e-5, eps=1e-2 / 32)

    rbuf = replay_buffers.ReplayBuffer(10**6)
    #rbuf = replay_buffers.PrioritizedReplayBuffer(10**6)

    explorer = explorers.Greedy()

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = pfrl.agents.IQN(
        q_func,
        opt,
        rbuf,
        gpu=-2,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=100,
        update_interval=100,
        phi=phi,
    )
    return agent