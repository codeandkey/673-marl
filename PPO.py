import numpy as np

from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from pfrl import nn as pnn
from pfrl import agents, q_functions, replay_buffers, explorers, action_value
from pfrl.policies import SoftmaxCategoricalHead
import pfrl


def PPO_agent(env, agent, steps):

    obs_size = env.observation_space(agent).shape[0]
    n_actions = env.action_space(agent).n

    model = nn.Sequential(
        nn.Linear(obs_size, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(50, n_actions,
                SoftmaxCategoricalHead(),
            ),)),
            nn.Linear(50, 1),
        )
 


    # vf = torch.nn.Sequential(
    #     nn.Linear(obs_size, 64),
    #     nn.Tanh(),
    #     nn.Linear(64, 64),
    #     nn.Tanh(),
    #     nn.Linear(64, 1),
    # )
    # model = pfrl.nn.Branched( model, vf)
    # Noisy nets; Sigma is the scaling factor of the initial weights of noise-scaling parameters


    # Optimizer; Use the same hyper parameters as https://arxiv.org/abs/1710.02298
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

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

    Agent = agents.PPO
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_size, clip_threshold=5
    )
    # target_update_interval: Determines how often the target Q-network is updated with the weights of the online Q-network
    # update_interval: This parameter determines how often the online Q-network is updated with the experience collected in the replay buffer
    #agent = Agent(q_func, opt, rbuf, gpu=-1, gamma=0.99, explorer=explorer, target_update_interval=100, update_interval=1, phi=phi)
    agent = Agent(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=-1,
        update_interval=100,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )

    return agent