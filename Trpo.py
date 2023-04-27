from pfrl import q_functions, replay_buffers
from pfrl import utils, explorers
from pfrl.agents import DQN
from torch import optim

import pfrl
import torch
from torch import nn
import gym
import numpy as np

def trpo_agent(env, agent):
    obs_size = env.observation_space(agent).shape[0]
    n_actions = env.action_space(agent).n
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, n_actions),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )
    
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1e-2)
    
    vf_opt = torch.optim.Adam(vf.parameters())
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = pfrl.agents.TRPO(
        policy=policy,
        vf=vf,
        phi=phi,
        vf_optimizer=vf_opt,
        obs_normalizer=None,
        gpu=None,
        update_interval=5000,
        max_kl=0.01,
        conjugate_gradient_max_iter=20,
        conjugate_gradient_damping=1e-1,
        gamma=0.995,
        lambd=0.97,
        vf_epochs=10,
        entropy_coef=0,
    )
    return agent