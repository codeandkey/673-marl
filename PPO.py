from pfrl import q_functions, replay_buffers
from pfrl import utils, explorers
from pfrl.agents import DQN
from torch import optim

# DQN agent, example at
# https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/examples/gym/train_dqn_gym.py#L175

import argparse
import functools

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO

def ppo_agent(env, agent):
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)

    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # initialize Q function
    policy = torch.nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n),
            pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_space,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

    vf = torch.nn.Sequential(
        nn.Linear(obs_space.shape[0], 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    # epsilon-greedy exploration
    explorer = explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=action_space.sample
    )

    # DQN agent
    model = pfrl.nn.Branched(policy, vf)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )
    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=-1,
        update_interval=1,
        minibatch_size=128,
        epochs=10,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.99,
        lambd=0.97,
    )

    return agent
