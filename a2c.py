from pfrl import q_functions, replay_buffers
from pfrl import utils, explorers
from pfrl.agents import A2C 
from torch import optim
import torch.nn as nn
import pfrl
from pfrl.policies import SoftmaxCategoricalHead
import torch

def a2c_agent(env, agent, args):
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)

    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # decoder + policy/value head
    model = nn.Sequential(
        nn.Linear(obs_space.shape[0], 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(50, action_space.n),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(50, 1),
        ),
    )

    if args.load:
        model = torch.load(f'{agent}.pt')

    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        model.parameters(),
        lr=args.lr,
        eps=args.rmsprop_epsilon,
        alpha=args.alpha,
    )

    agent = A2C(
        model,
        optimizer,
        gamma=args.gamma,
        gpu=args.gpu,
        num_processes=1,
        update_steps=args.update_steps,
        use_gae=args.use_gae,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
    )

    return agent
