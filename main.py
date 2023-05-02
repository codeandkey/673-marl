import argparse
import sys

# PettingZoo imports
from pettingzoo.mpe import simple_adversary_v2, simple_push_v2, simple_spread_v2

# PFRL imports
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers
from pfrl import utils

from train import train_agents_with_evaluation

# Parse command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--algorithm',
                    type=str,
                    default='dqn',
                    choices=['dqn', 'ddqn', 'a2c', 'ppo', 'rainbow'],
                    help='The algorithm to use')

parser.add_argument('--env',
                    type=str,
                    default='simple_spread',
                    choices=['simple_adversary', 'simple_push', 'simple_spread'],
                    help='The environment to use')

parser.add_argument('--tau',
                    type=float,
                    default=1.0,
                    help='Soft update coefficient (0, 1]')

parser.add_argument('--lr',
                    type=float,
                    default=0.00001,
                    help='Learning rate')

parser.add_argument('--gamma',
                    type=float,
                    default=0.99,
                    help='Discount factor')

parser.add_argument('--rmsprop_epsilon',
                    type=float,
                    default=1e-5,
                    help='RMSprop optimizer epsilon')

parser.add_argument('--alpha',
                    type=float,
                    default=0.99,
                    help='RMSprop optimizer alpha')

parser.add_argument('--use-gae',
                    action='store_true',
                    help='Use generalized advantage estimation')

parser.add_argument('--gpu',
                    type=int,
                    default=-1,
                    help='GPU device ID. Set to -1 to use CPUs only.')

parser.add_argument('--update-steps',
                    type=int,
                    default=5,
                    help='Frequency of target network updates')

parser.add_argument('--max-grad-norm',
                    type=float,
                    default=40.0,
                    help='Maximum norm of gradients')

parser.add_argument('--episodes',
                    type=int,
                    default=3000000,
                    help='The number of training episodes')

parser.add_argument('--outpath',
                    default='results.json',
                    help='Output file path.')

parser.add_argument('--eval-interval',
                    type=int,
                    default=250,
                    help='Interval in timesteps between evaluations.')

parser.add_argument('--load',
                    default=False,
                    action='store_true',
                    help='Load agent models from disk')

parser.add_argument('--render-mode',
                    default=None,
                    help='Test environment render mode')

parser.add_argument('--n-agents',
                    default=1,
                    type=int,
                    help='Number of agents')

args = parser.parse_args()

if __name__ == '__main__':
    print('Starting MARL expeirments')
    print('Algorithm: {}'.format(args.algorithm))
    print('Environment: {}'.format(args.env))

    # initialize training and eval environments
    if args.env == 'simple_adversary':
        train_env = simple_adversary_v2.env()
        test_env = simple_adversary_v2.env(render_mode=args.render_mode)
    elif args.env == 'simple_push':
        train_env = simple_push_v2.env()
        test_env = simple_push_v2.env()
    elif args.env == 'simple_spread':
        train_env = simple_spread_v2.env(N=args.n_agents, local_ratio=0, max_cycles=200)
        test_env = simple_spread_v2.env(N=args.n_agents, local_ratio=0, max_cycles=200)
    else:
        raise NotImplementedError(args.env)

    train_env.reset()
    test_env.reset()

    # initialize an algorithm
    # each environment makes use of two different agents,
    # so we initialize one for each.

    if args.algorithm == 'dqn':
        from dqn import dqn_agent

        agents = {
            agent_name: dqn_agent(train_env, agent_name, args)
            for agent_name in train_env.agents
        }
    elif args.algorithm == 'rainbow':
        from rainbow import rainbow_agent

        agents = {
            agent_name: rainbow_agent(train_env, agent_name, args)
            for agent_name in train_env.agents
        }
    elif args.algorithm == 'a2c':
        from a2c import a2c_agent

        agents = {
            agent_name: a2c_agent(train_env, agent_name, args)
            for agent_name in train_env.agents
        }
    else:
        raise NotImplementedError(args.algorithm)

    print('Starting MARL experiments')

    # we can't use the built in experiment runners unfortunately,
    # as they are too closely tied to single-agent gym environments

    # we reimplement the train_agent_with_evaluation method seen in the original
    # pfrl/experiments/train_agent.py, with changes to evaluation strategy
    # to work with multi-agent environments

    train_agents_with_evaluation(
        agents,
        train_env,
        test_env,
        args,
    )

    print('Finished MARL experiments')
