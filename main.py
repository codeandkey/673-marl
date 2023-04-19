import argparse
import sys

# PettingZoo imports
from pettingzoo.mpe import simple_adversary_v2, simple_push_v2

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
                    choices=['dqn', 'ddqn', 'a2c', 'ppo'],
                    help='The algorithm to use')

parser.add_argument('--env',
                    type=str,
                    default='simple_adversary',
                    choices=['simple_adversary', 'simple_push'],
                    help='The environment to use')

parser.add_argument('--episodes',
                    type=int,
                    default=1000,
                    help='The number of training episodes')

parser.add_argument('--outdir',
                    default='results',
                    help='Directory path to save output files.')

parser.add_argument('--steps',
                    type=int,
                    default=10**6,
                    help='Total number of timesteps to train the agent.')

parser.add_argument('--eval-interval',
                    type=int,
                    default=100,
                    help='Interval in timesteps between evaluations.')

args = parser.parse_args()

if __name__ == '__main__':
    print('Starting MARL expeirments')
    print('Algorithm: {}'.format(args.algorithm))
    print('Environment: {}'.format(args.env))

    # initialize experiment
    experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)

    # initialize training and eval environments
    if args.env == 'simple_adversary':
        train_env = simple_adversary_v2.env()
        test_env = simple_adversary_v2.env(render_mode='human')
    elif args.env == 'simple_push':
        train_env = simple_push_v2.env()
        test_env = simple_push_v2.env(render_mode='human')
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
            agent_name: dqn_agent(train_env, agent_name)
            for agent_name in train_env.agents
        }
    elif args.algorithm == 'dqn':
        from ddqn import ddqn_agent
        agent = ddqn_agent(env)
    elif args.algorithm == 'a2c':
        from a2c import a2c_agent
        agent = a3c_agent(env)
    elif args.algorithm == 'ppo':
        from ppo import ppo_agent
        agent = ppo_agent(env)
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
        args.steps,
        100,
        args.eval_interval,
        args.outdir
    )

    print('Finished MARL experiments')
