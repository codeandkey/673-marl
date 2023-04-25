from pfrl import q_functions, replay_buffers
from pfrl import utils, explorers
from pfrl.agents import DQN
from torch import optim

# DQN agent, example at
# https://github.com/pfnet/pfrl/blob/2ad3d51a7a971f3fe7f2711f024be11642990d61/examples/gym/train_dqn_gym.py#L175

def dqn_agent(env, agent):
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)

    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # initialize Q function
    q_function = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_space.shape[0],
        action_space.n,
        n_hidden_layers=2,
        n_hidden_channels=50,
    )

    # epsilon-greedy exploration
    explorer = explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=action_space.sample
    )

    # DQN agent
    agent = DQN(
        q_function,
        optimizer=optim.Adam(q_function.parameters(), eps=1e-2),
        replay_buffer=replay_buffers.ReplayBuffer(10 ** 6),
        gamma=0.99,
        explorer=explorer,
        gpu=-1,
        update_interval=1,
        target_update_interval=100,
    )

    return agent
