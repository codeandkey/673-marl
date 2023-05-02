from time import gmtime, strftime, sleep

def train_agents_with_evaluation(
  agents,
  train_env,
  test_env,
  args,
):
    """Trains agents in a multi-agent environment, periodically evaluating the
       performance of each agent over a single episode."""

    print('Agent set: ', agents.keys())
    train_reward_history = { a: [] for a in agents.keys() }

    """def write_results():
        test_reward_history = { a: [] for a in agents.keys() }

        for t in test_rewards:
            for k in t.keys():
                test_reward_history[k].append(t[k])

        test_reward_history['episodes'] = [
            i * args.eval_interval for i in range(len(test_rewards))
        ]

        with open(args.outpath, 'w') as f:
            f.write(str(test_reward_history))"""

    print('Training for {} episodes'.format(args.episodes))
    print('Writing results to {}'.format(args.outpath))

    for episode in range(args.episodes):
        #print('Main iter {} step {} / {}'.format(iteration, total_steps, steps))

        # eval interval is measured in training iterations, not steps

        train_rewards = train_agents(agents, train_env)

        # record the training rewards for each agent
        for k in train_rewards.keys():
            train_reward_history[k].append(train_rewards[k])

        mean_rewards = {
            k: sum(train_reward_history[k]) / max(1, len(train_reward_history[k]))
            for k in train_reward_history.keys()
        }

        print('Episode', episode, '/', args.episodes,
              'train reward', str(list(map(int, train_rewards.values()))),
              'mean reward', str(list(map(int, mean_rewards.values()))),
        )

        # Train the agents
        #train_rewards.append(train_agents(agents, train_env, train_steps))

        #print('Time {}, episode {}, train reward {}'.format(
        #    episode,
        #    strftime('%Y-%m-%d %H:%M:%S', gmtime()),
        #    train_reward
        #))

        """
        if episode % args.eval_interval == 0:
            # Evaluate the performance of each agent
            test_reward = evaluate_agents(agents, test_env)
            test_rewards.append(test_reward)

            print('Time {}, episode {}, test reward {}'.format(
                strftime('%Y-%m-%d %H:%M:%S', gmtime()),
                episode,
                test_reward,
            ))"""

        # we report training rewards directly instead of greedy evaluation,
        # to test the agents are learning and show a stabilized training curve
        with open(args.outpath, 'w') as f:
            f.write(str(train_reward_history | {'episodes': list(range(episode))}))

def evaluate_agents(pfrl_agents, test_env):
    """Evaluates the performance of each agent over a single episode."""
    
    test_env.reset()
    terminal = False
    trunc = False

    agent_rewards = {agent: 0 for agent in test_env.agents}

    for agent in test_env.agent_iter():
        if test_env.terminations[agent] or test_env.truncations[agent]:
            test_env.step(None)
            continue

        state, reward, terminal, trunc, _ = test_env.last()

        with pfrl_agents[agent].eval_mode():
            action = pfrl_agents[agent].act(state)

        agent_rewards[agent] += reward
        test_env.step(action)

        #print('Observe', agent, test_env.observe(agent), reward, terminal, trunc)
        pfrl_agents[agent].observe(
            test_env.observe(agent),
            reward,
            terminal,
            terminal or trunc
        )

        sleep (1 / 60)

    return agent_rewards

def train_agents(agents, train_env):
    """Trains the agents in a multi-agent environment for a given number of
       steps."""

    agent_rewards = {agent: 0 for agent in agents.keys()}

    train_env.reset()

    first_action = { agent: True for agent in agents.keys() }

    for agent in train_env.agent_iter():
        state, reward, terminal, trunc, _ = train_env.last()
        agent_rewards[agent] += reward

        if not first_action[agent]:
            agents[agent].observe(
                state,
                reward,
                terminal or trunc,
                terminal or trunc
            )
        else:
            first_action[agent] = False

        if terminal or trunc:
            action = None
        else:
            action = agents[agent].act(state)

        train_env.step(action)

        if state.shape != train_env.observation_space(agent).shape:
            raise ValueError('Observation shape {} does not match expected shape {}'.format(
                state.shape, train_env.observation_space(agent).shape
            ))

    #print('episode done')

    return agent_rewards
