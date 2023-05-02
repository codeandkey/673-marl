from time import gmtime, strftime, sleep

def train_agents_with_evaluation(
  agents,
  train_env,
  test_env,
  args,
):
    """Trains agents in a multi-agent environment, periodically evaluating the
       performance of each agent over a single episode."""

    train_rewards = [] # might not need recording, can remove if expensive
    test_rewards = []

    total_steps = 0 # the total number of training steps taken
    iteration = 0   # the number of training iterations taken

    def write_results():
        test_reward_history = { a: [] for a in agents.keys() }

        for t in test_rewards:
            for k in t.keys():
                test_reward_history[k].append(t[k])

        test_reward_history['episodes'] = [
            i * args.eval_interval for i in range(len(test_rewards))
        ]

        with open(args.outpath, 'w') as f:
            f.write(str(test_reward_history))

    print('Will train for {} episodes'.format(args.episodes))
    print('Writing results to {}'.format(args.outpath))

    for episode in range(args.episodes):
        #print('Main iter {} step {} / {}'.format(iteration, total_steps, steps))

        # eval interval is measured in training iterations, not steps

        train_reward = train_agents(agents, train_env, args.steps)
        train_rewards.append(train_reward)

        # Train the agents
        #train_rewards.append(train_agents(agents, train_env, train_steps))

        #print('Time {}, episode {}, train reward {}'.format(
        #    episode,
        #    strftime('%Y-%m-%d %H:%M:%S', gmtime()),
        #    train_reward
        #))

        if episode % args.eval_interval == 0:
            # Evaluate the performance of each agent
            test_reward = evaluate_agents(agents, test_env)
            test_rewards.append(test_reward)

            print('Time {}, episode {}, test reward {}'.format(
                strftime('%Y-%m-%d %H:%M:%S', gmtime()),
                episode,
                test_reward,
            ))

            write_results()

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

def train_agents(agents, train_env, steps):
    """Trains the agents in a multi-agent environment for a given number of
       steps."""

    agent_rewards = {agent: 0 for agent in agents.keys()}

    train_env.reset()

    for agent in train_env.agent_iter(max_iter=steps):
        if train_env.terminations[agent] or train_env.truncations[agent]:
            train_env.step(None)
            continue 

        #print('Agent', agent)
        # take environment observation
        state, reward, terminal, trunc, _ = train_env.last()
        agent_rewards[agent] += reward
        action = agents[agent].act(state)
        train_env.step(action)

        #print('Observe', agent, train_env.observe(agent), reward, terminal, trunc)
        agents[agent].observe(
            train_env.observe(agent),
            reward,
            terminal,
            terminal or trunc
        )

        if state.shape != train_env.observation_space(agent).shape:
            raise ValueError('Observation shape {} does not match expected shape {}'.format(
                state.shape, train_env.observation_space(agent).shape
            ))

        if not train_env.agents:
            break

    return agent_rewards
