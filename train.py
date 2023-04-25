from time import gmtime, strftime, sleep

def train_agents_with_evaluation(
  agents,
  train_env,
  test_env,
  steps,
  train_steps,
  eval_interval,
  outpath,
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
            i * train_steps for i in range(len(test_rewards))
        ]

        with open(outpath, 'w') as f:
            f.write(str(test_reward_history))

    print('Will train for {} steps'.format(steps))
    print('Writing results to {}'.format(outpath))

    while total_steps < steps:
        iteration += 1
        total_steps += train_steps 
        #print('Main iter {} step {} / {}'.format(iteration, total_steps, steps))

        # eval interval is measured in training iterations, not steps
        if iteration % eval_interval == 0:
            # Evaluate the performance of each agent
            test_rewards.append(evaluate_agents(agents, test_env))

            print('Time {}, test reward {}'.format(
                strftime('%Y-%m-%d %H:%M:%S', gmtime()),
                test_rewards[-1]
            ))

            write_results()

        # Train the agents
        train_rewards.append(train_agents(agents, train_env, train_steps))

        #print('Time {}, train reward {}'.format(
            #strftime('%Y-%m-%d %H:%M:%S', gmtime()),
            #train_rewards[-1]
        #))

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

    if not train_env.agents:
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
        train_env.reset()

    return agent_rewards
