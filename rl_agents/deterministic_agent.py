from rl_agents.agent_runner import ENV_CLASS, CONFIG_NAME, RENDER_MODE, TEST_MAX_STEPS

# Configuration
# Using TEST_MAX_STEPS for the deterministic run as it's likely for evaluation
MAX_STEPS = TEST_MAX_STEPS if TEST_MAX_STEPS is not None else 100
N_EPISODES = 1  # Deterministic agent only needs one test episode

env = ENV_CLASS(config_name=CONFIG_NAME, render_mode=RENDER_MODE)
env.max_steps = MAX_STEPS

total_agents = env.n_nodes
print(f"Running Deterministic Agent with {total_agents} nodes for {N_EPISODES} episodes.")

for episode in range(N_EPISODES):
    obs, infos = env.reset()
    
    # Agent state to track alternation
    # 0: Action 1 (Backoff 0)
    # 1: Action 2 (Backoff N-1)
    agent_states = {agent: 0 for agent in env.agents}
    
    episode_rewards = {agent: 0.0 for agent in env.agents}
    current_step = 0
    
    while True:
        actions = {}
        
        for agent in env.agents:
            # Parse agent index
            try:
                agent_idx = int(agent.split('_')[-1])
            except (ValueError, IndexError):
                agent_idx = 0

            # Check if agent can make a decision
            if infos[agent]['active_decision']:
                # Check if it's time to start (timestep == agent index)
                if current_step < agent_idx:
                    # Wait until the correct timestep
                    wait_time = agent_idx - current_step
                    # Cap at max_backoff
                    backoff_value = min(wait_time, env.max_backoff)
                    
                    actions[agent] = {
                        'queue_selection': 0,
                        'channel_selection': 0,
                        'backoff_value': backoff_value
                    }
                    continue

                # Get observation for queue lengths
                agent_obs = obs[agent]
                queue_high_len = agent_obs[0]
                queue_low_len = agent_obs[1]
                
                # Queue selection logic: High if available, else Low
                # 0 is High, 1 is Low (based on env constants)
                if queue_high_len > 0:
                    queue_selection = 0
                else:
                    queue_selection = 1
                    
                # Alternating logic
                if agent_states[agent] == 0:
                    # Action 1: Backoff 0
                    backoff_value = 0
                    # Switch state for next time
                    agent_states[agent] = 1
                else:
                    # Action 2: Backoff total_agents - 1
                    backoff_value = total_agents - 1
                    # Switch state for next time
                    agent_states[agent] = 0
                
                # Channel always 1
                actions[agent] = {
                    'queue_selection': queue_selection,
                    'channel_selection': 0, 
                    'backoff_value': backoff_value
                }
            else:
                # Agent is backing off, action doesn't matter much but must be valid
                actions[agent] = {
                    'queue_selection': 0,
                    'channel_selection': 0,
                    'backoff_value': 0
                }

        obs, rewards, terminations, truncations, infos = env.step(actions)
        current_step += 1
        
        for agent, reward in rewards.items():
            episode_rewards[agent] += reward
        
        if all(terminations.values()) or all(truncations.values()):
            print(f"Episode {episode} finished. Rewards: {episode_rewards}")
            break

env.close()
