import numpy as np
from rl_agents.agent_runner import ENV_CLASS, CONFIG_NAME, RENDER_MODE, TRAIN_EPS, TRAIN_MAX_STEPS, TEST_MAX_STEPS
import json

class ContextualAgent:
    def __init__(self, n_states, n_arms, epsilon=1.0):
        self.n_states = n_states
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros((n_states, n_arms))
        self.counts = np.zeros((n_states, n_arms))
    
    def select(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values[state])
    
    def update(self, state, arm, reward):
        self.counts[state, arm] += 1
        self.q_values[state, arm] += (reward - self.q_values[state, arm]) / self.counts[state, arm]

class NodeAgent:
    def __init__(self, node_id, n_channels, max_backoff, n_agents, epsilon=1.0):
        self.node_id = node_id
        self.n_agents = n_agents
        self.queue_agent = ContextualAgent(4, 2, epsilon)
        self.channel_agent = ContextualAgent(2**n_channels, n_channels, epsilon)
        
        # Backoff agent state: 
        # 1. Packets in any queue (2 states)
        # 2. Transmitted last step (2 states)
        # 3. Channel collision history (2^n_channels states)
        # Total states: 2 * 2 * 2^n_channels
        n_backoff_states = 4 * (2**n_channels)
        self.backoff_agent = ContextualAgent(n_backoff_states, max_backoff + 1, epsilon)
    
    def set_epsilon(self, epsilon):
        self.queue_agent.epsilon = epsilon
        self.channel_agent.epsilon = epsilon
        self.backoff_agent.epsilon = epsilon
    
    def select_action(self, obs, n_channels):
        queue_state = self._get_queue_state(obs)
        channel_state = self._get_channel_state(obs, n_channels)
        backoff_state = self._get_backoff_state(obs, n_channels)
        
        return {
            'queue_selection': self.queue_agent.select(queue_state),
            'channel_selection': self.channel_agent.select(channel_state),
            'backoff_value': self.backoff_agent.select(backoff_state)
        }, (queue_state, channel_state, backoff_state)
    
    def update(self, states, action, reward):
        queue_state, channel_state, backoff_state = states
        self.queue_agent.update(queue_state, action['queue_selection'], reward)
        self.channel_agent.update(channel_state, action['channel_selection'], reward)
        self.backoff_agent.update(backoff_state, action['backoff_value'], reward)
    
    def _get_queue_state(self, obs):
        high_empty = 1 if obs[0] == 0 else 0
        low_empty = 1 if obs[1] == 0 else 0
        return high_empty * 2 + low_empty
    
    def _get_channel_state(self, obs, n_channels):
        collision_history = obs[-n_channels:]
        state = 0
        for i, val in enumerate(collision_history):
            if val > 0.5:
                state += 2**i
        return state
    
    def _get_backoff_state(self, obs, n_channels):
        # 1. Whether there are packets in any of the two queues combined
        has_packets = 1 if (obs[0] > 0 or obs[1] > 0) else 0
        
        # 2. Whether the agent transmitted last step
        tx_last = int(obs[4])
        
        # 3. The recent collision history of the channels
        collision_state = self._get_channel_state(obs, n_channels)
        
        # Combine states
        # collision_state: 0 to 2^N - 1
        # tx_last: 0 or 1
        # has_packets: 0 or 1
        
        state = (has_packets * 2 * (2**n_channels)) + (tx_last * (2**n_channels)) + collision_state
        return state


# Training
env = ENV_CLASS(config_name=CONFIG_NAME)
if TRAIN_MAX_STEPS is not None:
    env.max_steps = TRAIN_MAX_STEPS
observations, infos = env.reset()

n_episodes = TRAIN_EPS
epsilon_start = 1.0
epsilon_end = 0.01

agents = {agent: NodeAgent(i, env.n_channels, env.max_backoff, len(env.possible_agents), epsilon_start) 
          for i, agent in enumerate(env.possible_agents)}

for episode in range(n_episodes):
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes
    for agent in agents.values():
        agent.set_epsilon(epsilon)
    
    episode_rewards = {agent: 0 for agent in env.possible_agents}
    observations, infos = env.reset()
    
    last_states = {agent: None for agent in env.agents}
    last_actions = {agent: None for agent in env.agents}
    
    while True:
        actions = {}
        
        for agent_name in env.agents:
            if infos[agent_name]['active_decision']:
                action, state = agents[agent_name].select_action(observations[agent_name], env.n_channels)
                actions[agent_name] = action
                last_states[agent_name] = state
                last_actions[agent_name] = action
            else:
                actions[agent_name] = {
                    'queue_selection': 0,
                    'channel_selection': 0,
                    'backoff_value': 0
                }
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent_name in env.agents:
            episode_rewards[agent_name] += rewards[agent_name]
            
            if infos[agent_name]['active_decision'] and last_states[agent_name] is not None:
                agents[agent_name].update(last_states[agent_name], last_actions[agent_name], rewards[agent_name])
        
        if all(truncations.values()) or all(terminations.values()):
            print(f"Episode {episode} rewards: {episode_rewards}")
            break

    # if episode_rewards[env.agents[0]] > 0 and episode_rewards[env.agents[1]] > 0:
    #     break

# Testing
env = ENV_CLASS(config_name=CONFIG_NAME, render_mode=RENDER_MODE)
if TEST_MAX_STEPS is not None:
    env.max_steps = TEST_MAX_STEPS

for agent in agents.values():
    agent.set_epsilon(0.0)

observations, infos = env.reset()

while True:
    actions = {}
    for agent_name in env.agents:
        if infos[agent_name]['active_decision']:
            action, _ = agents[agent_name].select_action(observations[agent_name], env.n_channels)
            actions[agent_name] = action
        else:
            actions[agent_name] = {
                'queue_selection': 0,
                'channel_selection': 0,
                'backoff_value': 0
            }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if all(truncations.values()) or all(terminations.values()):
        break

print("Throughput and latency metrics:")
print(json.dumps(infos[env.agents[0]]['node_latency_throughput'], indent=2))

