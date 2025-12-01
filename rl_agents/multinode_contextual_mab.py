import numpy as np
from envs import WirelessNetworkParallelEnv
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
    def __init__(self, n_channels, max_backoff, epsilon=1.0):
        self.queue_agent = ContextualAgent(4, 2, epsilon)
        self.channel_agent = ContextualAgent(2**n_channels, n_channels, epsilon)
        self.backoff_agent = ContextualAgent(4, 2, epsilon)
    
    def set_epsilon(self, epsilon):
        self.queue_agent.epsilon = epsilon
        self.channel_agent.epsilon = epsilon
        self.backoff_agent.epsilon = epsilon
    
    def select_action(self, obs, n_channels):
        queue_state = self._get_queue_state(obs)
        channel_state = self._get_channel_state(obs, n_channels)
        backoff_state = self._get_backoff_state(obs)
        
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
    
    def _get_backoff_state(self, obs):
        backoff_binary = 0 if obs[2] == 0 else 1
        timestep_binary = int(obs[3]) % 2
        return backoff_binary * 2 + timestep_binary


CONFIG_NAME = 'multi_node_competitive'

# Training
env = WirelessNetworkParallelEnv(config_name=CONFIG_NAME)
observations, infos = env.reset()

n_episodes = 300
epsilon_start = 1.0
epsilon_end = 0.01

agents = {agent: NodeAgent(env.n_channels, env.max_backoff, epsilon_start) 
          for agent in env.possible_agents}

for episode in range(n_episodes):
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes
    for agent in agents.values():
        agent.set_epsilon(epsilon)
    
    episode_rewards = {agent: 0 for agent in env.possible_agents}
    observations, infos = env.reset()
    
    while True:
        actions = {}
        states = {}
        for agent_name in env.agents:
            action, state = agents[agent_name].select_action(observations[agent_name], env.n_channels)
            actions[agent_name] = action
            states[agent_name] = state
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        for agent_name in env.agents:
            episode_rewards[agent_name] += rewards[agent_name]
            agents[agent_name].update(states[agent_name], actions[agent_name], rewards[agent_name])
        
        if all(truncations.values()) or all(terminations.values()):
            print(f"Episode {episode} rewards: {episode_rewards}")
            break

    # if episode_rewards[env.agents[0]] > 0 and episode_rewards[env.agents[1]] > 0:
    #     break

# Testing
env = WirelessNetworkParallelEnv(config_name=CONFIG_NAME, render_mode='human')
env.max_steps = 100

for agent in agents.values():
    agent.set_epsilon(0.0)

observations, infos = env.reset()

while True:
    actions = {}
    for agent_name in env.agents:
        action, _ = agents[agent_name].select_action(observations[agent_name], env.n_channels)
        actions[agent_name] = action
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if all(truncations.values()) or all(terminations.values()):
        break

print("Throughput and latency metrics:")
print(json.dumps(infos[env.agents[0]]['node_latency_throughput'], indent=2))

