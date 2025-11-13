import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from wn_env import WirelessNetworkEnv
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

def get_queue_state(obs):
    high_empty = 1 if obs[0] == 0 else 0
    low_empty = 1 if obs[1] == 0 else 0
    return high_empty * 2 + low_empty

def get_channel_state(obs, n_channels):
    collision_history = obs[-n_channels:]
    state = 0
    for i, val in enumerate(collision_history):
        if val > 0.5:
            state += 2**i
    return state

def get_backoff_state(obs):
    return 0 if obs[2] == 0 else 1


CONFIG_NAME = 'congested_network'

# Training
env = WirelessNetworkEnv(config_name=CONFIG_NAME, max_steps=500)
obs, info = env.reset()

n_episodes = 200
epsilon_start = 1.0
epsilon_end = 0.01

queue_agent = ContextualAgent(4, 2, epsilon_start)
channel_agent = ContextualAgent(2**env.n_channels, env.n_channels, epsilon_start)
backoff_agent = ContextualAgent(2, env.max_backoff + 1, epsilon_start)

for episode in range(n_episodes):
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes
    queue_agent.epsilon = epsilon
    channel_agent.epsilon = epsilon
    backoff_agent.epsilon = epsilon
    episode_rewards = 0
    
    obs, info = env.reset()
    
    while True:
        queue_state = get_queue_state(obs)
        channel_state = get_channel_state(obs, env.n_channels)
        backoff_state = get_backoff_state(obs)
        
        queue_sel = queue_agent.select(queue_state)
        channel_sel = channel_agent.select(channel_state)
        backoff_sel = backoff_agent.select(backoff_state)
        
        action = {
            'queue_selection': queue_sel,
            'channel_selection': channel_sel,
            'backoff_value': backoff_sel
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards += reward
        
        queue_agent.update(queue_state, queue_sel, reward)
        channel_agent.update(channel_state, channel_sel, reward)
        backoff_agent.update(backoff_state, backoff_sel, reward)
        
        if terminated or truncated:
            print(f"Episode {episode} reward: {episode_rewards}")
            break

# Testing
env = WirelessNetworkEnv(config_name=CONFIG_NAME, max_steps=100, render_mode='human')
queue_agent.epsilon = 0.0
channel_agent.epsilon = 0.0
backoff_agent.epsilon = 0.0

obs, info = env.reset()

while True:
    queue_state = get_queue_state(obs)
    channel_state = get_channel_state(obs, env.n_channels)
    backoff_state = get_backoff_state(obs)
    
    action = {
        'queue_selection': queue_agent.select(queue_state),
        'channel_selection': channel_agent.select(channel_state),
        'backoff_value': backoff_agent.select(backoff_state)
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

print("Throughput and latency metrics:")
print(json.dumps(info['node_latency_throughput'], indent=2))