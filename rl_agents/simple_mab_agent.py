import numpy as np
from envs import WirelessNetworkEnv

class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=1.0):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
    
    def select(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]

# Training
env = WirelessNetworkEnv(config_name='simple_network', max_steps=500)
obs, info = env.reset()

n_episodes = 100
epsilon_start = 1.0
epsilon_end = 0.01

queue_agent = EpsilonGreedyAgent(2, epsilon_start)
channel_agent = EpsilonGreedyAgent(env.n_channels, epsilon_start)
backoff_agent = EpsilonGreedyAgent(env.max_backoff + 1, epsilon_start)

for episode in range(n_episodes):
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / n_episodes
    queue_agent.epsilon = epsilon
    channel_agent.epsilon = epsilon
    backoff_agent.epsilon = epsilon
    episode_rewards = 0
    
    obs, info = env.reset()
    
    while True:
        queue_sel = queue_agent.select()
        channel_sel = channel_agent.select()
        backoff_sel = backoff_agent.select()
        
        action = {
            'queue_selection': queue_sel,
            'channel_selection': channel_sel,
            'backoff_value': backoff_sel
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards += reward
        
        queue_agent.update(queue_sel, reward)
        channel_agent.update(channel_sel, reward)
        backoff_agent.update(backoff_sel, reward)
        
        if terminated or truncated:
            print(f"Episode {episode} reward: {episode_rewards}")
            break

# Testing
env = WirelessNetworkEnv(config_name='congested_network', max_steps=100, render_mode='human')
queue_agent.epsilon = 0.0
channel_agent.epsilon = 0.0
backoff_agent.epsilon = 0.0

obs, info = env.reset()

while True:
    action = {
        'queue_selection': queue_agent.select(),
        'channel_selection': channel_agent.select(),
        'backoff_value': backoff_agent.select()
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

