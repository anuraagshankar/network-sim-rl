import numpy as np
from rl_agents.agent_runner import ENV_CLASS, CONFIG_NAME, RENDER_MODE, TRAIN_EPS, TRAIN_MAX_STEPS, TEST_MAX_STEPS
import json

class QLearningAgent:
    def __init__(self, n_states, n_arms, epsilon=1.0, alpha=0.1, gamma=0.95):
        self.n_states = n_states
        self.n_arms = n_arms
        self.epsilon = epsilon
        # Q-learning: learning rate (alpha) controls how much new information overrides old
        self.alpha = alpha
        # Q-learning: discount factor (gamma) weights importance of future rewards
        self.gamma = gamma
        self.q_values = np.zeros((n_states, n_arms))
    
    def select(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values[state])
    
    # Q-learning: update using Bellman equation with next state value
    def update(self, state, arm, reward, next_state):
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        best_next_value = np.max(self.q_values[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_values[state, arm]
        self.q_values[state, arm] += self.alpha * td_error

class NodeAgent:
    def __init__(self, n_channels, max_backoff, epsilon=1.0, alpha=0.1, gamma=0.95):
        self.queue_agent = QLearningAgent(4, 2, epsilon, alpha, gamma)
        self.channel_agent = QLearningAgent(2**n_channels, n_channels, epsilon, alpha, gamma)
        self.backoff_agent = QLearningAgent(4, 2, epsilon, alpha, gamma)
    
    def set_epsilon(self, epsilon):
        self.queue_agent.epsilon = epsilon
        self.channel_agent.epsilon = epsilon
        self.backoff_agent.epsilon = epsilon
    
    def select_action(self, obs, n_channels):
        queue_state = self.get_queue_state(obs)
        channel_state = self.get_channel_state(obs, n_channels)
        backoff_state = self.get_backoff_state(obs)
        
        return {
            'queue_selection': self.queue_agent.select(queue_state),
            'channel_selection': self.channel_agent.select(channel_state),
            'backoff_value': self.backoff_agent.select(backoff_state)
        }, (queue_state, channel_state, backoff_state)
    
    # Q-learning: update now requires next_states for temporal difference learning
    def update(self, states, action, reward, next_states):
        queue_state, channel_state, backoff_state = states
        next_queue_state, next_channel_state, next_backoff_state = next_states
        
        self.queue_agent.update(queue_state, action['queue_selection'], reward, next_queue_state)
        self.channel_agent.update(channel_state, action['channel_selection'], reward, next_channel_state)
        self.backoff_agent.update(backoff_state, action['backoff_value'], reward, next_backoff_state)
    
    def get_queue_state(self, obs):
        high_empty = 1 if obs[0] == 0 else 0
        low_empty = 1 if obs[1] == 0 else 0
        return high_empty * 2 + low_empty
    
    def get_channel_state(self, obs, n_channels):
        collision_history = obs[-n_channels:]
        state = 0
        for i, val in enumerate(collision_history):
            if val > 0.5:
                state += 2**i
        return state
    
    def get_backoff_state(self, obs):
        backoff_binary = 0 if obs[2] == 0 else 1
        timestep_binary = int(obs[3]) % 2
        return backoff_binary * 2 + timestep_binary


# Training
env = ENV_CLASS(config_name=CONFIG_NAME)
if TRAIN_MAX_STEPS is not None:
    env.max_steps = TRAIN_MAX_STEPS
observations, infos = env.reset()

n_episodes = TRAIN_EPS
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
        
        # Q-learning: extract next states for temporal difference updates
        next_states = {}
        for agent_name in env.agents:
            next_queue_state = agents[agent_name].get_queue_state(observations[agent_name])
            next_channel_state = agents[agent_name].get_channel_state(observations[agent_name], env.n_channels)
            next_backoff_state = agents[agent_name].get_backoff_state(observations[agent_name])
            next_states[agent_name] = (next_queue_state, next_channel_state, next_backoff_state)
        
        for agent_name in env.agents:
            episode_rewards[agent_name] += rewards[agent_name]
            agents[agent_name].update(states[agent_name], actions[agent_name], rewards[agent_name], next_states[agent_name])
        
        if all(truncations.values()) or all(terminations.values()):
            print(f"Episode {episode} rewards: {episode_rewards}")
            break

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
        action, _ = agents[agent_name].select_action(observations[agent_name], env.n_channels)
        actions[agent_name] = action
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if all(truncations.values()) or all(terminations.values()):
        break

print("Throughput and latency metrics:")
print(json.dumps(infos[env.agents[0]]['node_latency_throughput'], indent=2))

