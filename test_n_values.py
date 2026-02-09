"""
Flexible N-Reward Testing Script
---------------------------------
Easily test different N values with ANY agent type.

USAGE:
1. Choose your agent in the CONFIGURATION section below
2. Set N values to test
3. Run the script: python3 test_n_values.py

Available Agents:
- 'random'              : Random actions (baseline)
- 'epsilon_greedy'      : Simple epsilon-greedy
- 'contextual'          : Contextual epsilon-greedy (state-aware)
- 'ucb'                 : Upper Confidence Bound
- 'thompson'            : Thompson Sampling
"""

import numpy as np
import sys
sys.path.append('/home/claude/WNsim/network-sim-rl')
from standalone_n_reward_env import WirelessNetworkEnvNReward


# ============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS
# ============================================================================

# Choose agent type:
# Options: 'random', 'epsilon_greedy', 'contextual', 'ucb', 'thompson'
AGENT_TYPE = 'contextual'

# N values to test
N_VALUES = [1, 50, 100]

# Training settings
N_EPISODES = 100           # Number of training episodes
STEPS_PER_EPISODE = 5000   # Steps per episode
EPSILON_START = 1.0        # Initial exploration (for epsilon-greedy agents)
EPSILON_END = 0.01         # Final exploration
UCB_C = 2.0                # UCB exploration parameter

# Testing settings
N_TEST_EPISODES = 10       # Episodes for final testing

# Random seed for reproducibility
SEED = 42

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class RandomAgent:
    """Random action selection - no learning."""
    def __init__(self, n_arms):
        self.n_arms = n_arms
    
    def select(self, state=None):
        return np.random.randint(self.n_arms)
    
    def update(self, *args):
        pass  # No learning
    
    def get_best_action(self, state=None):
        return np.random.randint(self.n_arms)


class EpsilonGreedyAgent:
    """Simple epsilon-greedy agent."""
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
    
    def select(self, state=None):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]
    
    def get_best_action(self, state=None):
        return np.argmax(self.q_values)


class ContextualAgent:
    """Contextual epsilon-greedy agent."""
    def __init__(self, n_states, n_arms, epsilon=1.0):
        self.n_states = n_states
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros((n_states, n_arms))
        self.counts = np.zeros((n_states, n_arms))
    
    def select(self, state):
        if state is None:
            state = 0
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.q_values[state])
    
    def update(self, state, arm, reward):
        if state is None:
            state = 0
        self.counts[state, arm] += 1
        self.q_values[state, arm] += (reward - self.q_values[state, arm]) / self.counts[state, arm]
    
    def get_best_action(self, state):
        if state is None:
            state = 0
        return np.argmax(self.q_values[state])


class UCBAgent:
    """Upper Confidence Bound agent."""
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.total_counts = 0
    
    def select(self, state=None):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_counts) / self.counts
        )
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_counts += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]
    
    def get_best_action(self, state=None):
        return np.argmax(self.q_values)


class ThompsonSamplingAgent:
    """Thompson Sampling agent."""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def select(self, state=None):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_best_action(self, state=None):
        expected_values = self.alpha / (self.alpha + self.beta)
        return np.argmax(expected_values)


# ============================================================================
# STATE EXTRACTION (for contextual agent)
# ============================================================================

def get_queue_state(obs):
    """Extract queue state from observation."""
    high_empty = 1 if obs[0] == 0 else 0
    low_empty = 1 if obs[1] == 0 else 0
    return high_empty * 2 + low_empty


def get_channel_state(obs, n_channels):
    """Extract channel collision state from observation."""
    collision_history = obs[-n_channels:]
    state = 0
    for i, val in enumerate(collision_history):
        if val > 0.5:
            state += 2**i
    return min(state, 2**n_channels - 1)


def get_backoff_state(obs):
    """Extract backoff state from observation."""
    return 0 if obs[2] == 0 else 1


# ============================================================================
# AGENT FACTORY
# ============================================================================

def create_agents(agent_type, env):
    """Create agents based on agent_type string."""
    
    if agent_type == 'random':
        return (
            RandomAgent(2),
            RandomAgent(env.n_channels),
            RandomAgent(env.max_backoff + 1)
        ), False
    
    elif agent_type == 'epsilon_greedy':
        return (
            EpsilonGreedyAgent(2, EPSILON_START),
            EpsilonGreedyAgent(env.n_channels, EPSILON_START),
            EpsilonGreedyAgent(env.max_backoff + 1, EPSILON_START)
        ), False
    
    elif agent_type == 'contextual':
        return (
            ContextualAgent(4, 2, EPSILON_START),
            ContextualAgent(2**env.n_channels, env.n_channels, EPSILON_START),
            ContextualAgent(2, env.max_backoff + 1, EPSILON_START)
        ), True
    
    elif agent_type == 'ucb':
        return (
            UCBAgent(2, UCB_C),
            UCBAgent(env.n_channels, UCB_C),
            UCBAgent(env.max_backoff + 1, UCB_C)
        ), False
    
    elif agent_type == 'thompson':
        return (
            ThompsonSamplingAgent(2),
            ThompsonSamplingAgent(env.n_channels),
            ThompsonSamplingAgent(env.max_backoff + 1)
        ), False
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_agent(n_value, agent_type):
    """Train agent with specific N value."""
    
    print(f"\n{'='*70}")
    print(f"Training {agent_type.upper()} with N = {n_value}")
    print(f"{'='*70}")
    
    # Create environment
    env = WirelessNetworkEnvNReward(
        config_name='simple_network',
        n_reward_turns=n_value,
        max_steps=STEPS_PER_EPISODE,
        seed=SEED
    )
    
    obs, info = env.reset(seed=SEED)
    
    # Create agents
    (queue_agent, channel_agent, backoff_agent), is_contextual = create_agents(agent_type, env)
    
    # Track metrics
    episode_rewards = []
    episode_latencies = []
    episode_high_latencies = []
    episode_low_latencies = []
    
    # Training loop
    for episode in range(N_EPISODES):
        # Decay epsilon for epsilon-greedy agents
        if hasattr(queue_agent, 'epsilon'):
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * episode / N_EPISODES
            queue_agent.epsilon = epsilon
            channel_agent.epsilon = epsilon
            backoff_agent.epsilon = epsilon
        
        obs, info = env.reset(seed=SEED + episode)
        total_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            # Get action based on agent type
            if is_contextual:
                queue_state = get_queue_state(obs)
                channel_state = get_channel_state(obs, env.n_channels)
                backoff_state = get_backoff_state(obs)
                
                action = {
                    'queue_selection': queue_agent.select(queue_state),
                    'channel_selection': channel_agent.select(channel_state),
                    'backoff_value': backoff_agent.select(backoff_state)
                }
            else:
                action = {
                    'queue_selection': queue_agent.select(),
                    'channel_selection': channel_agent.select(),
                    'backoff_value': backoff_agent.select()
                }
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Update agents
            if is_contextual:
                queue_agent.update(queue_state, action['queue_selection'], reward)
                channel_agent.update(channel_state, action['channel_selection'], reward)
                backoff_agent.update(backoff_state, action['backoff_value'], reward)
            else:
                queue_agent.update(action['queue_selection'], reward)
                channel_agent.update(action['channel_selection'], reward)
                backoff_agent.update(action['backoff_value'], reward)
            
            if terminated or truncated:
                break
        
        # Get metrics
        metrics = env.compute_latency_metrics()
        if 0 in metrics:
            node_metrics = metrics[0]
            episode_latencies.append(node_metrics['latency'])
            episode_high_latencies.append(node_metrics['high_priority_latency'])
            episode_low_latencies.append(node_metrics['low_priority_latency'])
        else:
            episode_latencies.append(0)
            episode_high_latencies.append(0)
            episode_low_latencies.append(0)
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode+1}: Reward={total_reward:.1f}, "
                  f"Latency={episode_latencies[-1]:.1f} ms")
    
    print(f"  Final Avg Reward: {np.mean(episode_rewards[-10:]):.1f}")
    print(f"  Final Avg Latency: {np.mean(episode_latencies[-10:]):.1f} ms")
    
    return {
        'n_value': n_value,
        'agent_type': agent_type,
        'agents': (queue_agent, channel_agent, backoff_agent),
        'is_contextual': is_contextual,
        'episode_rewards': episode_rewards,
        'episode_latencies': episode_latencies,
        'episode_high_latencies': episode_high_latencies,
        'episode_low_latencies': episode_low_latencies
    }


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_agent(result):
    """Test trained agent."""
    
    print(f"  Testing N={result['n_value']}...")
    
    queue_agent, channel_agent, backoff_agent = result['agents']
    is_contextual = result['is_contextual']
    
    # Disable exploration
    if hasattr(queue_agent, 'epsilon'):
        queue_agent.epsilon = 0.0
        channel_agent.epsilon = 0.0
        backoff_agent.epsilon = 0.0
    
    test_latencies = []
    test_high_latencies = []
    test_low_latencies = []
    test_rewards = []
    
    for test_ep in range(N_TEST_EPISODES):
        env = WirelessNetworkEnvNReward(
            config_name='simple_network',
            n_reward_turns=1,  # Use N=1 for testing
            max_steps=5000,
            seed=1000 + test_ep
        )
        
        obs, info = env.reset(seed=1000 + test_ep)
        total_reward = 0
        
        for step in range(5000):
            if is_contextual:
                queue_state = get_queue_state(obs)
                channel_state = get_channel_state(obs, env.n_channels)
                backoff_state = get_backoff_state(obs)
                
                action = {
                    'queue_selection': queue_agent.select(queue_state),
                    'channel_selection': channel_agent.select(channel_state),
                    'backoff_value': backoff_agent.select(backoff_state)
                }
            else:
                action = {
                    'queue_selection': queue_agent.select(),
                    'channel_selection': channel_agent.select(),
                    'backoff_value': backoff_agent.select()
                }
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        metrics = env.compute_latency_metrics()
        if 0 in metrics:
            node_metrics = metrics[0]
            test_latencies.append(node_metrics['latency'])
            test_high_latencies.append(node_metrics['high_priority_latency'])
            test_low_latencies.append(node_metrics['low_priority_latency'])
        
        test_rewards.append(total_reward)
    
    result['test_latency'] = np.mean(test_latencies)
    result['test_latency_std'] = np.std(test_latencies)
    result['test_high_latency'] = np.mean(test_high_latencies)
    result['test_low_latency'] = np.mean(test_low_latencies)
    result['test_reward'] = np.mean(test_rewards)
    
    print(f"    Test Latency: {result['test_latency']:.2f} ± {result['test_latency_std']:.2f} ms")


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_results(results):
    """Print results table."""
    
    agent_type = results[0]['agent_type'].upper()
    
    print("\n" + "="*100)
    print(f"N-REWARD RESULTS - {agent_type} AGENT")
    print("="*100)
    print(f"{'N':<6} {'Overall Latency (ms)':<30} {'High Priority (ms)':<25} {'Low Priority (ms)':<25}")
    print(f"{'':6} {'Mean ± Std':<30} {'Mean':<25} {'Mean':<25}")
    print("-"*100)
    
    for result in results:
        n = result['n_value']
        overall = f"{result['test_latency']:.2f} ± {result['test_latency_std']:.2f}"
        high = f"{result['test_high_latency']:.2f}"
        low = f"{result['test_low_latency']:.2f}"
        
        print(f"{n:<6} {overall:<30} {high:<25} {low:<25}")
    
    print("="*100)
    
    print("\nREWARD SUMMARY:")
    print("-"*100)
    print(f"{'N':<6} {'Test Reward (Mean)':<30}")
    print("-"*100)
    
    for result in results:
        n = result['n_value']
        reward = f"{result['test_reward']:.1f}"
        print(f"{n:<6} {reward:<30}")
    
    print("="*100)


def save_results(results):
    """Save results to file."""
    
    agent_type = results[0]['agent_type']
    filename = f'/mnt/user-data/outputs/n_values_results_{agent_type}.txt'
    
    with open(filename, 'w') as f:
        f.write(f"N-REWARD RESULTS - {agent_type.upper()} AGENT\n")
        f.write("="*100 + "\n")
        f.write(f"{'N':<6} {'Overall Latency (ms)':<30} {'High Priority (ms)':<25} {'Low Priority (ms)':<25}\n")
        f.write(f"{'':6} {'Mean ± Std':<30} {'Mean':<25} {'Mean':<25}\n")
        f.write("-"*100 + "\n")
        
        for result in results:
            n = result['n_value']
            overall = f"{result['test_latency']:.2f} ± {result['test_latency_std']:.2f}"
            high = f"{result['test_high_latency']:.2f}"
            low = f"{result['test_low_latency']:.2f}"
            
            f.write(f"{n:<6} {overall:<30} {high:<25} {low:<25}\n")
        
        f.write("="*100 + "\n\n")
        f.write("REWARD SUMMARY:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'N':<6} {'Test Reward (Mean)':<30}\n")
        f.write("-"*100 + "\n")
        
        for result in results:
            n = result['n_value']
            reward = f"{result['test_reward']:.1f}"
            f.write(f"{n:<6} {reward:<30}\n")
        
        f.write("="*100 + "\n")
    
    print(f"\nResults saved to: {filename}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    
    print("\n" + "="*70)
    print("N-REWARD TESTING")
    print("="*70)
    print(f"\nAgent Type: {AGENT_TYPE.upper()}")
    print(f"N Values: {N_VALUES}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Steps per Episode: {STEPS_PER_EPISODE}")
    
    results = []
    
    # Train and test each N value
    for n in N_VALUES:
        result = train_agent(n, AGENT_TYPE)
        test_agent(result)
        results.append(result)
    
    # Display and save results
    print_results(results)
    save_results(results)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()