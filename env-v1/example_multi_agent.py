"""
Example: Multi-Agent Structure for WirelessNetworkEnv

This demonstrates how the Dict action space naturally supports
having 3 separate agents, each responsible for one action component.
"""
import numpy as np
from wn_gym_env import WirelessNetworkEnv


class SimpleQueueAgent:
    """Agent that decides which queue to transmit from."""
    
    def select_action(self, obs):
        # Simple priority policy: prefer high priority if non-empty
        if obs["queue_high_len"] > 0:
            return 0  # HIGH_PRIO
        elif obs["queue_low_len"] > 0:
            return 1  # LOW_PRIO
        else:
            return 0  # Default to high priority


class SimpleChannelAgent:
    """Agent that decides which channel to transmit on."""
    
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.channel_scores = np.ones(num_channels)  # Track channel performance
    
    def select_action(self, obs):
        # Choose channel based on recent observations
        # Prefer channels that were idle or had successful transmissions
        channel_states = obs["channel_states"]
        
        # Update scores based on observed states
        for i, state in enumerate(channel_states):
            if state == 0 or state == 1:  # idle or success
                self.channel_scores[i] = 0.9 * self.channel_scores[i] + 0.1
            else:  # collision or external interference
                self.channel_scores[i] = 0.9 * self.channel_scores[i]
        
        # Select channel with highest score
        return int(np.argmax(self.channel_scores))


class SimpleBackoffAgent:
    """Agent that decides the backoff value."""
    
    def __init__(self, max_backoff):
        self.max_backoff = max_backoff
        self.recent_collisions = 0
        self.recent_successes = 0
    
    def select_action(self, obs):
        # Adaptive backoff: increase backoff if more collisions detected
        # This is a simple heuristic - a learning agent would do better
        
        # Check if any channel had collision
        has_collision = any(state == 2 for state in obs["channel_states"])
        if has_collision:
            self.recent_collisions += 1
        else:
            self.recent_successes += 1
        
        # Decay counters
        self.recent_collisions = int(0.95 * self.recent_collisions)
        self.recent_successes = int(0.95 * self.recent_successes)
        
        # Higher collision rate -> higher backoff
        if self.recent_collisions > self.recent_successes:
            return np.random.randint(self.max_backoff // 2, self.max_backoff + 1)
        else:
            return np.random.randint(0, self.max_backoff // 2 + 1)


def run_multi_agent_example():
    """Run environment with three coordinated agents."""
    print("=" * 60)
    print("Multi-Agent Example: 3 Agents Working Together")
    print("=" * 60)
    
    # Create environment
    env = WirelessNetworkEnv(
        num_channels=4,
        num_nodes=2,
        queue_size_limit=10,
        max_backoff=7,
        max_steps=1000
    )
    
    # Create three agents
    queue_agent = SimpleQueueAgent()
    channel_agent = SimpleChannelAgent(num_channels=env.num_channels)
    backoff_agent = SimpleBackoffAgent(max_backoff=env.max_backoff)
    
    print("Agents created:")
    print("  - Queue Agent: Prioritizes high-priority traffic")
    print("  - Channel Agent: Learns which channels are less congested")
    print("  - Backoff Agent: Adapts backoff based on collision rate")
    print()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    episode_length = 200
    
    for step in range(episode_length):
        # Each agent selects its component of the action
        action = {
            "queue": queue_agent.select_action(obs),
            "channel": channel_agent.select_action(obs),
            "backoff": backoff_agent.select_action(obs)
        }
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{episode_length}")
            print(f"  Action: queue={action['queue']}, "
                  f"channel={action['channel']}, backoff={action['backoff']}")
            print(f"  Queue lengths: high={obs['queue_high_len']}, "
                  f"low={obs['queue_low_len']}")
            print(f"  Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"  Stats: {info}")
            print(f"  Channel scores: {np.round(channel_agent.channel_scores, 2)}")
            print()
        
        if terminated or truncated:
            break
    
    # Final results
    print("=" * 60)
    print(f"Episode Complete: {step + 1} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward per Step: {total_reward / (step + 1):.4f}")
    print()
    print("Final Statistics:")
    print(f"  Packets Sent: {info['total_sent']}")
    print(f"  Collisions: {info['total_collisions']}")
    print(f"  Drops: {info['total_drops']}")
    print(f"  Avg Latency: {info['avg_latency']:.2f} slots")
    print()
    print(f"  Success Rate: "
          f"{info['total_sent'] / max(1, info['total_sent'] + info['total_collisions']) * 100:.1f}%")
    print("=" * 60)
    
    env.close()


def compare_policies():
    """Compare random policy vs multi-agent policy."""
    print("\n" + "=" * 60)
    print("Comparison: Random Policy vs Multi-Agent Policy")
    print("=" * 60)
    
    num_episodes = 5
    episode_length = 200
    
    # Test random policy
    random_rewards = []
    for ep in range(num_episodes):
        env = WirelessNetworkEnv(max_steps=episode_length)
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        
        for _ in range(episode_length):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        random_rewards.append(total_reward)
        env.close()
    
    # Test multi-agent policy
    multi_agent_rewards = []
    for ep in range(num_episodes):
        env = WirelessNetworkEnv(max_steps=episode_length)
        obs, _ = env.reset(seed=ep)
        
        queue_agent = SimpleQueueAgent()
        channel_agent = SimpleChannelAgent(num_channels=env.num_channels)
        backoff_agent = SimpleBackoffAgent(max_backoff=env.max_backoff)
        
        total_reward = 0.0
        for _ in range(episode_length):
            action = {
                "queue": queue_agent.select_action(obs),
                "channel": channel_agent.select_action(obs),
                "backoff": backoff_agent.select_action(obs)
            }
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        multi_agent_rewards.append(total_reward)
        env.close()
    
    # Print comparison
    print(f"\nResults over {num_episodes} episodes:")
    print(f"  Random Policy:")
    print(f"    Mean Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Multi-Agent Policy:")
    print(f"    Mean Reward: {np.mean(multi_agent_rewards):.2f} ± {np.std(multi_agent_rewards):.2f}")
    print(f"\n  Improvement: {((np.mean(multi_agent_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_multi_agent_example()
    compare_policies()

