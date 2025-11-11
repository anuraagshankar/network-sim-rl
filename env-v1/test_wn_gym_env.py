"""
Example usage of the Wireless Network Gymnasium environment.
"""
import numpy as np
from wn_gym_env import WirelessNetworkEnv


def random_policy_example():
    """Run the environment with a random policy."""
    print("=" * 50)
    print("Testing WirelessNetworkEnv with Random Policy")
    print("=" * 50)
    
    # Create environment
    env = WirelessNetworkEnv(
        num_channels=4,
        num_nodes=2,
        queue_size_limit=10,
        max_backoff=7,
        p_arrival_high=0.01,
        p_arrival_low=0.03,
        max_steps=1000,
        render_mode=None  # Set to "human" to see step-by-step output
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    print()
    
    # Run episode with random actions
    total_reward = 0.0
    for step in range(100):  # Run for 100 steps
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Observation: {obs}")
            print(f"  Reward: {reward:.2f}")
            print(f"  Info: {info}")
            print()
        
        if terminated or truncated:
            break
    
    print("=" * 50)
    print(f"Episode finished after {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final stats: {info}")
    print("=" * 50)
    
    env.close()


def simple_heuristic_policy_example():
    """Run the environment with a simple heuristic policy."""
    print("\n" + "=" * 50)
    print("Testing WirelessNetworkEnv with Heuristic Policy")
    print("=" * 50)
    
    # Create environment
    env = WirelessNetworkEnv(
        num_channels=4,
        num_nodes=2,
        queue_size_limit=10,
        max_backoff=7,
        max_steps=1000
    )
    
    obs, info = env.reset(seed=123)
    
    total_reward = 0.0
    for step in range(100):
        # Simple heuristic:
        # - Always prefer high priority queue if non-empty
        # - Choose channel with most recent success (or random if none)
        # - Use moderate backoff (middle value)
        
        if obs["queue_high_len"] > 0:
            queue_choice = 0  # HIGH_PRIO
        elif obs["queue_low_len"] > 0:
            queue_choice = 1  # LOW_PRIO
        else:
            queue_choice = 0  # Default
        
        # Choose channel that was last seen as idle or successful
        channel_scores = []
        for ch_state in obs["channel_states"]:
            if ch_state == 0 or ch_state == 1:  # idle or success
                channel_scores.append(1)
            else:
                channel_scores.append(0)
        
        if sum(channel_scores) > 0:
            # Choose among good channels
            good_channels = [i for i, score in enumerate(channel_scores) if score > 0]
            channel_choice = np.random.choice(good_channels)
        else:
            # All channels look bad, choose randomly
            channel_choice = np.random.randint(0, env.num_channels)
        
        # Use moderate backoff
        backoff_choice = env.max_backoff // 2
        
        action = {
            "queue": queue_choice,
            "channel": channel_choice,
            "backoff": backoff_choice
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: Reward = {reward:.2f}, Total = {total_reward:.2f}, Info = {info}")
        
        if terminated or truncated:
            break
    
    print("=" * 50)
    print(f"Episode finished after {step + 1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final stats: {info}")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    # Run examples
    random_policy_example()
    simple_heuristic_policy_example()

