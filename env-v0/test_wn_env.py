"""
Test script for Wireless Network Gymnasium Environment
This demonstrates how to use the WN environment with random actions.
"""

import gymnasium as gym
from wn_env import WirelessNetworkEnv
import numpy as np


def test_random_agent(num_episodes=3, max_steps=100):
    """
    Test the environment with a random agent.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("="*60)
    print("Testing Wireless Network RL Environment")
    print("="*60)
    
    # Create the environment
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        q_limit=10,
        max_steps=max_steps,
        scenario_seed=42,  # For reproducibility
        render_mode="human"
    )
    
    print(f"\nEnvironment created successfully!")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"\nNetwork Configuration:")
    print(f"  - Nodes: {env.n_nodes}")
    print(f"  - Channels: {env.c_channels}")
    print(f"  - Queue Limit: {env.q_limit}")
    print(f"  - External Interference (GAMMA): {np.round(env.gamma, 3)}")
    print(f"  - P_Arrivals HIGH: {np.round(env.p_arrivals_high, 3)}")
    print(f"  - P_Arrivals LOW: {np.round(env.p_arrivals_low, 3)}")
    
    # Run episodes with random actions
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        observation, info = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Random action selection
            action = env.action_space.sample()
            
            # Take action in environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Render every 20 steps
            if step % 20 == 0:
                env.render()
            
            # Check if episode is done
            if terminated or truncated:
                reason = "Terminated" if terminated else "Truncated (Time Limit)"
                print(f"\nEpisode ended at step {step}: {reason}")
                break
        
        # Print episode summary
        print(f"\n--- Episode {episode + 1} Summary ---")
        print(f"Total Steps: {info['step']}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Successful Transmissions: {info['episode_successful_tx']}")
        print(f"Collisions: {info['episode_collisions']}")
        print(f"Dropped Packets: {info['episode_dropped_packets']}")
        print(f"Agent Packets Sent (High/Low): {info['agent_packets_sent_high']}/{info['agent_packets_sent_low']}")
        print(f"Agent Packets Dropped (High/Low): {info['agent_packets_dropped_high']}/{info['agent_packets_dropped_low']}")
    
    env.close()
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}")


def test_observation_structure():
    """Test that observations have the correct structure."""
    print("\n" + "="*60)
    print("Testing Observation Structure")
    print("="*60)
    
    env = WirelessNetworkEnv(n_nodes=2, c_channels=4, max_steps=100)
    observation, info = env.reset()
    
    print("\nObservation Keys:", observation.keys())
    print("\nObservation Details:")
    for key, value in observation.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        print(f"    Value: {value}")
    
    print("\nInfo Keys:", info.keys())
    
    # Take a few random steps
    print("\nTaking 5 random steps...")
    for i in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.2f}, "
              f"queue_high={observation['agent_queue_high'][0]}, "
              f"queue_low={observation['agent_queue_low'][0]}")
    
    env.close()
    print("\nObservation structure test completed!")


if __name__ == "__main__":
    # Test observation structure first
    test_observation_structure()
    
    # Run random agent test
    test_random_agent(num_episodes=2, max_steps=100)

