"""
Example Usage of Wireless Network RL Environment
This script demonstrates various ways to use the WN environment.
"""

import numpy as np
from wn_env import WirelessNetworkEnv


def example_random_policy():
    """Example 1: Using a random policy."""
    print("\n" + "="*60)
    print("Example 1: Random Policy")
    print("="*60)
    
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        max_steps=1000,
        scenario_seed=42
    )
    
    observation, info = env.reset()
    
    total_reward = 0
    for step in range(1000):
        # Random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {info['step']} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Successful Transmissions: {info['episode_successful_tx']}")
    print(f"Collisions: {info['episode_collisions']}")
    
    env.close()


def example_greedy_channel_selection():
    """Example 2: Greedy policy - always select channel with highest success rate."""
    print("\n" + "="*60)
    print("Example 2: Greedy Channel Selection Policy")
    print("="*60)
    
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        max_steps=1000,
        scenario_seed=42
    )
    
    observation, info = env.reset()
    
    total_reward = 0
    for step in range(1000):
        # Greedy action: Choose channel with highest success rate
        # (or random if all are equal, e.g., at the start)
        success_rates = observation['channel_success_rate']
        
        # If we have some data, choose best channel
        if np.max(success_rates) > 0:
            action = np.argmax(success_rates)
        else:
            # Random exploration if no data yet
            action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {info['step']} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Successful Transmissions: {info['episode_successful_tx']}")
    print(f"Collisions: {info['episode_collisions']}")
    
    env.close()


def example_epsilon_greedy():
    """Example 3: Epsilon-greedy policy."""
    print("\n" + "="*60)
    print("Example 3: Epsilon-Greedy Policy")
    print("="*60)
    
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        max_steps=1000,
        scenario_seed=42
    )
    
    epsilon = 0.1  # 10% exploration
    
    observation, info = env.reset()
    
    total_reward = 0
    for step in range(1000):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Explore: random action
            action = env.action_space.sample()
        else:
            # Exploit: choose best channel based on success rate
            success_rates = observation['channel_success_rate']
            if np.max(success_rates) > 0:
                action = np.argmax(success_rates)
            else:
                action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {info['step']} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Successful Transmissions: {info['episode_successful_tx']}")
    print(f"Collisions: {info['episode_collisions']}")
    
    env.close()


def example_avoid_collisions():
    """Example 4: Policy that avoids channels with high collision rates."""
    print("\n" + "="*60)
    print("Example 4: Collision-Avoidance Policy")
    print("="*60)
    
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        max_steps=1000,
        scenario_seed=42
    )
    
    observation, info = env.reset()
    
    total_reward = 0
    for step in range(1000):
        # Choose channel with lowest collision rate
        collision_rates = observation['channel_collision_rate']
        
        # If we have some data, avoid high-collision channels
        if np.max(collision_rates) > 0:
            # Select channel with minimum collision rate
            action = np.argmin(collision_rates)
        else:
            # Random if no data yet
            action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {info['step']} steps")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Successful Transmissions: {info['episode_successful_tx']}")
    print(f"Collisions: {info['episode_collisions']}")
    
    env.close()


def example_integration_with_rl_library():
    """Example 5: Template for integration with RL libraries (e.g., Stable-Baselines3)."""
    print("\n" + "="*60)
    print("Example 5: RL Library Integration Template")
    print("="*60)
    print("\nThis example shows how you would integrate with RL libraries.")
    print("Uncomment the code below and install stable-baselines3 to run:")
    print("  pip install stable-baselines3")
    
    # Uncomment the following code to use with Stable-Baselines3:
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    
    # Create environment
    env = WirelessNetworkEnv(
        n_nodes=2,
        c_channels=4,
        max_steps=1000,
        scenario_seed=42
    )
    
    # Wrap for vectorization (optional but recommended)
    # vec_env = make_vec_env(lambda: env, n_envs=1)
    
    # Create RL agent (e.g., PPO)
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=100000)
    
    # Save the model
    model.save("wn_ppo_agent")
    
    # Test the trained agent
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    print(f"Trained agent performance: {info['episode_rewards']:.2f}")
    """
    
    print("\nSee https://stable-baselines3.readthedocs.io/ for more details.")


if __name__ == "__main__":
    # Run all examples
    example_random_policy()
    example_greedy_channel_selection()
    example_epsilon_greedy()
    example_avoid_collisions()
    example_integration_with_rl_library()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)

