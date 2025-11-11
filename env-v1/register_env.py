"""
Optional: Register the WirelessNetworkEnv with Gymnasium.

This allows you to create the environment using:
    env = gymnasium.make('WirelessNetwork-v0')

Usage:
    import register_env  # Just import to register
    import gymnasium
    env = gymnasium.make('WirelessNetwork-v0')
"""

from gymnasium.envs.registration import register

register(
    id='WirelessNetwork-v0',
    entry_point='wn_gym_env:WirelessNetworkEnv',
    max_episode_steps=1000,
)

print("✓ WirelessNetwork-v0 registered with Gymnasium")

