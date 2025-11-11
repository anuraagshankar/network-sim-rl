# Wireless Network RL Environment

A Gymnasium-compatible Reinforcement Learning environment for wireless network channel selection and optimization.

## Overview

This environment simulates a wireless network where an RL agent learns to optimize channel selection and transmission strategies. The environment maintains all the functionality of the original `WNsimulator.py` while providing a standard Gymnasium interface for RL training.

## Features

- **Gymnasium Compatible**: Fully compatible with Gymnasium API and RL libraries (Stable-Baselines3, RLlib, etc.)
- **Multi-Node Network**: Simulates multiple nodes with configurable parameters
- **QoS Support**: Two priority levels (High and Low) for packet traffic
- **Channel Interference**: Realistic external interference modeling
- **Passive Sensing**: Agent learns channel statistics through observation
- **Flexible Configuration**: Easy customization via `network_config.py`

## Environment Structure

### Action Space

`Discrete(C_CHANNELS)` - The agent selects which channel to transmit on (0 to C_CHANNELS-1)

### Observation Space

The observation is a dictionary containing:

- **`agent_queue_high`**: Number of high-priority packets in queue [0, Q_LIMIT]
- **`agent_queue_low`**: Number of low-priority packets in queue [0, Q_LIMIT]
- **`agent_backoff`**: Current backoff timer value [0, BACKOFF_WINDOW]
- **`channel_success_rate`**: Success rate per channel (from passive sensing) [0.0, 1.0]^C
- **`channel_collision_rate`**: Collision rate per channel (from passive sensing) [0.0, 1.0]^C

### Reward Function

The current basic reward function is:

```
reward = +10.0  if successful transmission
         -5.0   if collision
         -1.0   per dropped packet
         -0.01  * (queue_occupancy) for latency penalty
```

**Note**: This is a basic reward function. You can modify it in `wn_env.py` (search for "Calculate Reward") to suit your objectives.

### Episode Termination

- **Truncated**: Episode ends after `max_steps` time slots (default: 20000)
- **Terminated**: Currently not used (can be customized for specific conditions)

## Installation

1. Ensure you have the virtual environment activated:
```bash
source wnsim/bin/activate
```

2. Verify Gymnasium is installed:
```bash
pip install gymnasium
```

## Quick Start

### Basic Usage

```python
from wn_env import WirelessNetworkEnv

# Create environment
env = WirelessNetworkEnv(
    n_nodes=2,
    c_channels=4,
    max_steps=1000
)

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(1000):
    # Select action (channel)
    action = env.action_space.sample()  # Random action
    
    # Take step
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
```

### With RL Library (Stable-Baselines3)

```python
from stable_baselines3 import PPO
from wn_env import WirelessNetworkEnv

# Create environment
env = WirelessNetworkEnv(n_nodes=2, c_channels=4, max_steps=1000)

# Create and train agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## File Structure

```
simulator/
├── wn_env.py              # Main Gymnasium environment
├── network_config.py      # Network configuration parameters
├── WNsimulator.py         # Original simulator (kept for reference)
├── test_wn_env.py         # Test script for environment
├── example_usage.py       # Example policies and usage patterns
└── README_RL_ENV.md       # This file
```

## Configuration

Edit `network_config.py` to customize network parameters:

```python
N_NODES = 2              # Number of network nodes
C_CHANNELS = 4           # Number of available channels
Q_LIMIT = 10             # Maximum queue size
MAX_EPISODE_STEPS = 20000  # Maximum steps per episode
FIXED_BACKOFF_WINDOW = 7   # Backoff window size
```

### Scenario Generation

The environment automatically generates random scenarios with:
- Packet arrival rates (high and low priority)
- External interference patterns per channel

You can control randomness with the `scenario_seed` parameter:

```python
env = WirelessNetworkEnv(scenario_seed=42)  # Reproducible scenario
```

## Environment Details

### Multi-Agent Setup

Currently, the environment is single-agent:
- **Agent controls**: Node 0
- **Other nodes**: Use fixed policy (random channel selection)

To extend to multi-agent, you can modify the environment or create multiple instances.

### Packet Queues

Each node maintains two queues:
- **High Priority Queue**: Lower arrival rate, higher importance
- **Low Priority Queue**: Higher arrival rate, lower importance

The node uses strict priority scheduling (high priority first).

### Channel Dynamics

- **External Interference**: Each channel has a probability of interference (GAMMA)
- **Collisions**: Multiple nodes transmitting on the same channel causes collision
- **Passive Sensing**: Non-transmitting nodes observe random channels to learn statistics

### Performance Metrics

The `info` dictionary returned by `step()` includes:

```python
{
    'step': current_step,
    'agent_queue_high': high_priority_queue_length,
    'agent_queue_low': low_priority_queue_length,
    'agent_packets_sent_high': total_high_priority_sent,
    'agent_packets_sent_low': total_low_priority_sent,
    'agent_packets_dropped_high': total_high_priority_dropped,
    'agent_packets_dropped_low': total_low_priority_dropped,
    'episode_rewards': cumulative_reward,
    'episode_successful_tx': successful_transmissions,
    'episode_collisions': collision_count,
    'episode_dropped_packets': dropped_packet_count,
}
```

## Examples

See `example_usage.py` for various policy implementations:

1. **Random Policy**: Baseline random channel selection
2. **Greedy Policy**: Always select channel with highest success rate
3. **Epsilon-Greedy**: Balance exploration and exploitation
4. **Collision-Avoidance**: Select channels with lowest collision rate
5. **RL Integration**: Template for Stable-Baselines3

Run examples:
```bash
python example_usage.py
```

## Testing

Run the test suite:
```bash
python test_wn_env.py
```

This will:
- Verify observation structure
- Test environment reset and step
- Run sample episodes with random actions
- Display performance metrics

## Customization

### Modifying the Reward Function

Edit the reward calculation in `wn_env.py` (around line 360):

```python
# --- Calculate Reward ---
# Customize this section for your objectives
if agent_transmitted:
    if agent_success:
        reward += 10.0  # Adjust success reward
    elif agent_collision:
        reward -= 5.0   # Adjust collision penalty

# Add your custom reward components here
```

### Adding Observations

To add new observations, modify:
1. `observation_space` definition in `__init__()` (line ~120)
2. `_get_observation()` method (line ~400)

### Changing Action Space

To modify actions (e.g., add power control), update:
1. `action_space` definition in `__init__()`
2. Action processing in `step()` method

## Performance Tips

1. **Vectorized Environments**: Use Gymnasium's vectorization for faster training
2. **Episode Length**: Shorter episodes train faster; adjust `max_steps` as needed
3. **Reward Scaling**: Normalize rewards for stable learning
4. **Observation Normalization**: Consider normalizing observations for neural networks

## Comparison with Original Simulator

The RL environment maintains all functionality from `WNsimulator.py`:

| Feature | WNsimulator.py | wn_env.py |
|---------|---------------|-----------|
| Multi-node network | ✅ | ✅ |
| QoS levels | ✅ | ✅ |
| Channel interference | ✅ | ✅ |
| Backoff mechanism | ✅ | ✅ |
| Passive sensing | ✅ | ✅ |
| Statistics tracking | ✅ | ✅ |
| RL interface | ❌ | ✅ |
| Gymnasium compatible | ❌ | ✅ |

## Future Enhancements

Potential extensions:
- Multi-agent support
- Power control actions
- Dynamic channel conditions
- More sophisticated reward shaping
- Rendering/visualization
- Different MAC protocols

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Original simulator: `WNsimulator.py`

## License

Same as the parent project.

## Contact

For questions or issues, please refer to the main project documentation.

