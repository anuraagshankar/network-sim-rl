# Wireless Network Gymnasium Environment

A minimal Gymnasium environment for wireless network simulation based on `WNsimulator.py`.

## Overview

This environment simulates a wireless network where an agent controls one node competing with other nodes (using fixed policies) for access to multiple channels. The agent must learn to:

1. **Select which queue** (QoS level) to transmit from
2. **Choose a channel** to transmit on
3. **Set a backoff value** for collision avoidance

## Installation

Ensure you have the required dependencies:

```bash
# Activate the virtual environment
source wnsim/bin/activate

# Install gymnasium if not already installed
pip install gymnasium numpy
```

## Quick Start

```python
from wn_gym_env import WirelessNetworkEnv

# Create environment
env = WirelessNetworkEnv(
    num_channels=4,
    num_nodes=2,
    queue_size_limit=10,
    max_backoff=7
)

# Reset environment
obs, info = env.reset(seed=42)

# Take a step
action = {
    "queue": 0,      # 0 = HIGH_PRIO, 1 = LOW_PRIO
    "channel": 2,    # Channel index [0, num_channels-1]
    "backoff": 3     # Backoff value [0, max_backoff]
}
obs, reward, terminated, truncated, info = env.step(action)
```

## Action Space

The action space is a `Dict` with three components:

```python
{
    "queue": Discrete(2),                  # Queue selection (0=HIGH_PRIO, 1=LOW_PRIO)
    "channel": Discrete(num_channels),     # Channel selection
    "backoff": Discrete(max_backoff + 1)   # Backoff value
}
```

This structure allows for **multi-agent learning** where each action component can be controlled by a separate agent.

## Observation Space

The observation space is a `Dict` containing:

```python
{
    "queue_high_len": Discrete(queue_size_limit + 1),  # High priority queue length
    "queue_low_len": Discrete(queue_size_limit + 1),   # Low priority queue length
    "backoff_timer": Discrete(max_backoff + 1),        # Current backoff timer
    "channel_states": Box(0, 3, shape=(num_channels,)) # Last state of each channel
}
```

Channel states encoding:
- `0`: Idle (no transmission)
- `1`: Success (successful transmission)
- `2`: Collision (multiple transmissions interfered)
- `3`: External interference

## Reward Structure

The reward function encourages:
- **Successful transmissions**: +5.0 for high priority, +2.0 for low priority
- **Low latency**: Small penalty proportional to packet latency
- **Avoiding collisions**: -0.5 penalty for collisions
- **Avoiding drops**: -2.0 for high priority drops, -1.0 for low priority drops

## Environment Parameters

- `num_channels`: Number of available channels (default: 4)
- `num_nodes`: Total number of nodes including agent (default: 2)
- `agent_node_id`: ID of the agent-controlled node (default: 0)
- `queue_size_limit`: Maximum packets per queue (default: 10)
- `max_backoff`: Maximum backoff value (default: 7)
- `p_arrival_high`: High priority packet arrival probability (default: 0.01)
- `p_arrival_low`: Low priority packet arrival probability (default: 0.03)
- `gamma`: Channel interference probabilities (default: randomly generated)
- `max_steps`: Maximum episode length (default: 1000)

## Examples

See `test_wn_gym_env.py` for complete examples:

```bash
python test_wn_gym_env.py
```

## Multi-Agent Learning

The Dict action space naturally supports multi-agent learning:

```python
# Three separate agents, one for each action component
queue_agent = ...   # Learns queue selection policy
channel_agent = ... # Learns channel selection policy
backoff_agent = ... # Learns backoff policy

# Combine their outputs
action = {
    "queue": queue_agent.select_action(obs),
    "channel": channel_agent.select_action(obs),
    "backoff": backoff_agent.select_action(obs)
}
```

## Differences from Original Simulator

The original `WNsimulator.py` uses fixed policies for all nodes. This Gymnasium environment:
- Allows RL agents to control one node's decisions
- Uses a Dict action space for flexibility
- Implements a reward function for learning
- Follows the standard Gymnasium API
- Supports episode truncation at `max_steps`

## Next Steps

To train an agent, you can use any RL library that supports Gymnasium, such as:
- Stable-Baselines3
- RLlib
- CleanRL
- Custom implementations

Example with Stable-Baselines3:

```python
from stable_baselines3 import PPO
from wn_gym_env import WirelessNetworkEnv

env = WirelessNetworkEnv()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

Note: Since the action space is a Dict, you'll need to use `MultiInputPolicy` or handle the Dict actions appropriately in your learning algorithm.

