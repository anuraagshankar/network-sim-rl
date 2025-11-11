# Wireless Network Simulator

A Gymnasium-compatible environment for simulating wireless network packet scheduling with multi-priority queues and channel interference.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from wn_env import WirelessNetworkEnv

# Load a configuration
env = WirelessNetworkEnv(config_name='simple_network')
obs, info = env.reset()

# Run episode
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### With Rendering

```python
# Enable rendering to generate replay and visualization
env = WirelessNetworkEnv(config_name='simple_network', render_mode='human')
obs, info = env.reset()

# Run episode - visualization appears at end
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Render Existing Replay

```bash
python replay_renderer.py replays/replay_TIMESTAMP.json
```

**Controls:**
- Space: Play/Pause
- Left/Right arrows: Step backward/forward
- ESC: Quit

## Configuration File Components

Configuration files are JSON files stored in `network-configs/` directory.

### Structure

```json
{
  "name": "Configuration Name",
  "description": "Description of the network scenario",
  "network": { ... },
  "arrival_rates": { ... },
  "channel_interference": { ... }
}
```

### Network Parameters

```json
"network": {
  "n_nodes": 1,
  "n_channels": 2,
  "max_backoff": 7,
  "queue_limit": 10,
  "max_steps": 1000,
  "t_s": 1.0
}
```

- **n_nodes**: Number of nodes in the network (only node 0 is RL-controlled)
- **n_channels**: Number of available channels
- **max_backoff**: Maximum backoff value (creates Discrete(max_backoff+1) action space)
- **queue_limit**: Maximum queue size for each priority level
- **max_steps**: Maximum episode length
- **t_s**: Time slot duration in milliseconds

### Arrival Rates

**Fixed arrival rates:**
```json
"arrival_rates": {
  "type": "fixed",
  "high_priority": [0.01, 0.005],
  "low_priority": [0.02, 0.03]
}
```
- Provide one value per node
- If fewer values than nodes, the last value is repeated

**Random arrival rates:**
```json
"arrival_rates": {
  "type": "random",
  "high_priority": {"min": 0.005, "max": 0.01},
  "low_priority": {"min": 0.01, "max": 0.05}
}
```
- Generates random arrival rates within specified ranges
- Rates are sampled uniformly on each `reset()`

### Channel Interference

**Fixed interference:**
```json
"channel_interference": {
  "type": "fixed",
  "values": [0.0, 0.1, 0.5, 0.9]
}
```
- Provide one value per channel (0.0 = clean, 1.0 = always busy)
- If fewer values than channels, the last value is repeated

**Random interference:**
```json
"channel_interference": {
  "type": "random",
  "clean_channels_ratio": 0.25,
  "noisy_range": {"min": 0.0, "max": 0.3}
}
```
- **clean_channels_ratio**: Fraction of channels with 0 interference
- **noisy_range**: Uniform range for noisy channel interference
- Channels are shuffled randomly on each `reset()`

## Visualization Components

The replay renderer displays six key components of the wireless network:

### 1. Nodes
- Circles with node IDs (N0, N1, etc.)
- Located in the upper portion of the screen
- Shows backoff timer (B:X) below each node

### 2. Queues
- Two vertical bars per node
- **Red bar (left)**: High priority queue
- **Green bar (right)**: Low priority queue
- Fill level indicates queue occupancy
- Label shows queue type (H/L) and current size

### 3. Queue Selection
- When a node decides to transmit, the selected queue is highlighted brighter
- Indicates which priority level was chosen for transmission

### 4. Packet Transmission
- Colored line from selected queue to target channel
- **Red line**: High priority packet
- **Green line**: Low priority packet
- Shows the transmission path

### 5. Channel Status
- Rectangular boxes in the lower portion
- **Green fill**: Successful transmission
- **Red fill**: Collision occurred
- **Orange fill**: External interference present
- **Gray outline**: Channel idle
- Shows channel ID (Ch0, Ch1, etc.) and interference rate

### 6. Channel Interference
- Displayed as numerical value on each channel box
- Represents probability of external interference (0.0 to 1.0)
- Channels with external interference in current step show orange fill

## Action Space

Dict with three components:
- **queue_selection**: Discrete(2) - select HIGH_PRIO (0) or LOW_PRIO (1)
- **channel_selection**: Discrete(n_channels) - select channel
- **backoff_value**: Discrete(max_backoff+1) - select backoff value

## Observation Space

Box containing:
- Queue lengths (high and low priority)
- Backoff timer
- Channel interference rates
- Recent collision indicators per channel

