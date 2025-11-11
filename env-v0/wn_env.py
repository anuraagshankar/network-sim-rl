"""
Wireless Network RL Environment (Gymnasium Compatible)
This environment simulates a wireless network where an RL agent learns
to optimize channel selection and transmission strategies.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, Tuple, Any, Optional

from network_config import (
    N_NODES, C_CHANNELS, Q_LIMIT, T_s, HIGH_PRIO, LOW_PRIO, 
    FIXED_BACKOFF_WINDOW, generate_random_scenario
)


# ==================================================================
# --- PACKET CLASS ---
# ==================================================================

class Packet:
    """Represents a network packet with arrival time and priority."""
    def __init__(self, arrival_slot: int, priority: int):
        self.arrival_slot = arrival_slot
        self.priority = priority


# ==================================================================
# --- NODE CLASS ---
# ==================================================================

class Node:
    """Represents a network node with packet queues and statistics."""
    
    def __init__(self, node_id: int, num_subchannels: int, queue_size_limit: int):
        self.id = node_id
        self.num_subchannels = num_subchannels
        self.queue_size_limit = queue_size_limit
        
        # Packet queues for different QoS levels
        self.packet_queue_high = deque()
        self.packet_queue_low = deque()
        
        # Backoff timer state
        self.backoff_timer = 0
        
        # Performance statistics
        self.stats = {
            HIGH_PRIO: {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0},
            LOW_PRIO:  {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0}
        }
        
        # Transmission statistics per channel
        self.tx_stats = [
            {'success': 0, 'collision': 0, 'total_tx': 0}
            for _ in range(num_subchannels)
        ]
        
        # Passive sensing statistics per channel
        self.learning_stats = [
            {'idle': 0, 'success': 0, 'collision': 0, 'external': 0, 'total_sensed': 0}
            for _ in range(num_subchannels)
        ]
    
    def get_queue_length(self, qos_level: int) -> int:
        """Returns the current queue length for a given QoS level."""
        if qos_level == HIGH_PRIO:
            return len(self.packet_queue_high)
        else:
            return len(self.packet_queue_low)
    
    def has_packets(self) -> bool:
        """Returns True if node has any packets to send."""
        return len(self.packet_queue_high) > 0 or len(self.packet_queue_low) > 0
    
    def get_priority_to_send(self) -> int:
        """Returns the priority of the next packet to send (-1 if none)."""
        if len(self.packet_queue_high) > 0:
            return HIGH_PRIO
        elif len(self.packet_queue_low) > 0:
            return LOW_PRIO
        else:
            return -1


# ==================================================================
# --- WIRELESS NETWORK GYMNASIUM ENVIRONMENT ---
# ==================================================================

class WirelessNetworkEnv(gym.Env):
    """
    A Gymnasium environment for wireless network channel selection.
    
    The agent controls one node (Node 0) and learns to select channels
    for transmission while other nodes use fixed policies.
    
    Action Space:
        Discrete(C_CHANNELS) - Select which channel to transmit on
    
    Observation Space:
        Dict containing:
            - agent_queue_high: Queue length for high priority packets
            - agent_queue_low: Queue length for low priority packets
            - agent_backoff: Current backoff timer value
            - channel_success_rate: Success rate per channel (from passive sensing)
            - channel_collision_rate: Collision rate per channel (from passive sensing)
    
    Reward:
        Basic reward function:
            +10 for successful transmission
            -5 for collision
            -1 for each dropped packet
            Small penalty for queue occupancy to encourage low latency
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self, 
        n_nodes: int = N_NODES,
        c_channels: int = C_CHANNELS,
        q_limit: int = Q_LIMIT,
        max_steps: int = 20000,
        scenario_seed: Optional[int] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Network parameters
        self.n_nodes = n_nodes
        self.c_channels = c_channels
        self.q_limit = q_limit
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Generate scenario parameters (traffic loads, interference)
        scenario = generate_random_scenario(n_nodes, c_channels, scenario_seed)
        self.p_arrivals_high = scenario['p_arrivals_high']
        self.p_arrivals_low = scenario['p_arrivals_low']
        self.gamma = scenario['gamma']
        
        # Agent controls Node 0
        self.agent_node_id = 0
        
        # Define action space: Choose which channel to transmit on
        self.action_space = spaces.Discrete(c_channels)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Queue information for the agent's node
            'agent_queue_high': spaces.Box(low=0, high=q_limit, shape=(1,), dtype=np.int32),
            'agent_queue_low': spaces.Box(low=0, high=q_limit, shape=(1,), dtype=np.int32),
            'agent_backoff': spaces.Box(low=0, high=FIXED_BACKOFF_WINDOW, shape=(1,), dtype=np.int32),
            # Channel statistics (learned through passive sensing)
            'channel_success_rate': spaces.Box(low=0.0, high=1.0, shape=(c_channels,), dtype=np.float32),
            'channel_collision_rate': spaces.Box(low=0.0, high=1.0, shape=(c_channels,), dtype=np.float32),
        })
        
        # Initialize state variables
        self.nodes = None
        self.current_step = 0
        self.total_successful_tx = 0
        self.total_collisions = 0
        
        # Tracking for episode statistics
        self.episode_stats = {
            'rewards': 0.0,
            'successful_tx': 0,
            'collisions': 0,
            'dropped_packets': 0
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Reset time step
        self.current_step = 0
        
        # Initialize all nodes
        self.nodes = [Node(k, self.c_channels, self.q_limit) for k in range(self.n_nodes)]
        
        # Reset global counters
        self.total_successful_tx = 0
        self.total_collisions = 0
        
        # Reset episode statistics
        self.episode_stats = {
            'rewards': 0.0,
            'successful_tx': 0,
            'collisions': 0,
            'dropped_packets': 0
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Channel selection for the agent's node (0 to C_CHANNELS-1)
        
        Returns:
            observation: New observation after taking action
            reward: Reward received for this action
            terminated: Whether episode has ended naturally
            truncated: Whether episode was cut off (time limit)
            info: Additional information dictionary
        """
        reward = 0.0
        agent_node = self.nodes[self.agent_node_id]
        
        # Track dropped packets this step for reward calculation
        dropped_this_step = 0
        
        # --- Step 1: Packet Arrival ---
        for k, node in enumerate(self.nodes):
            # High Priority Arrivals
            if np.random.rand() < self.p_arrivals_high[k]:
                node.stats[HIGH_PRIO]['received'] += 1
                if len(node.packet_queue_high) < node.queue_size_limit:
                    node.packet_queue_high.append(Packet(self.current_step, HIGH_PRIO))
                else:
                    node.stats[HIGH_PRIO]['dropped'] += 1
                    if k == self.agent_node_id:
                        dropped_this_step += 1
            
            # Low Priority Arrivals
            if np.random.rand() < self.p_arrivals_low[k]:
                node.stats[LOW_PRIO]['received'] += 1
                if len(node.packet_queue_low) < node.queue_size_limit:
                    node.packet_queue_low.append(Packet(self.current_step, LOW_PRIO))
                else:
                    node.stats[LOW_PRIO]['dropped'] += 1
                    if k == self.agent_node_id:
                        dropped_this_step += 1
        
        # --- Step 2: External Interference ---
        external_busy = [np.random.rand() < g for g in self.gamma]
        
        # --- Step 3: Transmission Decisions ---
        transmissions = [[] for _ in range(self.c_channels)]
        transmitting_nodes = set()
        actions_taken_this_slot = {}
        
        # Agent's transmission decision (if able to transmit)
        agent_transmitted = False
        if agent_node.backoff_timer == 0 and agent_node.has_packets():
            # Agent chooses channel based on RL policy
            chosen_channel = action
            qos_to_send = agent_node.get_priority_to_send()
            
            transmitting_nodes.add(self.agent_node_id)
            transmissions[chosen_channel].append(self.agent_node_id)
            actions_taken_this_slot[self.agent_node_id] = (chosen_channel, qos_to_send)
            agent_transmitted = True
            
            # Set new backoff timer for next transmission
            agent_node.backoff_timer = np.random.randint(0, FIXED_BACKOFF_WINDOW + 1)
        elif agent_node.backoff_timer > 0:
            agent_node.backoff_timer -= 1
        
        # Other nodes use fixed policy (random channel selection)
        for node in self.nodes:
            if node.id == self.agent_node_id:
                continue  # Skip agent node
            
            if node.backoff_timer > 0:
                node.backoff_timer -= 1
                continue
            
            qos_to_send = node.get_priority_to_send()
            if qos_to_send == -1:
                continue  # No packets to send
            
            # Set new backoff timer
            node.backoff_timer = np.random.randint(0, FIXED_BACKOFF_WINDOW + 1)
            
            if node.backoff_timer == 0:
                # Random channel selection
                chosen_channel = np.random.randint(0, self.c_channels)
                transmitting_nodes.add(node.id)
                transmissions[chosen_channel].append(node.id)
                actions_taken_this_slot[node.id] = (chosen_channel, qos_to_send)
        
        # --- Step 4: Resolve Outcomes ---
        slot_outcomes = [''] * self.c_channels
        agent_success = False
        agent_collision = False
        
        for j in range(self.c_channels):
            num_tx = len(transmissions[j])
            is_external = external_busy[j]
            
            if num_tx == 0:
                slot_outcomes[j] = 'external' if is_external else 'idle'
            
            elif num_tx == 1:
                node_id = transmissions[j][0]
                if is_external:
                    # Collision with external interference
                    slot_outcomes[j] = 'collision'
                    self.nodes[node_id].tx_stats[j]['collision'] += 1
                    self.total_collisions += 1
                    
                    if node_id == self.agent_node_id:
                        agent_collision = True
                else:
                    # Successful transmission
                    slot_outcomes[j] = 'success'
                    self.total_successful_tx += 1
                    
                    (j_chosen, qos_level) = actions_taken_this_slot[node_id]
                    tx_node = self.nodes[node_id]
                    queue = tx_node.packet_queue_high if qos_level == HIGH_PRIO else tx_node.packet_queue_low
                    packet = queue.popleft()
                    latency = (self.current_step + 1 - packet.arrival_slot) * T_s
                    
                    tx_node.stats[qos_level]['sent'] += 1
                    tx_node.stats[qos_level]['latency'] += latency
                    tx_node.tx_stats[j]['success'] += 1
                    
                    if node_id == self.agent_node_id:
                        agent_success = True
            
            else:  # num_tx > 1 - Collision between multiple nodes
                slot_outcomes[j] = 'collision'
                self.total_collisions += num_tx
                for node_id in transmissions[j]:
                    self.nodes[node_id].tx_stats[j]['collision'] += 1
                    if node_id == self.agent_node_id:
                        agent_collision = True
            
            # Update total TX attempts
            for node_id in transmissions[j]:
                self.nodes[node_id].tx_stats[j]['total_tx'] += 1
        
        # --- Step 5: Passive Sensing ---
        for node in self.nodes:
            if node.id not in transmitting_nodes:
                monitored_subchannel = np.random.randint(0, self.c_channels)
                state = slot_outcomes[monitored_subchannel]
                node.learning_stats[monitored_subchannel][state] += 1
                node.learning_stats[monitored_subchannel]['total_sensed'] += 1
        
        # --- Calculate Reward ---
        # Basic reward function (you can modify this later)
        if agent_transmitted:
            if agent_success:
                reward += 10.0  # Reward for successful transmission
                self.episode_stats['successful_tx'] += 1
            elif agent_collision:
                reward -= 5.0   # Penalty for collision
                self.episode_stats['collisions'] += 1
        
        # Penalty for dropped packets
        reward -= 1.0 * dropped_this_step
        self.episode_stats['dropped_packets'] += dropped_this_step
        
        # Small penalty for queue occupancy (encourages low latency)
        queue_penalty = 0.01 * (agent_node.get_queue_length(HIGH_PRIO) + 
                                agent_node.get_queue_length(LOW_PRIO))
        reward -= queue_penalty
        
        self.episode_stats['rewards'] += reward
        
        # --- Update Step Counter ---
        self.current_step += 1
        
        # Check termination conditions
        terminated = False  # Natural episode end (could be based on some condition)
        truncated = self.current_step >= self.max_steps
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construct observation from current environment state.
        
        Returns:
            Dictionary containing observation components
        """
        agent_node = self.nodes[self.agent_node_id]
        
        # Calculate channel statistics from passive sensing
        channel_success_rate = np.zeros(self.c_channels, dtype=np.float32)
        channel_collision_rate = np.zeros(self.c_channels, dtype=np.float32)
        
        for j in range(self.c_channels):
            stats = agent_node.learning_stats[j]
            total = stats['total_sensed']
            if total > 0:
                channel_success_rate[j] = stats['success'] / total
                channel_collision_rate[j] = stats['collision'] / total
        
        observation = {
            'agent_queue_high': np.array([agent_node.get_queue_length(HIGH_PRIO)], dtype=np.int32),
            'agent_queue_low': np.array([agent_node.get_queue_length(LOW_PRIO)], dtype=np.int32),
            'agent_backoff': np.array([agent_node.backoff_timer], dtype=np.int32),
            'channel_success_rate': channel_success_rate,
            'channel_collision_rate': channel_collision_rate,
        }
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary with diagnostic information
        """
        agent_node = self.nodes[self.agent_node_id]
        
        info = {
            'step': self.current_step,
            'agent_queue_high': agent_node.get_queue_length(HIGH_PRIO),
            'agent_queue_low': agent_node.get_queue_length(LOW_PRIO),
            'agent_packets_sent_high': agent_node.stats[HIGH_PRIO]['sent'],
            'agent_packets_sent_low': agent_node.stats[LOW_PRIO]['sent'],
            'agent_packets_dropped_high': agent_node.stats[HIGH_PRIO]['dropped'],
            'agent_packets_dropped_low': agent_node.stats[LOW_PRIO]['dropped'],
            'episode_rewards': self.episode_stats['rewards'],
            'episode_successful_tx': self.episode_stats['successful_tx'],
            'episode_collisions': self.episode_stats['collisions'],
            'episode_dropped_packets': self.episode_stats['dropped_packets'],
        }
        
        return info
    
    def render(self):
        """Render the environment (basic text output for now)."""
        if self.render_mode == "human":
            agent_node = self.nodes[self.agent_node_id]
            print(f"\n--- Step {self.current_step} ---")
            print(f"Agent Queue: High={agent_node.get_queue_length(HIGH_PRIO)}, "
                  f"Low={agent_node.get_queue_length(LOW_PRIO)}")
            print(f"Agent Backoff: {agent_node.backoff_timer}")
            print(f"Episode Reward: {self.episode_stats['rewards']:.2f}")
            print(f"Successful TX: {self.episode_stats['successful_tx']}, "
                  f"Collisions: {self.episode_stats['collisions']}")
    
    def close(self):
        """Clean up resources."""
        pass

