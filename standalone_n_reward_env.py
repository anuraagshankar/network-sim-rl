"""
Standalone Wireless Network Environment with N-turn Reward
Does not require gymnasium - uses basic numpy only
"""

import numpy as np
from collections import deque
import sys
sys.path.append('/home/claude/WNsim/network-sim-rl')
from network_configs.config_loader import ConfigLoader


class Packet:
    """Represents a packet in the network."""
    def __init__(self, arrival_slot, priority):
        self.arrival_slot = arrival_slot
        self.priority = priority


class Node:
    """Represents a network node."""
    def __init__(self, node_id, num_channels, queue_limit):
        self.id = node_id
        self.num_channels = num_channels
        self.queue_limit = queue_limit
        self.packet_queue_high = deque()
        self.packet_queue_low = deque()
        self.backoff_timer = 0
        self.stats = {
            0: {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0, 'collisions': 0},
            1: {'sent': 0, 'latency': 0.0, 'received': 0, 'dropped': 0, 'collisions': 0}
        }


class WirelessNetworkEnvNReward:
    """
    Standalone Wireless Network Environment with N-turn reward mechanism.
    
    The agent receives rewards only after successfully transmitting N packets.
    """
    
    # Reward configuration
    SUCCESS_REWARD_SINGLE = 10.0
    HIGH_PRIORITY_BONUS_SINGLE = 5.0
    COLLISION_PENALTY = 1.0
    LATENCY_PENALTY_COEFF = 0.0
    QUEUE_PENALTY_COEFF = 0.0
    
    def __init__(self, config_name=None, n_nodes=2, n_channels=4, max_backoff=7, 
                 queue_limit=10, max_steps=1000, seed=None, n_reward_turns=1):
        """
        Initialize the wireless network environment with N-turn reward.
        
        Args:
            n_reward_turns: Number of successful transmissions before giving reward (N)
                          N=1 means immediate reward (original behavior)
        """
        # N-turn reward parameter
        self.n_reward_turns = n_reward_turns
        self.accumulated_reward = 0.0
        self.successful_tx_count = 0
        
        # Load configuration if specified
        if config_name is not None:
            self.config = ConfigLoader.load(config_name)
            self.n_nodes = self.config.n_nodes
            self.n_channels = self.config.n_channels
            self.max_backoff = self.config.max_backoff
            self.queue_limit = self.config.queue_limit
            self.max_steps = self.config.max_steps
            self.t_s = self.config.t_s
        else:
            self.config = None
            self.n_nodes = n_nodes
            self.n_channels = n_channels
            self.max_backoff = max_backoff
            self.queue_limit = queue_limit
            self.max_steps = max_steps
            self.t_s = 1.0
        
        # Priority levels
        self.HIGH_PRIO = 0
        self.LOW_PRIO = 1
        
        if seed is not None:
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset N-turn reward tracking
        self.accumulated_reward = 0.0
        self.successful_tx_count = 0
        
        # Initialize nodes
        self.nodes = [Node(k, self.n_channels, self.queue_limit) for k in range(self.n_nodes)]
        
        # Load arrival rates and interference from config or generate random
        if self.config is not None:
            self.p_arrivals_high, self.p_arrivals_low = self.config.generate_arrival_rates()
            self.gamma = self.config.generate_channel_interference()
        else:
            self.p_arrivals_high = np.random.uniform(0.005, 0.01, self.n_nodes)
            self.p_arrivals_low = np.random.uniform(0.01, 0.05, self.n_nodes)
            num_clean = self.n_channels // 4
            num_noisy = self.n_channels - num_clean
            gamma_clean = np.zeros(num_clean)
            gamma_noisy = np.random.uniform(0.0, 0.3, num_noisy)
            self.gamma = np.concatenate([gamma_clean, gamma_noisy])
            np.random.shuffle(self.gamma)
        
        self.current_step = 0
        self.collision_history = np.zeros(self.n_channels)
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        """Get current observation."""
        rl_node = self.nodes[0]
        obs = np.array([
            len(rl_node.packet_queue_high),
            len(rl_node.packet_queue_low),
            rl_node.backoff_timer,
            *self.gamma,
            *self.collision_history
        ], dtype=np.float32)
        return obs
    
    def _get_info(self):
        """Get additional info."""
        node = self.nodes[0]
        return {
            'queue_high': len(node.packet_queue_high),
            'queue_low': len(node.packet_queue_low),
            'backoff': node.backoff_timer,
            'sent_high': node.stats[self.HIGH_PRIO]['sent'],
            'sent_low': node.stats[self.LOW_PRIO]['sent'],
            'collisions': sum(node.stats[q]['collisions'] for q in [0, 1]),
            'successful_tx_count': self.successful_tx_count,
            'accumulated_reward': self.accumulated_reward
        }
    
    def get_random_action(self):
        """Get a random action."""
        return {
            'queue_selection': np.random.randint(0, 2),
            'channel_selection': np.random.randint(0, self.n_channels),
            'backoff_value': np.random.randint(0, self.max_backoff + 1)
        }
    
    def compute_latency_metrics(self):
        """Compute latency and throughput metrics for each node."""
        if not self.nodes or self.current_step == 0:
            return {}
        
        elapsed_slots = self.current_step
        elapsed_time = elapsed_slots * self.t_s
        metrics = {}
        
        for node in self.nodes:
            # Overall metrics
            total_sent = sum(node.stats[q]['sent'] for q in [self.HIGH_PRIO, self.LOW_PRIO])
            total_latency = sum(node.stats[q]['latency'] for q in [self.HIGH_PRIO, self.LOW_PRIO])
            average_latency = total_latency / total_sent if total_sent > 0 else 0.0
            throughput = (total_sent / elapsed_time) if elapsed_time > 0 else 0.0
            
            # Per-QoS metrics
            high_sent = node.stats[self.HIGH_PRIO]['sent']
            low_sent = node.stats[self.LOW_PRIO]['sent']
            high_latency = node.stats[self.HIGH_PRIO]['latency'] / high_sent if high_sent > 0 else 0.0
            low_latency = node.stats[self.LOW_PRIO]['latency'] / low_sent if low_sent > 0 else 0.0
            
            metrics[node.id] = {
                'latency': average_latency,
                'throughput': throughput,
                'high_priority_latency': high_latency,
                'low_priority_latency': low_latency,
                'high_priority_sent': high_sent,
                'low_priority_sent': low_sent
            }
        
        return metrics
    
    def step(self, action):
        """Execute one time step with N-turn reward mechanism."""
        # Extract action components
        rl_queue_sel = action['queue_selection']
        rl_channel_sel = action['channel_selection']
        rl_backoff = action['backoff_value']
        
        # Step 1: Packet arrivals
        for k, node in enumerate(self.nodes):
            if np.random.rand() < self.p_arrivals_high[k]:
                node.stats[self.HIGH_PRIO]['received'] += 1
                if len(node.packet_queue_high) < self.queue_limit:
                    node.packet_queue_high.append(Packet(self.current_step, self.HIGH_PRIO))
                else:
                    node.stats[self.HIGH_PRIO]['dropped'] += 1
            
            if np.random.rand() < self.p_arrivals_low[k]:
                node.stats[self.LOW_PRIO]['received'] += 1
                if len(node.packet_queue_low) < self.queue_limit:
                    node.packet_queue_low.append(Packet(self.current_step, self.LOW_PRIO))
                else:
                    node.stats[self.LOW_PRIO]['dropped'] += 1
        
        # Step 2: External interference
        external_busy = [np.random.rand() < g for g in self.gamma]
        
        # Step 3: Transmission decisions
        transmissions = [[] for _ in range(self.n_channels)]
        actions_taken = {}
        
        # RL-controlled node (node 0)
        rl_node = self.nodes[0]
        if rl_node.backoff_timer > 0:
            rl_node.backoff_timer -= 1
        else:
            # Check if selected queue has packets
            if rl_queue_sel == self.HIGH_PRIO and len(rl_node.packet_queue_high) > 0:
                qos = self.HIGH_PRIO
            elif rl_queue_sel == self.LOW_PRIO and len(rl_node.packet_queue_low) > 0:
                qos = self.LOW_PRIO
            else:
                qos = -1
            
            if qos != -1:
                rl_node.backoff_timer = rl_backoff
                if rl_backoff == 0:
                    transmissions[rl_channel_sel].append(0)
                    actions_taken[0] = (rl_channel_sel, qos)
        
        # Other nodes follow fixed policy
        for k in range(1, self.n_nodes):
            node = self.nodes[k]
            if node.backoff_timer > 0:
                node.backoff_timer -= 1
                continue
            
            qos = -1
            if len(node.packet_queue_high) > 0:
                qos = self.HIGH_PRIO
            elif len(node.packet_queue_low) > 0:
                qos = self.LOW_PRIO
            else:
                continue
            
            node.backoff_timer = np.random.randint(0, self.max_backoff + 1)
            
            if node.backoff_timer == 0:
                channel = np.random.randint(0, self.n_channels)
                transmissions[channel].append(k)
                actions_taken[k] = (channel, qos)
        
        # Step 4: Resolve outcomes and calculate immediate reward
        immediate_reward = 0.0
        rl_success_this_step = False
        self.collision_history *= 0.9
        
        for j in range(self.n_channels):
            num_tx = len(transmissions[j])
            is_external = external_busy[j]
            
            if num_tx == 1:
                node_id = transmissions[j][0]
                if not is_external:
                    # Success
                    channel, qos = actions_taken[node_id]
                    
                    node = self.nodes[node_id]
                    queue = (node.packet_queue_high if qos == self.HIGH_PRIO 
                            else node.packet_queue_low)
                    packet = queue.popleft()
                    latency = (self.current_step + 1 - packet.arrival_slot) * self.t_s
                    
                    node.stats[qos]['sent'] += 1
                    node.stats[qos]['latency'] += latency
                    
                    # Calculate reward for RL node
                    if node_id == 0:
                        rl_success_this_step = True
                        immediate_reward += self.SUCCESS_REWARD_SINGLE
                        immediate_reward -= latency * self.LATENCY_PENALTY_COEFF
                        if qos == self.HIGH_PRIO:
                            immediate_reward += self.HIGH_PRIORITY_BONUS_SINGLE
                else:
                    # Collision with external interference
                    node_id = transmissions[j][0]
                    if node_id == 0:
                        immediate_reward -= self.COLLISION_PENALTY
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    self.collision_history[j] = 1.0
            
            elif num_tx > 1:
                # Collision between nodes
                for node_id in transmissions[j]:
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    if node_id == 0:
                        immediate_reward -= self.COLLISION_PENALTY
                self.collision_history[j] = 1.0
        
        # Queue penalty for RL node
        rl_node = self.nodes[0]
        queue_occupancy = (len(rl_node.packet_queue_high) + 
                          len(rl_node.packet_queue_low)) / (2 * self.queue_limit)
        immediate_reward -= queue_occupancy * self.QUEUE_PENALTY_COEFF
        
        # N-turn reward mechanism
        reward = 0.0
        self.accumulated_reward += immediate_reward
        
        if rl_success_this_step:
            self.successful_tx_count += 1
            
            # Check if we've reached N successful transmissions
            if self.successful_tx_count >= self.n_reward_turns:
                reward = self.accumulated_reward
                self.accumulated_reward = 0.0
                self.successful_tx_count = 0
        
        # Step counter
        self.current_step += 1
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info