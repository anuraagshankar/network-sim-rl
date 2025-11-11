import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class Packet:
    """Packet class from the original simulator."""
    def __init__(self, arrival_slot, priority):
        self.arrival_slot = arrival_slot
        self.priority = priority


class WirelessNetworkEnv(gym.Env):
    """
    Minimal Gymnasium environment for wireless network simulation.
    
    A single agent controls one node in a multi-node wireless network.
    The agent must decide:
    1. Which queue (QoS level) to transmit from
    2. Which channel to use
    3. What backoff value to set
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        num_channels=4,
        num_nodes=2,
        agent_node_id=0,
        queue_size_limit=10,
        max_backoff=7,
        p_arrival_high=0.01,
        p_arrival_low=0.03,
        gamma=None,
        render_mode=None,
        max_steps=1000
    ):
        super().__init__()
        
        # Environment parameters
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.agent_node_id = agent_node_id  # Which node the agent controls
        self.queue_size_limit = queue_size_limit
        self.max_backoff = max_backoff
        self.p_arrival_high = p_arrival_high
        self.p_arrival_low = p_arrival_low
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Priority levels
        self.HIGH_PRIO = 0
        self.LOW_PRIO = 1
        
        # Channel interference probabilities
        if gamma is None:
            # Default: some clean channels, some noisy
            num_clean = num_channels // 4
            num_noisy = num_channels - num_clean
            gamma_clean = np.zeros(num_clean)
            gamma_noisy = np.random.uniform(0.1, 0.3, num_noisy)
            self.gamma = np.concatenate([gamma_clean, gamma_noisy])
            np.random.shuffle(self.gamma)
        else:
            self.gamma = np.array(gamma)
        
        # Define action space as Dict with 3 components
        # This allows for multi-agent learning where each component can have a separate agent
        self.action_space = spaces.Dict({
            "queue": spaces.Discrete(2),  # 0 = HIGH_PRIO, 1 = LOW_PRIO
            "channel": spaces.Discrete(num_channels),  # Which channel to transmit on
            "backoff": spaces.Discrete(max_backoff + 1)  # Backoff value [0, max_backoff]
        })
        
        # Define observation space
        # Includes: queue lengths, backoff timer, recent channel outcomes
        self.observation_space = spaces.Dict({
            "queue_high_len": spaces.Discrete(queue_size_limit + 1),
            "queue_low_len": spaces.Discrete(queue_size_limit + 1),
            "backoff_timer": spaces.Discrete(max_backoff + 1),
            # Last observed state of each channel (0=idle, 1=success, 2=collision, 3=external)
            "channel_states": spaces.Box(low=0, high=3, shape=(num_channels,), dtype=np.int32)
        })
        
        # Initialize state variables
        self.current_step = 0
        self.agent_queue_high = deque()
        self.agent_queue_low = deque()
        self.agent_backoff_timer = 0
        self.channel_states = np.zeros(num_channels, dtype=np.int32)
        
        # Other nodes (using fixed policy for simplicity)
        self.other_nodes_backoff = [0] * (num_nodes - 1)
        self.other_nodes_queues_high = [deque() for _ in range(num_nodes - 1)]
        self.other_nodes_queues_low = [deque() for _ in range(num_nodes - 1)]
        
        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_collisions": 0,
            "total_drops": 0,
            "total_latency": 0.0
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.agent_queue_high.clear()
        self.agent_queue_low.clear()
        self.agent_backoff_timer = 0
        self.channel_states = np.zeros(self.num_channels, dtype=np.int32)
        
        # Reset other nodes
        for i in range(self.num_nodes - 1):
            self.other_nodes_backoff[i] = 0
            self.other_nodes_queues_high[i].clear()
            self.other_nodes_queues_low[i].clear()
        
        # Reset statistics
        self.stats = {
            "total_sent": 0,
            "total_collisions": 0,
            "total_drops": 0,
            "total_latency": 0.0
        }
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one time step."""
        self.current_step += 1
        reward = 0.0
        
        # Parse action
        queue_choice = action["queue"]
        channel_choice = action["channel"]
        backoff_choice = action["backoff"]
        
        # Step 1: Packet arrivals for agent node
        if np.random.rand() < self.p_arrival_high:
            if len(self.agent_queue_high) < self.queue_size_limit:
                self.agent_queue_high.append(
                    Packet(arrival_slot=self.current_step, priority=self.HIGH_PRIO)
                )
            else:
                self.stats["total_drops"] += 1
                reward -= 2.0  # Penalty for dropping packet
        
        if np.random.rand() < self.p_arrival_low:
            if len(self.agent_queue_low) < self.queue_size_limit:
                self.agent_queue_low.append(
                    Packet(arrival_slot=self.current_step, priority=self.LOW_PRIO)
                )
            else:
                self.stats["total_drops"] += 1
                reward -= 1.0  # Smaller penalty for low priority drop
        
        # Step 2: Packet arrivals for other nodes (similar rates)
        for i in range(self.num_nodes - 1):
            if np.random.rand() < self.p_arrival_high:
                if len(self.other_nodes_queues_high[i]) < self.queue_size_limit:
                    self.other_nodes_queues_high[i].append(
                        Packet(arrival_slot=self.current_step, priority=self.HIGH_PRIO)
                    )
            if np.random.rand() < self.p_arrival_low:
                if len(self.other_nodes_queues_low[i]) < self.queue_size_limit:
                    self.other_nodes_queues_low[i].append(
                        Packet(arrival_slot=self.current_step, priority=self.LOW_PRIO)
                    )
        
        # Step 3: External interference
        external_busy = [np.random.rand() < g for g in self.gamma]
        
        # Step 4: Transmission decisions
        transmissions = [[] for _ in range(self.num_channels)]
        
        # Agent transmission decision
        agent_transmits = False
        agent_qos = None
        
        if self.agent_backoff_timer > 0:
            self.agent_backoff_timer -= 1
        else:
            # Check if agent has packets to send in the chosen queue
            if queue_choice == self.HIGH_PRIO and len(self.agent_queue_high) > 0:
                agent_transmits = True
                agent_qos = self.HIGH_PRIO
                transmissions[channel_choice].append(self.agent_node_id)
                self.agent_backoff_timer = backoff_choice
            elif queue_choice == self.LOW_PRIO and len(self.agent_queue_low) > 0:
                agent_transmits = True
                agent_qos = self.LOW_PRIO
                transmissions[channel_choice].append(self.agent_node_id)
                self.agent_backoff_timer = backoff_choice
        
        # Other nodes use fixed policy (strict priority, random channel, fixed backoff)
        for i in range(self.num_nodes - 1):
            if self.other_nodes_backoff[i] > 0:
                self.other_nodes_backoff[i] -= 1
            else:
                qos_to_send = -1
                if len(self.other_nodes_queues_high[i]) > 0:
                    qos_to_send = self.HIGH_PRIO
                elif len(self.other_nodes_queues_low[i]) > 0:
                    qos_to_send = self.LOW_PRIO
                
                if qos_to_send >= 0:
                    self.other_nodes_backoff[i] = np.random.randint(0, self.max_backoff + 1)
                    if self.other_nodes_backoff[i] == 0:
                        channel = np.random.randint(0, self.num_channels)
                        # Use negative indices to distinguish from agent
                        transmissions[channel].append(-(i + 1))
        
        # Step 5: Resolve outcomes
        slot_outcomes = np.zeros(self.num_channels, dtype=np.int32)  # 0=idle, 1=success, 2=collision, 3=external
        
        for j in range(self.num_channels):
            num_tx = len(transmissions[j])
            is_external = external_busy[j]
            
            if num_tx == 0:
                slot_outcomes[j] = 3 if is_external else 0  # external or idle
            elif num_tx == 1:
                node_id = transmissions[j][0]
                if is_external:
                    slot_outcomes[j] = 2  # collision with external
                    if node_id == self.agent_node_id:
                        self.stats["total_collisions"] += 1
                        reward -= 0.5  # Penalty for collision
                else:
                    slot_outcomes[j] = 1  # success
                    if node_id == self.agent_node_id:
                        # Agent successful transmission
                        queue = self.agent_queue_high if agent_qos == self.HIGH_PRIO else self.agent_queue_low
                        packet = queue.popleft()
                        latency = self.current_step - packet.arrival_slot
                        
                        self.stats["total_sent"] += 1
                        self.stats["total_latency"] += latency
                        
                        # Reward based on QoS and latency
                        base_reward = 5.0 if agent_qos == self.HIGH_PRIO else 2.0
                        latency_penalty = 0.1 * latency
                        reward += base_reward - latency_penalty
            else:  # num_tx > 1
                slot_outcomes[j] = 2  # collision
                if self.agent_node_id in transmissions[j]:
                    self.stats["total_collisions"] += 1
                    reward -= 0.5  # Penalty for collision
        
        # Update channel states for observation
        self.channel_states = slot_outcomes
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """Get current observation."""
        return {
            "queue_high_len": len(self.agent_queue_high),
            "queue_low_len": len(self.agent_queue_low),
            "backoff_timer": self.agent_backoff_timer,
            "channel_states": self.channel_states.copy()
        }
    
    def _get_info(self):
        """Get auxiliary information."""
        avg_latency = (
            self.stats["total_latency"] / self.stats["total_sent"]
            if self.stats["total_sent"] > 0
            else 0.0
        )
        return {
            "total_sent": self.stats["total_sent"],
            "total_collisions": self.stats["total_collisions"],
            "total_drops": self.stats["total_drops"],
            "avg_latency": avg_latency
        }
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"  Queue High: {len(self.agent_queue_high)}, Queue Low: {len(self.agent_queue_low)}")
            print(f"  Backoff Timer: {self.agent_backoff_timer}")
            print(f"  Channel States: {self.channel_states}")
            print(f"  Stats: {self._get_info()}")
    
    def close(self):
        """Clean up resources."""
        pass

