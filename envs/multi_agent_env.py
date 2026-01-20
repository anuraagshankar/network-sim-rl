import json
import os
from datetime import datetime

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from envs.reward_config import (CENTRAL_COLLISION_PENALTY,
                                CENTRAL_IDLE_PENALTY_BASE,
                                CENTRAL_IDLE_PENALTY_FACTOR, CENTRAL_TX_REWARD,
                                COLLISION_PENALTY, HIGH_PRIORITY_BONUS,
                                LATENCY_PENALTY_COEFF, QUEUE_PENALTY_COEFF,
                                SUCCESS_REWARD)
from envs.single_agent_env import Node, Packet
from network_configs.config_loader import ConfigLoader


class WirelessNetworkParallelEnv(ParallelEnv):
    """
    PettingZoo parallel environment for wireless network simulation.
    All nodes are RL agents that take actions simultaneously.
    """
    
    metadata = {'render_modes': ['human'], 'name': 'wireless_network_parallel_v0'}
    
    def __init__(self, config_name=None, n_nodes=2, n_channels=4, max_backoff=7, 
                 queue_limit=10, max_steps=1000, seed=None, render_mode=None):
        """Initialize the parallel wireless network environment."""
        super().__init__()
        
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
        
        self.HIGH_PRIO = 0
        self.LOW_PRIO = 1
        
        self.possible_agents = [f"node_{i}" for i in range(self.n_nodes)]
        self.agents = self.possible_agents[:]
        
        self._action_spaces = {
            agent: spaces.Dict({
                'queue_selection': spaces.Discrete(2),
                'channel_selection': spaces.Discrete(self.n_channels),
                'backoff_value': spaces.Discrete(self.max_backoff + 1)
            }) for agent in self.possible_agents
        }
        
        obs_dim = (
            2 +  # queue lengths (high, low)
            1 +  # backoff timer
            1 +  # timestep
            self.n_channels +  # channel interference rates
            self.n_channels  # recent collision indicators per channel
        )
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        if seed is not None:
            np.random.seed(seed)
        
        self.render_mode = render_mode
        self.replay_log = []
        self.replay_file = None
        
        self.nodes = None
        self.current_step = 0
        self.p_arrivals_high = None
        self.p_arrivals_low = None
        self.gamma = None
        self.collision_history = None
    
    @property
    def num_agents(self):
        return len(self.agents)
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.nodes = [Node(k, self.n_channels, self.queue_limit) 
                      for k in range(self.n_nodes)]
        
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
        
        self.collision_history = np.zeros(self.n_channels)
        self.current_step = 0
        
        if self.render_mode is not None:
            self.replay_log = []
            self.replay_log.append({
                'type': 'init',
                'step': 0,
                'n_nodes': self.n_nodes,
                'n_channels': self.n_channels,
                'max_backoff': self.max_backoff,
                'queue_limit': self.queue_limit,
                'p_arrivals_high': self.p_arrivals_high.tolist(),
                'p_arrivals_low': self.p_arrivals_low.tolist(),
                'gamma': self.gamma.tolist()
            })
        
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: self._get_info(i) for i, agent in enumerate(self.agents)}
        
        return observations, infos
    
    def _get_obs(self, node_id):
        """Get observation for a specific node."""
        node = self.nodes[node_id]
        
        obs = np.array([
            len(node.packet_queue_high),
            len(node.packet_queue_low),
            node.backoff_timer,
            self.current_step,
            *self.gamma,
            *self.collision_history
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self, node_id):
        """Get info for a specific node."""
        node = self.nodes[node_id]
        return {
            'step': self.current_step,
            'queue_high': len(node.packet_queue_high),
            'queue_low': len(node.packet_queue_low),
            'backoff': node.backoff_timer,
            'sent_high': node.stats[self.HIGH_PRIO]['sent'],
            'sent_low': node.stats[self.LOW_PRIO]['sent'],
            'collisions': sum(node.stats[q]['collisions'] for q in [0, 1])
        }
    
    def _compute_node_latency_throughput(self):
        """Compute latency and throughput metrics for each node."""
        if not self.nodes or self.current_step == 0:
            return {}
        
        elapsed_slots = self.current_step
        elapsed_time = elapsed_slots * self.t_s
        metrics = {}
        
        for node in self.nodes:
            total_sent = sum(node.stats[q]['sent'] for q in [self.HIGH_PRIO, self.LOW_PRIO])
            total_latency = sum(node.stats[q]['latency'] for q in [self.HIGH_PRIO, self.LOW_PRIO])
            average_latency = total_latency / total_sent if total_sent > 0 else 0.0
            throughput = (total_sent / elapsed_time) if elapsed_time > 0 else 0.0
            
            metrics[node.id] = {
                'latency': average_latency,
                'throughput': throughput
            }
        
        return metrics
    
    def step(self, actions):
        """Execute one time step with all agents acting in parallel."""
        
        # Step 1: Packet arrivals
        arrivals = {}
        for k, node in enumerate(self.nodes):
            arrivals[k] = {'high': False, 'low': False, 'dropped_high': False, 'dropped_low': False}
            if np.random.rand() < self.p_arrivals_high[k]:
                node.stats[self.HIGH_PRIO]['received'] += 1
                if len(node.packet_queue_high) < self.queue_limit:
                    node.packet_queue_high.append(
                        Packet(self.current_step, self.HIGH_PRIO)
                    )
                    arrivals[k]['high'] = True
                else:
                    node.stats[self.HIGH_PRIO]['dropped'] += 1
                    arrivals[k]['dropped_high'] = True
            
            if np.random.rand() < self.p_arrivals_low[k]:
                node.stats[self.LOW_PRIO]['received'] += 1
                if len(node.packet_queue_low) < self.queue_limit:
                    node.packet_queue_low.append(
                        Packet(self.current_step, self.LOW_PRIO)
                    )
                    arrivals[k]['low'] = True
                else:
                    node.stats[self.LOW_PRIO]['dropped'] += 1
                    arrivals[k]['dropped_low'] = True
        
        # Step 2: External interference
        external_busy = [np.random.rand() < g for g in self.gamma]
        
        # Step 3: Transmission decisions (all agents act in parallel)
        transmissions = [[] for _ in range(self.n_channels)]
        actions_taken = {}
        
        for i, agent in enumerate(self.agents):
            node = self.nodes[i]
            action = actions[agent]
            
            queue_sel = action['queue_selection']
            channel_sel = action['channel_selection']
            backoff = action['backoff_value']
            
            if node.backoff_timer > 0:
                node.backoff_timer -= 1
            
            if node.backoff_timer == 0:
                # Check if selected queue has packets
                if queue_sel == self.HIGH_PRIO and len(node.packet_queue_high) > 0:
                    qos = self.HIGH_PRIO
                elif queue_sel == self.LOW_PRIO and len(node.packet_queue_low) > 0:
                    qos = self.LOW_PRIO
                else:
                    qos = -1
                
                if qos != -1:
                    node.backoff_timer = backoff
                    if backoff == 0:
                        transmissions[channel_sel].append(i)
                        actions_taken[i] = (channel_sel, qos)
        
        # Step 4: Resolve outcomes
        successful_tx = {}
        rewards = {agent: 0.0 for agent in self.agents}
        self.collision_history *= 0.9
        
        for j in range(self.n_channels):
            num_tx = len(transmissions[j])
            is_external = external_busy[j]
            
            if num_tx == 1:
                node_id = transmissions[j][0]
                if not is_external:
                    # Success
                    _, qos = actions_taken[node_id]
                    successful_tx[node_id] = qos
                    
                    node = self.nodes[node_id]
                    queue = (node.packet_queue_high if qos == self.HIGH_PRIO 
                            else node.packet_queue_low)
                    packet = queue.popleft()
                    latency = (self.current_step + 1 - packet.arrival_slot) * self.t_s
                    
                    node.stats[qos]['sent'] += 1
                    node.stats[qos]['latency'] += latency
                    
                    # Reward
                    agent_name = self.agents[node_id]
                    rewards[agent_name] += SUCCESS_REWARD
                    rewards[agent_name] -= latency * LATENCY_PENALTY_COEFF
                    if qos == self.HIGH_PRIO:
                        rewards[agent_name] += HIGH_PRIORITY_BONUS
                else:
                    # Collision with external interference
                    node_id = transmissions[j][0]
                    agent_name = self.agents[node_id]
                    rewards[agent_name] -= COLLISION_PENALTY
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    self.collision_history[j] = 1.0
            
            elif num_tx > 1:
                # Collision between nodes
                for node_id in transmissions[j]:
                    agent_name = self.agents[node_id]
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    rewards[agent_name] -= COLLISION_PENALTY
                self.collision_history[j] = 1.0
        
        # Queue penalty for all agents
        for i, agent in enumerate(self.agents):
            node = self.nodes[i]
            queue_occupancy = (len(node.packet_queue_high) + 
                              len(node.packet_queue_low)) / (2 * self.queue_limit)
            rewards[agent] -= queue_occupancy * QUEUE_PENALTY_COEFF
        
        # Log step data for replay
        if self.render_mode is not None:
            step_data = {
                'type': 'step',
                'step': int(self.current_step),
                'arrivals': arrivals,
                'external_busy': [bool(x) for x in external_busy],
                'queue_states': {k: {
                    'high': len(node.packet_queue_high),
                    'low': len(node.packet_queue_low),
                    'backoff': int(node.backoff_timer)
                } for k, node in enumerate(self.nodes)},
                'actions': {k: {'channel': int(v[0]), 'queue': int(v[1])} for k, v in actions_taken.items()},
                'transmissions': {j: [int(n) for n in tx] for j, tx in enumerate(transmissions) if tx},
                'successes': {k: int(v) for k, v in successful_tx.items()},
                'collisions': [int(j) for j in range(self.n_channels) if self.collision_history[j] > 0.5]
            }
            self.replay_log.append(step_data)
        
        self.current_step += 1
        
        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        
        # Save replay and render if episode finished
        if all(truncations.values()) and self.render_mode is not None:
            self._save_replay()
            if self.render_mode == 'human':
                self._render_replay()
        
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: self._get_info(i) for i, agent in enumerate(self.agents)}
        
        if all(truncations.values()):
            metrics = self._compute_node_latency_throughput()
            for i, agent in enumerate(self.agents):
                infos[agent]['node_latency_throughput'] = metrics
        
        return observations, rewards, terminations, truncations, infos
    
    def _save_replay(self):
        """Save replay log to file."""
        os.makedirs('replays', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.replay_file = f'replays/replay_{timestamp}.json'
        with open(self.replay_file, 'w', encoding='utf-8') as f:
            json.dump(self.replay_log, f, indent=2)
    
    def _render_replay(self):
        """Render the replay using the renderer."""
        if self.replay_file:
            from envs.replay_renderer import render_replay
            render_replay(self.replay_file)
    
    def render(self):
        """Render the environment."""
        return None
    
    def state(self):
        """Return global state (not used in parallel environments)."""
        return None


class CentralizedRewardParallelEnv(WirelessNetworkParallelEnv):
    """
    Extends WirelessNetworkParallelEnv with a centralized reward structure.
    
    Additional Rewards:
    - When a node transmits, all other nodes receive a small reward (cooperative incentive).
    - When a node stays idle, all other nodes receive a penalty that grows with idle duration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track last transmission time for each node
        self.node_last_tx_step = None 

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        
        # Initialize last transmission steps to 0
        self.node_last_tx_step = np.zeros(self.n_nodes, dtype=int)
        
        return observations, infos

    def step(self, actions):
        # Capture stats before step to detect transmissions
        pre_stats = []
        for node in self.nodes:
            sent = node.stats[self.HIGH_PRIO]['sent'] + node.stats[self.LOW_PRIO]['sent']
            collisions = node.stats[self.HIGH_PRIO]['collisions'] + node.stats[self.LOW_PRIO]['collisions']
            pre_stats.append({'sent': sent, 'collisions': collisions})
            
        # Execute base environment step
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # Determine which nodes transmitted (successful or collision)
        did_transmit = [False] * self.n_nodes
        cur_stats = []
        for i, node in enumerate(self.nodes):
            sent = node.stats[self.HIGH_PRIO]['sent'] + node.stats[self.LOW_PRIO]['sent']
            collisions = node.stats[self.HIGH_PRIO]['collisions'] + node.stats[self.LOW_PRIO]['collisions']
            cur_stats.append({'sent': sent, 'collisions': collisions})
            
            # Check if sent count or collision count increased
            if sent > pre_stats[i]['sent'] or collisions > pre_stats[i]['collisions']:
                did_transmit[i] = True
                self.node_last_tx_step[i] = self.current_step
        
        # Calculate centralized adjustments
        central_adjustments = {agent: 0.0 for agent in self.agents}
        
        for i in range(self.n_nodes):
            if did_transmit[i]:
                # Node i transmitted: Reward others
                for j, agent in enumerate(self.agents):
                    if i == j: 
                        continue
                    if cur_stats[j]['sent'] > pre_stats[j]['sent']:
                        central_adjustments[agent] += CENTRAL_TX_REWARD
                    if cur_stats[j]['collisions'] > pre_stats[j]['collisions']:
                        central_adjustments[agent] -= CENTRAL_COLLISION_PENALTY
            else:
                # Node i was idle: Penalize others based on idle duration
                # idle_duration is calculated from current step
                idle_duration = self.current_step - self.node_last_tx_step[i]
                penalty = CENTRAL_IDLE_PENALTY_BASE + (idle_duration * CENTRAL_IDLE_PENALTY_FACTOR)
                
                for j, agent in enumerate(self.agents):
                    if i != j:
                        central_adjustments[agent] -= penalty
                        
        # Apply adjustments to rewards
        for agent in self.agents:
            if agent in rewards:
                rewards[agent] += central_adjustments[agent]
                
        return observations, rewards, terminations, truncations, infos
