import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from network_configs.config_loader import ConfigLoader
import json
from datetime import datetime
import os
from envs.single_agent_env import Packet, Node


class NonstationaryWirelessNetworkEnv(ParallelEnv):
    """
    PettingZoo parallel environment for wireless network with non-stationarity.
    Nodes can enter and leave the network dynamically.
    """
    
    metadata = {'render_modes': ['human'], 'name': 'wireless_network_nonstationary_v0'}
    
    def __init__(self, config_name=None, seed=None, render_mode=None):
        """Initialize the non-stationary wireless network environment."""
        super().__init__()
        
        if config_name is None:
            raise ValueError("config_name required for non-stationary environment")
        
        self.config = ConfigLoader.load(config_name)
        self.n_nodes = self.config.n_nodes
        self.n_channels = self.config.n_channels
        self.max_backoff = self.config.max_backoff
        self.queue_limit = self.config.queue_limit
        self.max_steps = self.config.max_steps
        self.t_s = self.config.t_s
        
        ns_config = self.config.raw_config.get('non_stationary', {})
        if not ns_config.get('enabled', False):
            raise ValueError("Non-stationary must be enabled in config")
        
        self.n_fixed_nodes = self.n_nodes - ns_config['nodes']
        self.n_nonstationary_nodes = ns_config['nodes']
        self.node_entry_prob = np.array(ns_config['node_entry_prob'])
        self.node_exit_prob = np.array(ns_config['node_exit_prob'])
        self.ns_high_priority = np.array(ns_config['high_priority'])
        self.ns_low_priority = np.array(ns_config['low_priority'])
        
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
            2 +
            1 +
            1 +
            self.n_channels +
            self.n_channels
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
        self.node_active = None
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
        
        self.node_active = np.zeros(self.n_nodes, dtype=bool)
        self.node_active[:self.n_fixed_nodes] = True
        
        fixed_high, fixed_low = self.config.generate_arrival_rates()
        self.p_arrivals_high = np.zeros(self.n_nodes)
        self.p_arrivals_low = np.zeros(self.n_nodes)
        self.p_arrivals_high[:self.n_fixed_nodes] = fixed_high
        self.p_arrivals_low[:self.n_fixed_nodes] = fixed_low
        self.p_arrivals_high[self.n_fixed_nodes:] = self.ns_high_priority
        self.p_arrivals_low[self.n_fixed_nodes:] = self.ns_low_priority
        
        self.gamma = self.config.generate_channel_interference()
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
                'gamma': self.gamma.tolist(),
                'n_fixed_nodes': self.n_fixed_nodes
            })
        
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: self._get_info(i) for i, agent in enumerate(self.agents)}
        
        return observations, infos
    
    def _update_node_states(self):
        """Update which nodes are active based on entry/exit probabilities."""
        for i in range(self.n_fixed_nodes, self.n_nodes):
            ns_idx = i - self.n_fixed_nodes
            if self.node_active[i]:
                if np.random.rand() < self.node_exit_prob[ns_idx]:
                    self.node_active[i] = False
                    self.nodes[i].packet_queue_high.clear()
                    self.nodes[i].packet_queue_low.clear()
                    self.nodes[i].backoff_timer = 0
            else:
                if np.random.rand() < self.node_entry_prob[ns_idx]:
                    self.node_active[i] = True
    
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
            'collisions': sum(node.stats[q]['collisions'] for q in [0, 1]),
            'active': bool(self.node_active[node_id])
        }
    
    def step(self, actions):
        """Execute one time step with all agents acting in parallel."""
        from envs.reward_config import (
            SUCCESS_REWARD, HIGH_PRIORITY_BONUS,
            COLLISION_PENALTY, LATENCY_PENALTY_COEFF, QUEUE_PENALTY_COEFF
        )
        
        self._update_node_states()
        
        arrivals = {}
        for k, node in enumerate(self.nodes):
            arrivals[k] = {'high': False, 'low': False, 'dropped_high': False, 'dropped_low': False}
            if not self.node_active[k]:
                continue
            
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
        
        external_busy = [np.random.rand() < g for g in self.gamma]
        
        transmissions = [[] for _ in range(self.n_channels)]
        actions_taken = {}
        
        for i, agent in enumerate(self.agents):
            if not self.node_active[i]:
                continue
            
            node = self.nodes[i]
            action = actions[agent]
            
            queue_sel = action['queue_selection']
            channel_sel = action['channel_selection']
            backoff = action['backoff_value']
            
            if node.backoff_timer > 0:
                node.backoff_timer -= 1
            
            if node.backoff_timer == 0:
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
        
        successful_tx = {}
        rewards = {agent: 0.0 for agent in self.agents}
        self.collision_history *= 0.9
        
        for j in range(self.n_channels):
            num_tx = len(transmissions[j])
            is_external = external_busy[j]
            
            if num_tx == 1:
                node_id = transmissions[j][0]
                if not is_external:
                    _, qos = actions_taken[node_id]
                    successful_tx[node_id] = qos
                    
                    node = self.nodes[node_id]
                    queue = (node.packet_queue_high if qos == self.HIGH_PRIO 
                            else node.packet_queue_low)
                    packet = queue.popleft()
                    latency = (self.current_step + 1 - packet.arrival_slot) * self.t_s
                    
                    node.stats[qos]['sent'] += 1
                    node.stats[qos]['latency'] += latency
                    
                    agent_name = self.agents[node_id]
                    rewards[agent_name] += SUCCESS_REWARD
                    rewards[agent_name] -= latency * LATENCY_PENALTY_COEFF
                    if qos == self.HIGH_PRIO:
                        rewards[agent_name] += HIGH_PRIORITY_BONUS
                else:
                    node_id = transmissions[j][0]
                    agent_name = self.agents[node_id]
                    rewards[agent_name] -= COLLISION_PENALTY
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    self.collision_history[j] = 1.0
            
            elif num_tx > 1:
                for node_id in transmissions[j]:
                    agent_name = self.agents[node_id]
                    self.nodes[node_id].stats[actions_taken[node_id][1]]['collisions'] += 1
                    rewards[agent_name] -= COLLISION_PENALTY
                self.collision_history[j] = 1.0
        
        for i, agent in enumerate(self.agents):
            if not self.node_active[i]:
                continue
            node = self.nodes[i]
            queue_occupancy = (len(node.packet_queue_high) + 
                              len(node.packet_queue_low)) / (2 * self.queue_limit)
            rewards[agent] -= queue_occupancy * QUEUE_PENALTY_COEFF
        
        if self.render_mode is not None:
            step_data = {
                'type': 'step',
                'step': int(self.current_step),
                'node_active': [bool(x) for x in self.node_active],
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
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        
        if all(truncations.values()) and self.render_mode is not None:
            self._save_replay()
            if self.render_mode == 'human':
                self._render_replay()
        
        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        infos = {agent: self._get_info(i) for i, agent in enumerate(self.agents)}
        
        return observations, rewards, terminations, truncations, infos
    
    def _save_replay(self):
        """Save replay log to file."""
        os.makedirs('replays', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.replay_file = f'replays/replay_ns_{timestamp}.json'
        with open(self.replay_file, 'w', encoding='utf-8') as f:
            json.dump(self.replay_log, f, indent=2)
    
    def _render_replay(self):
        """Render the replay using the custom renderer."""
        if self.replay_file:
            from envs.replay_renderer import render_nonstationary_replay
            render_nonstationary_replay(self.replay_file)
    
    def render(self):
        """Render the environment."""
        return None
    
    def state(self):
        """Return global state."""
        return None


if __name__ == '__main__':
    print("Running non-stationary environment demo...")
    
    env = NonstationaryWirelessNetworkEnv(
        config_name='simple_nonstationary',
        seed=42,
        render_mode='human'
    )
    env.max_steps = 100
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done:
        actions = {}
        for agent in env.agents:
            actions[agent] = {
                'queue_selection': np.random.randint(0, 2),
                'channel_selection': np.random.randint(0, env.n_channels),
                'backoff_value': np.random.randint(0, env.max_backoff + 1)
            }
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        if step % 10 == 0:
            active_nodes = [i for i, active in enumerate(env.node_active) if active]
            print(f"Step {step}: Active nodes: {active_nodes}")
        
        done = all(truncs.values()) or all(terms.values())
        step += 1
    
    print("Demo completed!")

