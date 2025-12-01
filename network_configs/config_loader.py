"""
Configuration loader for wireless network environments.
Handles loading and parsing network configuration files.
"""
import json
import os
import numpy as np
from typing import Dict, Any, Tuple


class NetworkConfig:
    """Stores and manages network configuration parameters."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize network configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        self.raw_config = config_dict
        self.name = config_dict.get('name', 'Unnamed Configuration')
        self.description = config_dict.get('description', '')
        
        # Network parameters
        network = config_dict.get('network', {})
        self.n_nodes = network.get('n_nodes', 2)
        self.n_channels = network.get('n_channels', 4)
        self.max_backoff = network.get('max_backoff', 7)
        self.queue_limit = network.get('queue_limit', 10)
        self.max_steps = network.get('max_steps', 1000)
        self.t_s = network.get('t_s', 1.0)
        
        # Arrival rates configuration
        self.arrival_config = config_dict.get('arrival_rates', {})
        
        # Channel interference configuration
        self.interference_config = config_dict.get('channel_interference', {})
    
    def generate_arrival_rates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate arrival rates for high and low priority packets.
        
        Returns:
            Tuple of (high_priority_rates, low_priority_rates) as numpy arrays
        """
        if self.arrival_config.get('type') == 'random':
            # Random generation
            high_cfg = self.arrival_config.get('high_priority', {})
            low_cfg = self.arrival_config.get('low_priority', {})
            
            p_high = np.random.uniform(
                high_cfg.get('min', 0.005),
                high_cfg.get('max', 0.01),
                self.n_nodes
            )
            p_low = np.random.uniform(
                low_cfg.get('min', 0.01),
                low_cfg.get('max', 0.05),
                self.n_nodes
            )
        else:
            # Fixed values from config
            p_high = np.array(self.arrival_config.get('high_priority', [0.01] * self.n_nodes))
            p_low = np.array(self.arrival_config.get('low_priority', [0.01] * self.n_nodes))
            
            # Ensure arrays match n_nodes
            if len(p_high) < self.n_nodes:
                # Extend with last value if too short
                p_high = np.pad(p_high, (0, self.n_nodes - len(p_high)), 
                               mode='edge')
            elif len(p_high) > self.n_nodes:
                p_high = p_high[:self.n_nodes]
            
            if len(p_low) < self.n_nodes:
                p_low = np.pad(p_low, (0, self.n_nodes - len(p_low)), 
                              mode='edge')
            elif len(p_low) > self.n_nodes:
                p_low = p_low[:self.n_nodes]
        
        return p_high, p_low
    
    def generate_channel_interference(self) -> np.ndarray:
        """
        Generate channel interference rates.
        
        Returns:
            Numpy array of interference rates per channel
        """
        if self.interference_config.get('type') == 'random':
            # Random generation with clean/noisy split
            clean_ratio = self.interference_config.get('clean_channels_ratio', 0.25)
            num_clean = max(1, int(self.n_channels * clean_ratio))
            num_noisy = self.n_channels - num_clean
            
            noisy_range = self.interference_config.get('noisy_range', {})
            gamma_clean = np.zeros(num_clean)
            gamma_noisy = np.random.uniform(
                noisy_range.get('min', 0.0),
                noisy_range.get('max', 0.3),
                num_noisy
            )
            gamma = np.concatenate([gamma_clean, gamma_noisy])
            np.random.shuffle(gamma)
        else:
            # Fixed values from config
            gamma = np.array(self.interference_config.get('values', [0.0] * self.n_channels))
            
            # Ensure array matches n_channels
            if len(gamma) < self.n_channels:
                gamma = np.pad(gamma, (0, self.n_channels - len(gamma)), mode='edge')
            elif len(gamma) > self.n_channels:
                gamma = gamma[:self.n_channels]
        
        return gamma
    
    def __str__(self):
        """String representation of configuration."""
        return f"NetworkConfig(name='{self.name}', nodes={self.n_nodes}, channels={self.n_channels})"
    
    def __repr__(self):
        return self.__str__()


class ConfigLoader:
    """Loads network configurations from JSON files."""
    
    # Default config directory
    DEFAULT_CONFIG_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'configs'
    )
    
    # Default config file
    DEFAULT_CONFIG = 'default.json'
    
    @classmethod
    def load(cls, config_name: str = None, config_dir: str = None) -> NetworkConfig:
        """
        Load a network configuration from file.
        
        Args:
            config_name: Name of config file (with or without .json extension).
                        If None, loads default config.
            config_dir: Directory containing config files. If None, uses default.
        
        Returns:
            NetworkConfig object
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if config_dir is None:
            config_dir = cls.DEFAULT_CONFIG_DIR
        
        if config_name is None:
            config_name = cls.DEFAULT_CONFIG
        
        # Add .json extension if not present
        if not config_name.endswith('.json'):
            config_name = f"{config_name}.json"
        
        config_path = os.path.join(config_dir, config_name)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Available configs in {config_dir}: {cls.list_configs(config_dir)}"
            )
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        
        return NetworkConfig(config_dict)
    
    @classmethod
    def list_configs(cls, config_dir: str = None) -> list:
        """
        List available configuration files.
        
        Args:
            config_dir: Directory to search. If None, uses default.
        
        Returns:
            List of config file names (without .json extension)
        """
        if config_dir is None:
            config_dir = cls.DEFAULT_CONFIG_DIR
        
        if not os.path.exists(config_dir):
            return []
        
        configs = []
        for filename in os.listdir(config_dir):
            if filename.endswith('.json'):
                configs.append(filename[:-5])  # Remove .json extension
        
        return sorted(configs)
    
    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]) -> NetworkConfig:
        """
        Create a NetworkConfig directly from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        
        Returns:
            NetworkConfig object
        """
        return NetworkConfig(config_dict)


if __name__ == "__main__":
    # Test config loader
    print("=== Network Configuration Loader Test ===\n")
    
    # List available configs
    print("Available configurations:")
    configs = ConfigLoader.list_configs()
    for config in configs:
        print(f"  - {config}")
    print()
    
    # Load and display simple network config
    print("Loading 'simple_network' configuration...")
    config = ConfigLoader.load('simple_network')
    print(f"Config: {config}")
    print(f"Description: {config.description}")
    print(f"Nodes: {config.n_nodes}")
    print(f"Channels: {config.n_channels}")
    print()
    
    # Generate parameters
    print("Generating parameters from config:")
    p_high, p_low = config.generate_arrival_rates()
    gamma = config.generate_channel_interference()
    print(f"High priority arrival rates: {p_high}")
    print(f"Low priority arrival rates: {p_low}")
    print(f"Channel interference rates: {gamma}")
    print()
    
    # Load default config
    print("Loading 'default' configuration...")
    config = ConfigLoader.load('default')
    print(f"Config: {config}")
    print(f"Description: {config.description}")
    p_high, p_low = config.generate_arrival_rates()
    gamma = config.generate_channel_interference()
    print(f"High priority arrival rates: {p_high}")
    print(f"Low priority arrival rates: {p_low}")
    print(f"Channel interference rates: {gamma}")

