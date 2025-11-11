"""
Network Configuration for Wireless Network RL Environment
This file contains all network-specific parameters and configurations.
"""
import numpy as np

# ==================================================================
# --- NETWORK PARAMETERS ---
# ==================================================================

# Network Size
N_NODES = 2          # Number of nodes (transmitter/receiver pairs)
C_CHANNELS = 4       # Number of available communication channels
MAX_EPISODE_STEPS = 20000  # Maximum steps per episode

# Queue Parameters
Q_LIMIT = 10         # Maximum number of packets allowed in each queue

# Time Parameters
T_s = 1.0            # Duration of a single time slot in milliseconds

# QoS Levels
HIGH_PRIO = 0        # Identifier for High Priority traffic
LOW_PRIO = 1         # Identifier for Low Priority traffic
QOS_LEVELS = [HIGH_PRIO, LOW_PRIO]

# Fixed Policy Parameters (for non-learning nodes)
FIXED_BACKOFF_WINDOW = 7  # Max backoff window size for non-learning nodes

# ==================================================================
# --- SCENARIO GENERATION ---
# ==================================================================

def generate_random_scenario(n_nodes, c_channels, seed=None):
    """
    Generates random traffic loads and channel interference patterns.
    
    Args:
        n_nodes: Number of nodes in the network
        c_channels: Number of channels
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Dictionary with scenario parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random Packet Arrival Rates
    # High Priority traffic arrives less frequently
    p_arrivals_high = np.random.uniform(0.005, 0.01, n_nodes)
    # Low Priority traffic arrives more frequently
    p_arrivals_low = np.random.uniform(0.01, 0.05, n_nodes)
    
    # Generate random External Interference Rates (GAMMA)
    # Approximately 25% of channels are designated as "clean" (0% interference)
    num_clean = c_channels // 4
    num_noisy = c_channels - num_clean
    gamma_clean = np.zeros(num_clean)
    # Remaining channels have random interference levels
    gamma_noisy = np.random.uniform(0.0, 0.3, num_noisy)
    gamma = np.concatenate([gamma_clean, gamma_noisy])
    np.random.shuffle(gamma)
    
    return {
        'p_arrivals_high': p_arrivals_high,
        'p_arrivals_low': p_arrivals_low,
        'gamma': gamma
    }

