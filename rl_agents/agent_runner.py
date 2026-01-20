from envs import WirelessNetworkEnv, WirelessNetworkParallelEnv, CentralizedRewardParallelEnv

# Configuration
ENV_CLASS = CentralizedRewardParallelEnv
CONFIG_NAME = "scenario_2_three_agents_two_channels"
RENDER_MODE = "human"
TRAIN_EPS = 500
TRAIN_MAX_STEPS = 1000
TEST_MAX_STEPS = 100
