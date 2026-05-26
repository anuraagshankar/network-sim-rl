# Single agent reward configuration

SUCCESS_REWARD_SINGLE = 10.0
HIGH_PRIORITY_BONUS_SINGLE = 5.0

# Multi-agent reward configuration

# Success rewards
SUCCESS_REWARD = 1.0

# Priority bonuses
HIGH_PRIORITY_BONUS = 0.1

# Penalties
COLLISION_PENALTY = 1.0
LATENCY_PENALTY_COEFF = 0
QUEUE_PENALTY_COEFF = 0

# Centralized reward configuration
# Cooperative shaping added per slot (see CentralizedRewardParallelEnv):
#   TX_REWARD          : credit each node when another node succeeds (share the channel)
#   COLLISION_PENALTY  : penalize others when a node collides (discourage piling on)
#   IDLE_PENALTY_*     : penalize others while a node sits idle (discourage free-riding);
#                        kept small so it breaks the silence equilibrium without
#                        suppressing throughput (large values made agents over-cautious).
CENTRAL_TX_REWARD = 0.5
CENTRAL_COLLISION_PENALTY = 1.0
CENTRAL_IDLE_PENALTY_BASE = 0.05
CENTRAL_IDLE_PENALTY_FACTOR = 0.02
