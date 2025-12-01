import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from wn_env import WirelessNetworkParallelEnv


class ContextualAgent:
    """Simple contextual bandit with epsilon-greedy exploration."""

    def __init__(self, n_states, n_arms, epsilon=1.0):
        self.n_states = n_states
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros((n_states, n_arms))
        self.counts = np.zeros((n_states, n_arms))

    def select(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.q_values[state]))

    def update(self, state, arm, reward):
        self.counts[state, arm] += 1
        alpha = 1.0 / self.counts[state, arm]
        self.q_values[state, arm] += alpha * (reward - self.q_values[state, arm])


class ChannelPolicy(nn.Module):
    """Two-layer MLP that outputs channel logits."""

    def __init__(self, input_dim, n_channels, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
        )

    def forward(self, x):
        return self.model(x)


class REINFORCEChannelAgent:
    """
    Policy-gradient (REINFORCE) channel selector.
    Takes the raw collision history vector as state.
    """

    def __init__(
        self,
        n_channels,
        hidden_dim=64,
        lr=1e-3,
        entropy_coef=0.01,
        baseline_momentum=0.9,
        max_grad_norm=1.0,
    ):
        self.n_channels = n_channels
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.policy = ChannelPolicy(n_channels, n_channels, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.baseline_momentum = baseline_momentum
        self.baseline = 0.0
        self.max_grad_norm = max_grad_norm

    def select_action(self, collision_history, deterministic=False):
        """
        Args:
            collision_history: numpy array of shape (n_channels,)
            deterministic: when True, pick argmax channel (used for evaluation)
        Returns:
            action (int), trace (dict or None)
        """
        state = np.asarray(collision_history, dtype=np.float32).copy()
        state_tensor = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            logits = self.policy(state_tensor)
            if deterministic:
                action = int(torch.argmax(logits).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

        trace = None if deterministic else {"state": state, "action": action}
        return action, trace

    def update(self, trace, reward):
        if trace is None:
            return

        state_tensor = torch.tensor(trace["state"], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(trace["action"], dtype=torch.long, device=self.device)

        logits = self.policy(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()

        reward = float(reward)
        self.baseline = self.baseline_momentum * self.baseline + (1 - self.baseline_momentum) * reward
        advantage = reward - self.baseline

        loss = -advantage * log_prob - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()


class NodeAgent:
    """Combines contextual bandits for queue/backoff with an MLP channel policy."""

    def __init__(self, n_channels, max_backoff, epsilon=1.0, hidden_dim=64, lr=1e-3):
        self.queue_agent = ContextualAgent(4, 2, epsilon)
        self.backoff_agent = ContextualAgent(2, max_backoff + 1, epsilon)
        self.channel_agent = REINFORCEChannelAgent(
            n_channels=n_channels,
            hidden_dim=hidden_dim,
            lr=lr,
        )

    def set_epsilon(self, epsilon):
        self.queue_agent.epsilon = epsilon
        self.backoff_agent.epsilon = epsilon

    def select_action(self, obs, n_channels, deterministic=False):
        queue_state = self._get_queue_state(obs)
        backoff_state = self._get_backoff_state(obs)
        collision_history = obs[-n_channels:]

        channel_action, channel_trace = self.channel_agent.select_action(
            collision_history, deterministic=deterministic
        )

        action = {
            "queue_selection": self.queue_agent.select(queue_state),
            "channel_selection": channel_action,
            "backoff_value": self.backoff_agent.select(backoff_state),
        }

        internal_state = {
            "queue_state": queue_state,
            "backoff_state": backoff_state,
            "channel_trace": channel_trace,
        }

        return action, internal_state

    def update(self, internal_state, action, reward):
        self.queue_agent.update(internal_state["queue_state"], action["queue_selection"], reward)
        self.backoff_agent.update(internal_state["backoff_state"], action["backoff_value"], reward)
        self.channel_agent.update(internal_state["channel_trace"], reward)

    @staticmethod
    def _get_queue_state(obs):
        high_empty = 1 if obs[0] == 0 else 0
        low_empty = 1 if obs[1] == 0 else 0
        return high_empty * 2 + low_empty

    @staticmethod
    def _get_backoff_state(obs):
        return 0 if obs[2] == 0 else 1


CONFIG_NAME = "multi_node_competitive"


def train_agents(n_episodes=200, epsilon_start=1.0, epsilon_end=0.05):
    env = WirelessNetworkParallelEnv(config_name=CONFIG_NAME)
    observations, infos = env.reset()

    agents = {
        agent_name: NodeAgent(env.n_channels, env.max_backoff, epsilon_start)
        for agent_name in env.possible_agents
    }

    for episode in range(n_episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / max(1, n_episodes - 1))
        epsilon = max(epsilon_end, epsilon)
        for agent in agents.values():
            agent.set_epsilon(epsilon)

        episode_rewards = {agent_name: 0.0 for agent_name in env.possible_agents}
        observations, infos = env.reset()

        while True:
            actions = {}
            states = {}
            for agent_name in env.agents:
                action, internal_state = agents[agent_name].select_action(
                    observations[agent_name], env.n_channels, deterministic=False
                )
                actions[agent_name] = action
                states[agent_name] = internal_state

            observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent_name in env.agents:
                reward = rewards[agent_name]
                agents[agent_name].update(states[agent_name], actions[agent_name], reward)
                episode_rewards[agent_name] += reward

            if all(truncations.values()) or all(terminations.values()):
                print(f"Episode {episode} rewards: {episode_rewards}")
                break

    return agents


def evaluate_agents(agents, max_steps=100):
    env = WirelessNetworkParallelEnv(config_name=CONFIG_NAME, render_mode="human")
    env.max_steps = max_steps

    for agent in agents.values():
        agent.set_epsilon(0.0)

    observations, infos = env.reset()

    while True:
        actions = {}
        for agent_name in env.agents:
            action, _ = agents[agent_name].select_action(
                observations[agent_name], env.n_channels, deterministic=True
            )
            actions[agent_name] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if all(truncations.values()) or all(terminations.values()):
            break

    metrics = infos[env.agents[0]]["node_latency_throughput"]
    print("Throughput and latency metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    trained_agents = train_agents()
    evaluate_agents(trained_agents)


