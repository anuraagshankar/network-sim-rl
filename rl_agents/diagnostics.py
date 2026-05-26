"""Diagnostic harness for the contextual-MAB agents on the wireless env.

Trains the real CentralizedRewardParallelEnv across an arrival-rate sweep and
reports, per node:
  - backoff sub-agent Q-values, visit counts, and best-vs-2nd-best separation
    per phase state (the signal that tells us whether backoff is genuinely learned
    rather than argmax-over-noise)
  - the greedy firing schedule (to eyeball interleaving)
  - per-node throughput and fairness

Run: python -m rl_agents.diagnostics

Agent classes are copied from multinode_contextual_mab because that module trains
at import time; keep them in sync.
"""
import numpy as np

from envs import CentralizedRewardParallelEnv
from network_configs.config_loader import ConfigLoader


class ContextualAgent:
    def __init__(self, n_states, n_arms, epsilon=1.0):
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
        self.q_values[state, arm] += (reward - self.q_values[state, arm]) / self.counts[state, arm]


class NodeAgent:
    def __init__(self, node_id, n_channels, max_backoff, n_agents, epsilon=1.0):
        self.node_id = node_id
        self.n_agents = n_agents
        self.queue_agent = ContextualAgent(4, 2, epsilon)
        self.channel_agent = ContextualAgent(2 ** n_channels, n_channels, epsilon)
        self.backoff_agent = ContextualAgent(n_agents, max_backoff + 1, epsilon)

    def set_epsilon(self, e):
        for ag in (self.queue_agent, self.channel_agent, self.backoff_agent):
            ag.epsilon = e

    def select_action(self, obs, n_channels):
        qs = (1 if obs[0] == 0 else 0) * 2 + (1 if obs[1] == 0 else 0)
        cs = 0
        for i, v in enumerate(obs[-n_channels:]):
            if v > 0.5:
                cs += 2 ** i
        bs = int(obs[3]) % self.n_agents
        return {
            'queue_selection': self.queue_agent.select(qs),
            'channel_selection': self.channel_agent.select(cs),
            'backoff_value': self.backoff_agent.select(bs),
        }, (qs, cs, bs)

    def update(self, states, action, reward):
        qs, cs, bs = states
        self.queue_agent.update(qs, action['queue_selection'], reward)
        self.channel_agent.update(cs, action['channel_selection'], reward)
        self.backoff_agent.update(bs, action['backoff_value'], reward)


def train(env, n_episodes, seed):
    np.random.seed(seed)
    agents = {a: NodeAgent(i, env.n_channels, env.max_backoff, len(env.possible_agents))
              for i, a in enumerate(env.possible_agents)}
    for ep in range(n_episodes):
        eps = 1.0 - (1.0 - 0.01) * ep / n_episodes
        for ag in agents.values():
            ag.set_epsilon(eps)
        obs, infos = env.reset()
        pending = {a: None for a in env.possible_agents}
        last = {a: None for a in env.possible_agents}
        rewards = {a: 0.0 for a in env.possible_agents}
        while True:
            actions = {}
            for a in env.agents:
                if infos[a]['can_act']:
                    if pending[a] is not None:
                        ps, pa = pending[a]
                        agents[a].update(ps, pa, rewards[a])
                    act, st = agents[a].select_action(obs[a], env.n_channels)
                    actions[a], last[a], pending[a] = act, act, (st, act)
                else:
                    actions[a] = last[a]
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(truncs.values()):
                break
    return agents


def evaluate(env, agents, horizon=60):
    for ag in agents.values():
        ag.set_epsilon(0.0)
    env.max_steps = horizon
    obs, infos = env.reset()
    last = {a: None for a in env.possible_agents}
    fired = {a: [] for a in env.possible_agents}
    prev = {a: 0 for a in env.possible_agents}
    step = 0
    while True:
        actions = {}
        for a in env.agents:
            if infos[a]['can_act']:
                act, _ = agents[a].select_action(obs[a], env.n_channels)
                actions[a], last[a] = act, act
            else:
                actions[a] = last[a]
        obs, r, terms, truncs, infos = env.step(actions)
        for a in env.agents:
            s = infos[a]['sent_high'] + infos[a]['sent_low']
            if s > prev[a]:
                fired[a].append(step)
            prev[a] = s
        step += 1
        if all(truncs.values()):
            break
    return fired, infos[env.possible_agents[0]]['node_latency_throughput']


def run(rate, n_episodes=400, seed=0):
    cfg = ConfigLoader.load("scenario_2_three_agents_two_channels")
    cfg.arrival_config = {"type": "fixed",
                          "high_priority": [rate] * 3, "low_priority": [rate] * 3}
    env = CentralizedRewardParallelEnv(config_name="scenario_2_three_agents_two_channels")
    env.config = cfg
    env.max_steps = 1000
    agents = train(env, n_episodes, seed)
    fired, tp = evaluate(env, agents)

    print(f"\n========== arrival rate = {rate}  (seed {seed}) ==========")
    for a in env.possible_agents:
        bo = agents[a].backoff_agent
        print(f"  {a}:")
        for s in range(len(bo.q_values)):
            q = bo.q_values[s]
            c = bo.counts[s].astype(int)
            best = int(np.argmax(q))
            order = np.sort(q)[::-1]
            sep = float(order[0] - order[1])
            print(f"    phase {s}: argmax=backoff {best}  sep={sep:+.3f}  "
                  f"Q={np.round(q, 2)}")
        print(f"    fires at: {fired[a][:10]}")
    tp_str = ', '.join(f"{k}:{v['throughput']:.3f}" for k, v in tp.items())
    total = sum(v['throughput'] for v in tp.values())
    print(f"  throughput: {{{tp_str}}}  total={total:.3f}")
    return tp


def stability_summary(rates=(0.1, 0.5, 0.9), seeds=(0, 1, 2, 3, 4), n_episodes=400):
    """Report fairness + total throughput across seeds. Stability = low variance and
    no starved node (min throughput well above 0)."""
    print("\n############ STABILITY SUMMARY (per rate, across seeds) ############")
    for rate in rates:
        totals, mins, jains = [], [], []
        for seed in seeds:
            cfg = ConfigLoader.load("scenario_2_three_agents_two_channels")
            cfg.arrival_config = {"type": "fixed",
                                  "high_priority": [rate] * 3, "low_priority": [rate] * 3}
            env = CentralizedRewardParallelEnv(config_name="scenario_2_three_agents_two_channels")
            env.config = cfg
            env.max_steps = 1000
            agents = train(env, n_episodes, seed)
            _, tp = evaluate(env, agents)
            vals = np.array([v['throughput'] for v in tp.values()])
            totals.append(vals.sum())
            mins.append(vals.min())
            # Jain's fairness index: 1.0 = perfectly fair, 1/n = maximally unfair.
            jains.append((vals.sum() ** 2) / (len(vals) * (vals ** 2).sum() + 1e-12))
        print(f"  rate={rate}: total={np.mean(totals):.3f}±{np.std(totals):.3f}  "
              f"min_node={np.mean(mins):.3f}±{np.std(mins):.3f}  "
              f"fairness(Jain)={np.mean(jains):.3f}±{np.std(jains):.3f}")


def main():
    for r in (0.1, 0.5, 0.9):
        run(r)
    stability_summary()


if __name__ == "__main__":
    main()
