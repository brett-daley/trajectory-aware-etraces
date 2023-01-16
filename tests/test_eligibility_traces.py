import itertools

import gym_classics
import numpy as np

from trajectory_aware_etraces import EnvSampler

# Change this import statement to test a different method:
from trajectory_aware_etraces import RBIS as etrace_cls


def test_online(env_id, behavior_policy, target_policy, etraces, n_episodes):
    sampler = EnvSampler(env_id, seed=0)
    env = sampler.env
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Sample episodes from the environment
    for _ in range(n_episodes):
        states = []
        actions = []
        done = False
        while not done:
            s, a, reward, next_state, done = sampler.step(lambda s: behavior_policy)
            states.append(s)
            actions.append(a)
            # For simplicity, just use the rewards for the TD errors (bootstrapping disabled)
            td_error = reward

            # Calculate online update
            updates = etraces.step(td_error, behavior_policy[a], target_policy[a], done)
            # Apply update (assumes alpha=1)
            sa_pairs = (states, actions)
            np.add.at(Q, sa_pairs, updates)

    print(Q)


def test_offline(env_id, behavior_policy, target_policy, etraces, n_episodes):
    sampler = EnvSampler(env_id, seed=0)
    env = sampler.env
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Sample episodes from the environment
    episodes = sampler.sample_episodes(lambda s: behavior_policy, n_episodes)
    transitions = tuple(itertools.chain(*episodes))

    # Convert transitions to numpy arrays
    states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))
    behavior_probs = behavior_policy[actions]
    target_probs = target_policy[actions]
    # For simplicity, just use the rewards for the TD errors (bootstrapping disabled)
    td_errors = rewards.copy()

    # Calculate offline updates
    updates = etraces.trajectory(td_errors, behavior_probs, target_probs, dones)
    # Apply updates (assumes alpha=1)
    sa_pairs = (states, actions)
    np.add.at(Q, sa_pairs, updates)
    print(Q)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    n_episodes = 2
    behavior_policy = np.array([0.5, 0.5])
    target_policy = behavior_policy.copy()

    # Test case 1: Online updates. Since we disabled bootstrapping, this
    # should produce the same results as offline updates.
    etraces = etrace_cls(discount=1.0, lambd=0.9)
    print("Online:")
    test_online("19Walk-v0", behavior_policy, target_policy, etraces, n_episodes)

    print('---')

    # Test case 2: Offline updates.
    etraces = etrace_cls(discount=1.0, lambd=0.9)
    print("Offline:")
    test_offline("19Walk-v0", behavior_policy, target_policy, etraces, n_episodes)
