import gym_classics
import numpy as np

from trajectory_aware_etraces.experiments.sampling import EnvSampler

# Change these import statements to test a different method:
from trajectory_aware_etraces.etraces.offline import RBIS as offline_etrace_cls
from trajectory_aware_etraces.etraces.online import RBIS as online_etrace_cls


def test_online(env_id, behavior_policy, target_policy, etraces):
    sampler = EnvSampler(env_id, seed=0)
    env = sampler.env

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    etraces.set(Q, alpha=1)

    # Sample one episode from the environment
    done = False
    while not done:
        s, a, reward, next_state, done = sampler.step(lambda s: behavior_policy)
        # For simplicity, just use the rewards for the TD errors (bootstrapping disabled)
        td_error = reward

        etraces.step(s, a, td_error, behavior_policy[a], target_policy[a])

    print(Q)


def test_offline(env_id, behavior_policy, target_policy, etraces):
    sampler = EnvSampler(env_id, seed=0)
    env = sampler.env
    # Sample one episode from the environment
    episode = sampler.sample_one_episode(lambda s: behavior_policy)

    # Convert transitions to numpy arrays
    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    # For simplicity, just use the rewards for the TD errors (bootstrapping disabled)
    td_errors = rewards.copy()

    updates = etraces(td_errors, behavior_policy[actions], target_policy[actions], dones)

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    sa_pairs = (states, actions)
    np.add.at(Q, sa_pairs, updates)
    print(Q)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    behavior_policy = np.array([0.5, 0.5])
    target_policy = behavior_policy.copy()

    # Test case 1: Online updates. Since we disabled bootstrapping, this
    # should produce the same results as offline updates.
    etraces = online_etrace_cls(discount=1.0, lambd=0.9)
    print("Online:")
    test_online("19Walk-v0", behavior_policy, target_policy, etraces)

    print('---')

    # Test case 2: Offline updates.
    etraces = offline_etrace_cls(discount=1.0, lambd=0.9)
    print("Offline:")
    test_offline("19Walk-v0", behavior_policy, target_policy, etraces)
