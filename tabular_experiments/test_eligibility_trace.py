import itertools
import sys
sys.path.append('..')

import gym_classics
import numpy as np

# Change this import statement to test a different method:
from dqn.experience_replay.eligibility_traces import Retrace as etrace_cls
from training import sample_episodes
import walk5_dual_reward


def test(behavior_policy, target_policy, discount, lambd):
    # Sample episodes from the random walk
    env, episodes = sample_episodes('5WalkDualReward-v0', behavior_policy, n_episodes=2, seed=2)
    transitions = tuple(itertools.chain(*episodes))

    # Convert transitions to numpy arrays
    states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))
    dones = dones.astype(np.float32)

    # For simplicity, just use the rewards for the TD errors
    td_errors = rewards.copy()

    etrace = etrace_cls(discount, lambd)
    updates = etrace(td_errors, behavior_policy[actions], target_policy[actions], dones)

    print(transitions)
    print("TD error:", td_errors)
    print("IS ratio:", target_policy[actions] / behavior_policy[actions])
    print("Update:", updates)


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    # Test case 1: Check that discounting/eligibility decay are working.
    behavior_policy = np.array([1/2, 1/2])
    target_policy = np.array([1/2, 1/2])
    test(behavior_policy, target_policy, discount=0.9, lambd=0.8)

    print('---')

    # Test case 2: Check that importance sampling is working.
    behavior_policy = np.array([1/3, 2/3])
    target_policy = np.array([2/3, 1/3])
    test(behavior_policy, target_policy, discount=1.0, lambd=1.0)
