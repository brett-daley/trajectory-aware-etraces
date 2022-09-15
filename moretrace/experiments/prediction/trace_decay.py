from argparse import ArgumentParser
import sys
sys.path.append('..')

import gym
import gym_classics
import numpy as np


def sample_episode(env_id, behavior_policy, seed):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    random_state = np.random.RandomState(seed)

    state = env.reset()
    done = False
    transitions = []
    while not done:
        action = random_state.choice(env.action_space.n, p=behavior_policy)
        next_state, reward, done, _ = env.step(action)
        transitions.append( (state, action, reward, next_state, done) )
        state = next_state
    return tuple(transitions)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('estimator', type=str)
    parser.add_argument('lambd', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('--discount', type=float, default=1.0)
    args = parser.parse_args()

    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.1, 0.9])
    episode = sample_episode('19Walk-v0', behavior_policy, args.seed)

    # NOTE: Need to edit the e-trace classes to make sure they print out the data,
    # otherwise this script won't do anything
    etrace = getattr(eligibility_traces, args.estimator)(args.discount, args.lambd)

    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    # Just use zeros for the TD errors since we don't care whether the values are correct
    td_errors = np.zeros_like(rewards)
    updates = etrace(td_errors, target_policy[actions], behavior_policy[actions], dones)
