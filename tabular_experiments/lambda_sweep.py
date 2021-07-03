import sys
sys.path.append('..')

import gym
import gym_classics
from gym_classics.dynamic_programming import policy_evaluation
import numpy as np

from dqn.experience_replay.traces import get_trace_function


def policy_evaluation(env, discount, policy, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)

    while True:
        Q_old = Q.copy()

        for s in env.states():
            for a in env.actions():
                Q[s, a] = backup(env, discount, policy, Q, s, a)

        if np.abs(Q - Q_old).max() <= precision:
            return Q


def backup(env, discount, policy, Q, state, action):
    next_states, rewards, dones, probs = env.model(state, action)
    next_V = (policy[None] @ Q[next_states].T)[0]
    next_V *= (1.0 - dones)
    return np.sum(probs * (rewards + discount * next_V))


def sample_episodes(env_id, behavior_policy, n, seed):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    random_state = np.random.RandomState(seed)

    transitions = []
    for _ in range(n):
        state = env.reset()
        done = False
        while not done:
            action = random_state.choice(env.action_space.n, p=behavior_policy)
            next_state, reward, done, _ = env.step(action)
            transitions.append( (state, action, reward, next_state, done) )
            state = next_state if not done else env.reset()
    return env, tuple(transitions)


def run_trial(Q, experience, behavior_policy, target_policy, discount, trace_function,
              learning_rate):
    eligibility = np.zeros_like(Q)
    for (s, a, reward, ns, done) in experience:
        td_error = reward - Q[s, a]
        if not done:
            td_error += discount * (target_policy * Q[ns]).sum()

        trace = trace_function(a, target_policy, behavior_policy)
        eligibility *= discount * trace
        eligibility[s, a] += 1.0
        Q += learning_rate * td_error * eligibility
        if done:
            eligibility *= 0.0


def rms(Q1, Q2):
    return np.sqrt(np.mean(np.square(Q1 - Q2)))


def main():
    behavior_policy = np.array([0.25, 0.25, 0.25, 0.25])
    target_policy = np.array([0.25, 0.25, 0.25, 0.25])
    env, experience = sample_episodes('ClassicGridworld-v0', behavior_policy, n=1000, seed=0)

    discount = 0.9
    trace_function = get_trace_function('IS', lambd=0.0)
    learning_rate = 0.1

    Q_pi = policy_evaluation(env, discount, target_policy, precision=1e-6)
    print(np.round(np.mean(Q_pi, axis=1), 2))

    Q = np.zeros([env.observation_space.n, env.action_space.n])
    run_trial(Q, experience, behavior_policy, target_policy, discount, trace_function, learning_rate)
    print(np.round(np.mean(Q, axis=1), 2))
    print(rms(Q, Q_pi))


if __name__ == '__main__':
    main()
