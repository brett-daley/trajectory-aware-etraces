import sys
sys.path.append('..')

import gym
import gym_classics
import numpy as np

from dqn.eligibility_traces import *


def policy_evaluation_Q(env, discount, policy, precision=1e-3):
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


def policy_evaluation_V(env, discount, policy, precision=1e-3):
    Q = policy_evaluation_Q(env, discount, policy, precision)
    return policy[None] @ Q.T


def backup(env, discount, policy, Q, state, action):
    next_states, rewards, dones, probs = env.model(state, action)
    next_V = (policy[None] @ Q[next_states].T)[0]
    next_V *= (1.0 - dones)
    return np.sum(probs * (rewards + discount * next_V))


def sample_episodes(env_id, behavior_policy, n_episodes, seed):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)
    random_state = np.random.RandomState(seed)

    episodes = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        transitions = []
        while not done:
            action = random_state.choice(env.action_space.n, p=behavior_policy)
            next_state, reward, done, _ = env.step(action)
            transitions.append( (state, action, reward, next_state, done) )
            state = next_state if not done else env.reset()
        episodes.append(tuple(transitions))
    return env, tuple(episodes)


def train_V(V, episode, behavior_policy, target_policy, discount, etrace, learning_rate):
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0

    # Accumulate updates using the eligibility trace
    for (s, a, reward, ns, done) in episode:
        td_error = reward - V[s]
        if not done:
            td_error += discount * V[ns]
        etrace.update(td_error, target_policy[a], behavior_policy[a], done)

    # Now apply the updates to each visited state-action pair
    updates = etrace.get_updates_and_reset()
    for t, (s, _, _, _, _) in enumerate(episode):
        updates[t] += V[s]
    for t, (s, _, _, _, _) in enumerate(episode):
        V[s] += learning_rate * (updates[t] - V[s])


def train_Q(Q, episode, behavior_policy, target_policy, discount, etrace, learning_rate):
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0

    # Accumulate updates using the eligibility trace
    for (s, a, reward, ns, done) in episode:
        td_error = reward - Q[s, a]
        if not done:
            td_error += discount * (target_policy * Q[ns]).sum()
        etrace.update(td_error, target_policy[a], behavior_policy[a], done)

    # Now apply the updates to each visited state-action pair
    updates = etrace.get_updates_and_reset()
    for t, (s, a, _, _, _) in enumerate(episode):
        updates[t] += Q[s, a]
    for t, (s, a, _, _, _) in enumerate(episode):
        Q[s, a] += learning_rate * (updates[t] - Q[s, a])


def rms(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def random_walk_experiment(lambd, etrace_cls, learning_rate, seed):
    behavior_policy = np.array([0.5, 0.5])
    target_policy = np.array([0.5, 0.5])
    discount = 1.0

    env, episodes = sample_episodes('19Walk-v0', behavior_policy, n_episodes=10, seed=seed)

    etrace = etrace_cls(discount, lambd, maxlen=10_000)

    # TODO: We don't want to recompute this every time
    V_pi = policy_evaluation_V(env, discount, target_policy, precision=1e-6)

    V = np.zeros(env.observation_space.n)
    rms_errors = []
    for e in episodes:
        train_V(V, e, behavior_policy, target_policy, discount, etrace, learning_rate)
        rms_errors.append(rms(V, V_pi))
    return rms_errors


if __name__ == '__main__':
    return_estimators = [Qlambda]
    lambda_values = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
    learning_rates = np.linspace(0.0, 1.0, 10 + 1)

    for estimator in return_estimators:
        print(estimator)
        for lambd in lambda_values:
            for lr in learning_rates:
                rms_errors = []
                for seed in range(10):
                    rms_errors.extend(
                        random_walk_experiment(lambd, estimator, learning_rate=lr, seed=seed)
                    )
                avg_error = np.mean(rms_errors)
                print('{:.2f}'.format(lr), '{:.2f}'.format(lambd), avg_error)
            print()
        print()
