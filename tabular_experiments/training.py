from collections import defaultdict
from itertools import product
import sys
sys.path.append('..')

import gym
import numpy as np

from dqn import eligibility_traces


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
    return (policy[None] @ Q.T)[0]


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


def run_trial_V(V_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, learning_rate):
    etrace_cls = getattr(eligibility_traces, estimator)
    etrace = etrace_cls(discount, lambd, maxlen=1_000_000)

    V = np.zeros_like(V_pi)
    rms_errors = []
    for episode in experience:
        train_V(V, episode, behavior_policy, target_policy, discount, etrace, learning_rate)
        rms_errors.append(rms(V, V_pi))
    return rms_errors


def run_trial_Q(Q_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, learning_rate):
    etrace_cls = getattr(eligibility_traces, estimator)
    etrace = etrace_cls(discount, lambd, maxlen=1_000_000)

    Q = np.zeros_like(Q_pi)
    rms_errors = []
    for episode in experience:
        train_Q(Q, episode, behavior_policy, target_policy, discount, etrace, learning_rate)
        rms_errors.append(rms(Q, Q_pi))
    return rms_errors


def run_sweep_V(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds):
    assert behavior_policy.sum() == 1.0
    assert target_policy.sum() == 1.0

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    V_pi = None
    results = defaultdict(list)

    for s in seeds:
        env, experience = sample_episodes(env_id, behavior_policy, n_episodes=10, seed=s)

        if V_pi is None:
            V_pi = policy_evaluation_V(env, discount, target_policy, precision=1e-9)
            V_pi.flags.writeable = False

        for (estimator, lambd, lr) in all_combos:
            key = (estimator, lambd, lr)
            results[key].extend(
                run_trial_V(V_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, lr)
            )

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = results[key]
        mean = np.mean(rms_errors)
        std = np.std(rms_errors, ddof=1)
        results[key] = (mean, std)
        print('{:.3f}'.format(lr), '{:.3f}'.format(lambd), mean, std)
    return results


def run_sweep_Q(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds):
    assert behavior_policy.sum() == 1.0
    assert target_policy.sum() == 1.0

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    Q_pi = None
    results = defaultdict(list)

    for s in seeds:
        env, experience = sample_episodes(env_id, behavior_policy, n_episodes=10, seed=s)

        if Q_pi is None:
            Q_pi = policy_evaluation_Q(env, discount, target_policy, precision=1e-9)
            Q_pi.flags.writeable = False

        for (estimator, lambd, lr) in all_combos:
            key = (estimator, lambd, lr)
            results[key].extend(
                run_trial_Q(Q_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, lr)
            )

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = results[key]
        mean = np.mean(rms_errors)
        std = np.std(rms_errors, ddof=1)
        results[key] = (mean, std)
        print('{:.3f}'.format(lr), '{:.3f}'.format(lambd), mean, std)
    return results
