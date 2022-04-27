from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import sys
sys.path.append('..')

import gym
import numpy as np

from dqn.experience_replay import eligibility_traces


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
    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    td_errors = rewards - V[states]
    td_errors += discount * (1.0 - dones) * V[next_states]
    updates = etrace(td_errors, target_policy[actions], behavior_policy[actions], dones)

    # Now apply the updates to each visited state-action pair
    returns = V[states] + updates
    for t, (s, _, _, _, _) in enumerate(episode):
        V[s] += learning_rate * (returns[t] - V[s])


def train_Q(Q, episode, behavior_policy, target_policy, discount, etrace, learning_rate):
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0

    # Accumulate updates using the eligibility trace
    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    td_errors = rewards - Q[states, actions]
    next_V = (target_policy[None] * Q[next_states]).sum(axis=1)
    td_errors += discount * (1.0 - dones) * next_V
    updates = etrace(td_errors, target_policy[actions], behavior_policy[actions], dones)

    # Now apply the updates to each visited state-action pair
    returns = Q[states, actions] + updates
    for t, (s, a, _, _, _) in enumerate(episode):
        Q[s, a] += learning_rate * (returns[t] - Q[s, a])


def rms(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def run_trial_V(V_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, learning_rate):
    etrace_cls = getattr(eligibility_traces, estimator)
    etrace = etrace_cls(discount, lambd)

    V = np.zeros_like(V_pi)
    rms_errors = []
    for episode in experience:
        train_V(V, episode, behavior_policy, target_policy, discount, etrace, learning_rate)
        rms_errors.append(rms(V, V_pi))
    return rms_errors


def run_trial_Q(Q_pi, experience, behavior_policy, target_policy, discount, estimator, lambd, learning_rate):
    etrace_cls = getattr(eligibility_traces, estimator)
    etrace = etrace_cls(discount, lambd)

    Q = np.zeros_like(Q_pi)
    rms_errors = []
    for episode in experience:
        train_Q(Q, episode, behavior_policy, target_policy, discount, etrace, learning_rate)
        rms_errors.append(rms(Q, Q_pi))
    return rms_errors


def run_sweep_V(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds):
    assert behavior_policy.sum() == 1.0
    assert target_policy.sum() == 1.0

    n_episodes = 100
    n_seeds = len(list(seeds))

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    V_pi = None
    results = defaultdict(list)

    key_to_future_dict = {}
    with ProcessPoolExecutor() as executor:
        for s in seeds:
            env, experience = sample_episodes(env_id, behavior_policy, n_episodes, seed=s)

            if V_pi is None:
                V_pi = policy_evaluation_V(env, discount, target_policy, precision=1e-9)
                V_pi.flags.writeable = False

            for (estimator, lambd, lr) in all_combos:
                key = (estimator, lambd, lr, s)
                future = executor.submit(run_trial_V, V_pi, experience,
                    behavior_policy, target_policy, discount, estimator, lambd, lr)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        rms_errors = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(rms_errors)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = np.array(results[key])
        assert rms_errors.shape == (n_seeds, n_episodes)

        mean = np.mean(rms_errors)
        error = np.std(rms_errors.mean(axis=1), ddof=1)
        results[key] = (mean, error)
        print('{:.3f}'.format(lr), '{:.3f}'.format(lambd), mean, error)

    return results


def run_sweep_Q(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds):
    assert behavior_policy.sum() == 1.0
    assert target_policy.sum() == 1.0

    n_episodes = 100
    n_seeds = len(list(seeds))

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    Q_pi = None
    results = defaultdict(list)

    key_to_future_dict = {}
    with ProcessPoolExecutor() as executor:
        for s in seeds:
            env, experience = sample_episodes(env_id, behavior_policy, n_episodes, seed=s)

            if Q_pi is None:
                Q_pi = policy_evaluation_Q(env, discount, target_policy, precision=1e-9)
                Q_pi.flags.writeable = False

            for (estimator, lambd, lr) in all_combos:
                key = (estimator, lambd, lr, s)
                future = executor.submit(run_trial_Q, Q_pi, experience,
                    behavior_policy, target_policy, discount, estimator, lambd, lr)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        rms_errors = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(rms_errors)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = np.array(results[key])
        assert rms_errors.shape == (n_seeds, n_episodes)

        mean = np.mean(rms_errors)
        error = np.std(rms_errors.mean(axis=1), ddof=1)
        results[key] = (mean, error)
        print('{:.3f}'.format(lr), '{:.3f}'.format(lambd), mean, error)

    return results