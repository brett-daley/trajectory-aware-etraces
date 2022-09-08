from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import sys
sys.path.append('..')

import gym
import numpy as np

from eligibility_traces import online_eligibility_traces as eligibility_traces


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
            state = next_state
        episodes.append(tuple(transitions))
    return env, tuple(episodes)


def train_V(V, episode, behavior_policy, target_policy, etrace, learning_rate):
    discount = etrace.discount
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0

    # Accumulate updates using the eligibility trace
    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    td_errors = rewards - V[states]
    td_errors += discount * (1.0 - dones) * V[next_states]
    updates = etrace(td_errors, behavior_policy[actions], target_policy[actions], dones)

    # Now apply the updates to each visited state
    returns = V[states] + updates
    for t, (s, _, _, _, _) in enumerate(episode):
        V[s] += learning_rate * (returns[t] - V[s])


def train_Q(Q, episode, behavior_policy, target_policy, etrace, learning_rate):
    discount = etrace.discount
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0
    # TODO: Ideally, we'd just pass these args into the constructor
    etrace.set(Q, learning_rate)

    # Calculate the 1-step TD errors
    states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
    dones = dones.astype(np.float32)
    td_errors = rewards - Q[states, actions]
    next_V = (target_policy[None] * Q[next_states]).sum(axis=1)
    td_errors += discount * (1.0 - dones) * next_V

    # Now apply the updates to each visited state-action pair
    etrace.reset_traces()
    for t, (s, a, _, _, _) in enumerate(episode):
        etrace.step(s, a, td_errors[t], behavior_policy[a], target_policy[a])
    # print(Q.mean(axis=1))


def rms(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def run_trial_V(env_id, behavior_policy, target_policy, etrace, learning_rate, n_episodes, seed):
    env, experience = sample_episodes(env_id, behavior_policy, n_episodes, seed)

    V = np.zeros([env.observation_space.n], dtype=np.float64)
    V_pi = policy_evaluation_V(env, etrace.discount, target_policy, precision=1e-9)

    rms_errors = []
    for episode in experience:
        train_V(V, episode, behavior_policy, target_policy, etrace, learning_rate)
        rms_errors.append(rms(V, V_pi))
    return rms_errors


def run_trial_Q(env_id, behavior_policy, target_policy, etrace, learning_rate, n_episodes, seed):
    env, experience = sample_episodes(env_id, behavior_policy, n_episodes, seed)

    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)
    Q_pi = policy_evaluation_Q(env, etrace.discount, target_policy, precision=1e-9)

    rms_errors = [rms(Q, Q_pi)]
    for episode in experience:
        train_Q(Q, episode, behavior_policy, target_policy, etrace, learning_rate)
        rms_errors.append(rms(Q, Q_pi))
    return rms_errors


def run_sweep_V(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds):
    assert behavior_policy.sum() == 1.0
    assert target_policy.sum() == 1.0

    n_episodes = 100
    n_seeds = len(list(seeds))

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    results = defaultdict(list)

    key_to_future_dict = {}
    with ProcessPoolExecutor() as executor:
        for s in seeds:
            for (estimator, lambd, lr) in all_combos:
                key = (estimator, lambd, lr, s)
                etrace = getattr(eligibility_traces, estimator)(discount, lambd)
                future = executor.submit(run_trial_V, env_id,
                    behavior_policy, target_policy, etrace, lr, n_episodes, seed=s)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        rms_errors = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(rms_errors)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = np.array(results[key])
        assert rms_errors.shape == (n_seeds, n_episodes + 1)
        results[key] = rms_errors

    return results


def run_sweep_Q(env_id, behavior_policy, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds, n_episodes):
    assert np.isclose(behavior_policy.sum(), 1.0)
    assert np.isclose(target_policy.sum(), 1.0)

    n_seeds = len(list(seeds))

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    results = defaultdict(list)

    key_to_future_dict = {}
    with ProcessPoolExecutor() as executor:
        for s in seeds:
            for (estimator, lambd, lr) in all_combos:
                key = (estimator, lambd, lr, s)
                etrace = getattr(eligibility_traces, estimator.replace(' ', ''))(discount, lambd)
                future = executor.submit(run_trial_Q, env_id,
                    behavior_policy, target_policy, etrace, lr, n_episodes, seed=s)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        rms_errors = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(rms_errors)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        rms_errors = np.array(results[key])
        assert rms_errors.shape == (n_seeds, n_episodes + 1)
        results[key] = rms_errors

    return results
