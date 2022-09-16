from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import gym
import numpy as np

from moretrace import grid_walk
import moretrace.eligibility_traces.online as eligibility_traces
from moretrace.experiments.sampling import EnvSampler


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


def train(Q, episode, behavior_policy, target_policy, etrace, learning_rate):
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
        etrace.step(s, a, td_errors[t], behavior_policy(s)[a], target_policy[a])
    # print(Q.mean(axis=1))


def rms(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def epsilon_greedy_policy(Q, epsilon):
    assert 0.0 <= epsilon <= 1.0
    n = Q.shape[1]
    def policy(s):
        # One-hot greedy probabilities
        greedy = np.zeros(n)
        greedy[Q[s] == Q[s].max()] = 1
        greedy /= greedy.sum()
        # Uniform-random probabilities
        random = np.ones(n)
        random /= random.sum()
        # Return the epsilon-mixture of the distributions
        return epsilon * random + (1-epsilon) * greedy
    return policy


def run_prediction_trial(env_id, behavior_eps, target_policy, etrace, learning_rate, n_episodes, seed):
    assert behavior_eps == 1.0  # Random behavior policy only
    sampler = EnvSampler(env_id, seed)

    env = sampler.env
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)
    Q_pi = policy_evaluation(env, etrace.discount, target_policy, precision=1e-9)

    rms_errors = [rms(Q, Q_pi)]
    for _ in range(n_episodes):
        behavior_policy = epsilon_greedy_policy(Q, behavior_eps)
        episode = sampler.sample_one_episode(behavior_policy)
        train(Q, episode, behavior_policy, target_policy, etrace, learning_rate)
        rms_errors.append(rms(Q, Q_pi))
    return rms_errors


def run_control_trial(env_id, behavior_eps, target_policy, etrace, learning_rate, n_episodes, seed):
    sampler = EnvSampler(env_id, seed)

    env = sampler.env
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)

    lengths = []
    while len(lengths) <= n_episodes:
        behavior_policy = epsilon_greedy_policy(Q, behavior_eps)
        episode = sampler.sample_one_episode(behavior_policy)
        lengths.append(len(episode))
        train(Q, episode, behavior_policy, target_policy, etrace, learning_rate)
    return lengths


def run_prediction_sweep(*args, **kwargs):
    return _run_sweep(*args, **kwargs, trial_fn=run_prediction_trial)

def run_control_sweep(*args, **kwargs):
    return _run_sweep(*args, **kwargs, trial_fn=run_control_trial)

def _run_sweep(env_id, behavior_eps, target_policy, discount, return_estimators, lambda_values, learning_rates, seeds, n_episodes, trial_fn):
    assert 0.0 <= behavior_eps <= 1.0
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
                future = executor.submit(trial_fn, env_id,
                    behavior_eps, target_policy, etrace, lr, n_episodes, seed=s)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        yields = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(yields)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        yields = np.array(results[key])
        assert yields.shape == (n_seeds, n_episodes + 1)
        results[key] = yields

    return results
