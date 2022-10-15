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


def run_prediction_trial(env_id, behavior_probs, target_probs, etrace, learning_rate, n_episodes, seed):
    sampler = EnvSampler(env_id, seed)
    env = sampler.env
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)
    Q_pi = policy_evaluation(env, etrace.discount, target_probs, precision=1e-9)
    # TODO: Ideally, we'd just pass these args into the constructor
    etrace.set(Q, learning_rate)

    def train(episode):
        discount = etrace.discount
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= learning_rate <= 1.0

        # Calculate the 1-step TD errors
        states, actions, rewards, next_states, dones = map(np.array, zip(*episode))
        dones = dones.astype(np.float32)
        td_errors = rewards - Q[states, actions]
        next_V = (target_probs[None] * Q[next_states]).sum(axis=1)
        td_errors += discount * (1.0 - dones) * next_V

        # Now apply the updates to each visited state-action pair
        etrace.reset_traces()
        for t, (s, a, _, _, _) in enumerate(episode):
            etrace.step(s, a, td_errors[t], behavior_probs[a], target_probs[a])

    rms_errors = [rms(Q, Q_pi)]
    for _ in range(n_episodes):
        episode = sampler.sample_one_episode(lambda s: behavior_probs)
        train(episode)
        rms_errors.append(rms(Q, Q_pi))
    return rms_errors


def run_control_trial(env_id, behavior_eps, target_eps, etrace, learning_rate, n_timesteps, seed):
    sampler = EnvSampler(env_id, seed)
    env = sampler.env
    # NOTE: It's really important to randomly initialize the Q-function
    Q = 0.1 * sampler.np_random.randn(env.observation_space.n, env.action_space.n)
    # TODO: Ideally, we'd just pass these args into the constructor
    etrace.set(Q, learning_rate)

    def train():
        discount = etrace.discount
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= learning_rate <= 1.0

        etrace.reset_traces()
        done = False
        behavior_policy = epsilon_greedy_policy(Q, behavior_eps)
        target_policy = epsilon_greedy_policy(Q, target_eps)

        episodes = 0
        all_values = []
        # Apply the updates to each visited state-action pair online
        for t in range(n_timesteps + 1):
            all_values.append(episodes)
            s, a, reward, next_state, done = sampler.step(behavior_policy)

            td_error = reward - Q[s, a]
            if not done:
                next_V = (target_policy(next_state)[None] * Q[next_state]).sum(axis=1)
                td_error += discount * next_V
            etrace.step(s, a, td_error, behavior_policy(s)[a], target_policy(s)[a])

            if done:
                episodes += 1
                # Update the policies only at the end of each episode
                behavior_policy = epsilon_greedy_policy(Q, behavior_eps)
                target_policy = epsilon_greedy_policy(Q, target_eps)

        assert len(all_values) == n_timesteps + 1
        return all_values

    return train()


def run_prediction_sweep(*args, **kwargs):
    return _run_sweep(*args, **kwargs, trial_fn=run_prediction_trial)

def run_control_sweep(*args, **kwargs):
    return _run_sweep(*args, **kwargs, trial_fn=run_control_trial)

def _run_sweep(env_id, behavior, target, discount, return_estimators, lambda_values, learning_rates, seeds, n_timesteps, trial_fn):
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
                    behavior, target, etrace, lr, n_timesteps, seed=s)
                key_to_future_dict[key] = future

    for key in key_to_future_dict.keys():
        yields = key_to_future_dict[key].result()
        (estimator, lambd, lr, _) = key
        results[(estimator, lambd, lr)].append(yields)

    for (estimator, lambd, lr) in all_combos:
        key = (estimator, lambd, lr)
        yields = np.array(results[key])
        # assert yields.shape == (n_seeds, n_episodes + 1)
        results[key] = yields

    return results