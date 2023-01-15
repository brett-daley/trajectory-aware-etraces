from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import count, product

import gym
import numpy as np

import moretrace.eligibility_traces.online as eligibility_traces
from moretrace.experiments.sampling import EnvSampler


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


def smooth(episode_stats, window=100):
    new_stats = []
    buffer = deque(maxlen=window)
    for x, y in episode_stats:
        buffer.append(y)
        new_stats.append((x, np.mean(buffer)))
    return new_stats


def linear_interpolation(episode_stats, n_timesteps):
    values = []
    for (x1, y1), (x2, y2) in zip(episode_stats[:-1], episode_stats[1:]):
        dx = x2 - x1
        values.append(np.linspace(y1, y2, num=dx))
    values = np.concatenate(values)[:n_timesteps + 1]
    assert len(values) == n_timesteps + 1

    # Sanity check
    for x, y in episode_stats:
        if x >= len(values):
            break
        assert values[x] == y

    return values


# Time limit to ensure training/benchmarking doesn't get stuck in infinite loop
SAFE_TERMINATE_AFTER = 50


def run_control_trial(env_id, behavior_eps, target_eps, etrace, learning_rate, n_timesteps, seed):
    sampler = EnvSampler(env_id, seed)
    test_sampler = EnvSampler(env_id, seed + 1)
    env = sampler.env
    # NOTE: It's really important to randomly initialize the Q-function
    Q = 0.01 * sampler.np_random.randn(env.observation_space.n, env.action_space.n)
    # TODO: Ideally, we'd just pass these args into the constructor
    etrace.set(Q, learning_rate)

    discount = etrace.discount
    assert 0.0 <= discount <= 1.0
    assert 0.0 <= learning_rate <= 1.0

    # To make sure the agent sees the goal, we set the behavior epsilon to 1 for the first episodes
    def get_behavior_policy(n_episodes):
        eps = 1.0 if (n_episodes < 5) else behavior_eps
        return epsilon_greedy_policy(Q, eps)

    def benchmark_policy():
        policy = epsilon_greedy_policy(Q, epsilon=0.05)
        disc_return = 0.0

        for t in count():
            s, a, reward, next_state, done = test_sampler.step(policy)
            disc_return += pow(discount, t) * reward

            if done or (t >= SAFE_TERMINATE_AFTER):
                return disc_return

    def train():
        etrace.reset_traces()
        done = False
        behavior_policy = get_behavior_policy(n_episodes=0)
        target_policy = epsilon_greedy_policy(Q, target_eps)

        episode_stats = [(0, 0.0)]
        disc_return = 0.0
        t_start = 0
        for t in count():
            s, a, reward, next_state, done = sampler.step(behavior_policy)
            disc_return += pow(discount, t - t_start) * reward

            td_error = reward - Q[s, a]
            if not done:
                next_V = (target_policy(next_state)[None] * Q[next_state]).sum(axis=1)
                td_error += discount * next_V
            etrace.step(s, a, td_error, behavior_policy(s)[a], target_policy(s)[a])

            if done:
                episode_stats.append((t, benchmark_policy()))

                t_start = t + 1
                disc_return = 0.0

                # Update the policies only at the end of each episode
                behavior_policy = get_behavior_policy(n_episodes=len(episode_stats) - 1)
                target_policy = epsilon_greedy_policy(Q, target_eps)

                if t > n_timesteps:
                    episode_stats = smooth(episode_stats)
                    return linear_interpolation(episode_stats, n_timesteps)

            # If the episode still hasn't terminated by now, just end it
            if t > n_timesteps + SAFE_TERMINATE_AFTER:
                episode_stats.append((t, benchmark_policy()))
                episode_stats = smooth(episode_stats)
                return linear_interpolation(episode_stats, n_timesteps)

    return train()


def run_control_sweep(env_id, behavior, target, discount, return_estimators, lambda_values, learning_rates, seeds, n_timesteps):
    n_seeds = len(list(seeds))

    all_combos = tuple(product(return_estimators, lambda_values, learning_rates))
    results = defaultdict(list)

    key_to_future_dict = {}
    with ProcessPoolExecutor() as executor:
        for s in seeds:
            for (estimator, lambd, lr) in all_combos:
                key = (estimator, lambd, lr, s)
                etrace = getattr(eligibility_traces, estimator.replace(' ', ''))(discount, lambd)
                future = executor.submit(run_control_trial, env_id,
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
