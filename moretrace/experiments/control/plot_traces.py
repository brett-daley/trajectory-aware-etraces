import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import moretrace.eligibility_traces.online as eligibility_traces
import moretrace.envs
from moretrace.experiments.plot_formatting import preformat_plots, postformat_plots
from moretrace.experiments.sampling import EnvSampler
from moretrace.experiments.training import epsilon_greedy_policy


DISCOUNT = 0.9
BIFURCATION_STATE = (2, 0)

N = 100
CMAP = plt.cm.get_cmap('turbo', N)
np.random.seed(0)
COLORS = [CMAP(np.random.randint(N)) for _ in range(N)]


def value_iteration(env, discount, precision=1e-3):
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    Q = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.float64)

    while True:
        Q_old = Q.copy()

        for s in env.states():
            for a in env.actions():
                next_states, rewards, dones, probs = env.model(s, a)
                Q[s,a] = np.sum(probs * (rewards + discount * np.max(Q[next_states], axis=1)))

        if np.abs(Q - Q_old).max() <= precision:
            return Q


def generate_trace(estimator, discount, lambd, seed, target_action):
    env_id = 'BifurcatedGridworld-v0'
    sampler = EnvSampler(env_id, seed)
    env = sampler.env

    Q_star = value_iteration(env, discount)
    behavior_policy = epsilon_greedy_policy(Q_star, epsilon=0.2)
    target_policy = epsilon_greedy_policy(Q_star, epsilon=0.1)

    assert 0.0 <= lambd <= 1.0
    estimator = estimator.replace(' ', '')
    etrace = getattr(eligibility_traces, estimator)(discount, lambd)
    # Set alpha=0 because we don't need to actually learn
    etrace.set(Q_star, alpha=0)

    # Convert the bifurcation
    target_sa_pair = (env._encode(BIFURCATION_STATE), target_action)

    eligibilities = []
    start_collecting_data = False

    done = False
    while not done:
        s, a, reward, next_state, done = sampler.step(behavior_policy)

        # Set td_error=0 because we don't care about actually learning
        td_error = 0.0
        etrace.step(s, a, td_error, behavior_policy(s)[a], target_policy(s)[a])

        states = etrace.states.numpy()
        actions = etrace.actions.numpy()
        betas = etrace.betas.numpy()

        e = 0.0
        for i, _ in enumerate(states):
            sa_pair = (states[i], actions[i])
            if sa_pair == target_sa_pair:
                e += betas[i]

        if e > 0.0:
            start_collecting_data = True
        if start_collecting_data:
            eligibilities.append(e)

    return eligibilities


def plot(estimator, lambd, action):
    action = {
        'up': 0,
        'right': 1,
        'down': 2,
        'left': 3,
    }[action]
    print(action)
    traces = [generate_trace(estimator, DISCOUNT, lambd=lambd, seed=s, target_action=action)
              for s in range(N)]

    max_len = max([len(tr) for tr in traces]) - 1
    X = np.arange(max_len + 1)

    for i, tr in enumerate(traces):
        plt.plot(X[:len(tr)], tr, label=estimator, color=COLORS[i], alpha=0.5)

        plt.xlim([0, max_len])
        plt.ylim([0, 1.4])


def main():
    specs = {
        'Retrace': 0.6,
        'Truncated IS': 0.4,
        'Recursive Retrace': 0.7,
        'RBIS': 0.4,
    }

    plt.figure()
    # preformat_plots()

    for estimator, lambd in specs.items():
        for action in ['up', 'right', 'down', 'left']:
            plot(estimator, lambd, action)

            estimator = estimator.replace(' ', '')
            plot_name = f"traces_{estimator}_{action}"
            plot_path = os.path.join('plots', plot_name)
            print(plot_path)
            plt.savefig(plot_path + '.png')
            # plt.savefig(plot_path + '.pdf', format='pdf')


if __name__ == '__main__':
    main()
