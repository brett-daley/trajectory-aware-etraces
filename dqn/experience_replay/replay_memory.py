from collections import deque

import numpy as np

from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities


class ReplayMemory:
    def __init__(self, dqn, capacity, cache_size, block_size, discount, return_estimator):
        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        self._size_now = 0
        self._capacity = capacity

        self._cache = ReplayCache(dqn, cache_size, block_size, discount, return_estimator)
        self._completed_episodes = deque()
        self._current_episode = Episode()

    def save(self, state, action, reward, done, epsilon):
        self._current_episode.append_transition(state, action, reward, done, epsilon)
        if done:
            self._completed_episodes.append(self._current_episode)
            self._size_now += len(self._current_episode)
            self._current_episode = Episode()

        # Memory management
        while self._size_now > self._capacity:
            episode = self._completed_episodes.popleft()
            self._size_now -= len(episode)

    def sample(self, batch_size):
        return self._cache.sample(batch_size)

    def refresh_cache(self, pi_epsilon):
        self._cache.refresh(self._completed_episodes, pi_epsilon)


class ReplayCache:
    def __init__(self, dqn, capacity, block_size, discount, return_estimator):
        assert block_size <= capacity, "block size cannot be larger than cache size"
        assert (capacity % block_size) == 0, "block size must evenly divide cache size"
        assert 0.0 <= discount <= 1.0, "discount must be in the interval [0,1]"
        self._dqn = dqn
        self._capacity = capacity
        self._block_size = block_size

        self._discount = discount
        self._compute_trace = get_trace_function(return_estimator)

        # Number of samples currently in the cache (may be less than capacity)
        self._size_now = 0

    def refresh(self, episode_list, pi_epsilon):
        N = 0
        while True:
            # Sample a random episode
            episode = np.random.choice(episode_list)

            # If adding this episode will make the cache too large, exit the loop
            if N + len(episode) > self._capacity:
                break

            states, actions, rewards, dones, mu_epsilons = map(np.array,
                [episode.states, episode.actions, episode.rewards, episode.dones, episode.epsilons])

            if not hasattr(self, '_states'):
                # Allocate arrays only once to save time on the next iterations

                def allocate_like(x, shape=None):
                    if shape is None:
                        shape = (self._capacity, *x.shape[1:])
                    return np.empty_like(x, shape=shape)

                self._states = allocate_like(states)
                self._actions = allocate_like(actions)
                self._rewards = allocate_like(rewards)
                self._dones = allocate_like(dones)
                self._mu_epsilons = allocate_like(mu_epsilons)
                self._q_values = allocate_like(self._rewards, shape=[self._capacity, self._dqn.n])
                self._returns = self._rewards.copy()

            # Add all transitions from the episode to the cache
            s = slice(N, N + len(episode))
            self._states[s] = states
            self._actions[s] = actions
            self._rewards[s] = rewards
            self._dones[s] = dones
            self._mu_epsilons[s] = mu_epsilons
            N += len(episode)

        # Shorter names to make the code easier to read below
        states, actions, rewards, dones, mu_epsilons, q_values, returns = (
            self._states, self._actions, self._rewards, self._dones,
            self._mu_epsilons, self._q_values, self._returns)

        # Get Q-values from the DQN
        for i in range(self._capacity // self._block_size):
            s = slice(i * self._block_size, (i + 1) * self._block_size)
            q_values[s] = self._dqn.predict(states[s])

        # Compute the multistep returns
        assert dones[N-1], "trajectory must end at an episode boundary"
        np.copyto(dst=returns, src=rewards)  # All returns start with the reward
        for j in range(N):
            i = (N - 1) - j
            if dones[i]:
                # This is a terminal transition so we're already done
                continue

            # Compute the action probabilities (assuming epsilon-greedy policies)
            pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
            mu = epsilon_greedy_probabilities(q_values[i], mu_epsilons[i])

            # Add the discounted expected value of the next state
            returns[i] += self._discount * (pi * q_values[i+1]).sum()

            # Recursion: Propagate the discounted future multistep TD error backwards,
            # weighted by the current trace
            trace = self._compute_trace(actions[i], pi, mu)
            next_td_error = returns[i+1] - q_values[i+1, actions[i+1]]
            returns[i] += self._discount * trace * next_td_error

        # Update the cache size for sampling
        self._size_now = N

    def sample(self, batch_size):
        assert self._states is not None, "replay cache must be refreshed before sampling"
        j = np.random.randint(self._size_now, size=batch_size)
        return (self._states[j], self._actions[j], self._returns[j])


class Episode:
    def __init__(self):
        self.states, self.actions, self.rewards, self.dones, self.epsilons = [], [], [], [], []
        self._already_done = False

    def __len__(self):
        return len(self.states)

    def append_transition(self, state, action, reward, done, epsilon):
        assert not self._already_done
        self._already_done = done or self._already_done

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.epsilons.append(epsilon)
