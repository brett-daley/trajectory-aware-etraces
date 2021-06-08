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

        self._states = None
        self._actions = None
        self._returns = None

    def refresh(self, episode_list, pi_epsilon):
        states, actions, rewards, dones, mu_epsilons = [], [], [], [], []
        while True:
            # Sample a random episode
            i = np.random.randint(len(episode_list))
            episode = episode_list[i]

            # If adding this episode will make the cache too large, exit the loop
            if len(states) + len(episode) > self._capacity:
                break

            # Add all transitions from the episode to the cache
            states.extend(episode.states)
            actions.extend(episode.actions)
            rewards.extend(episode.rewards)
            dones.extend(episode.dones)
            mu_epsilons.extend(episode.epsilons)

        # Convert to numpy arrays for efficient return calculation and sampling
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        done_mask = 1.0 - np.array(dones, dtype=np.float32)

        # Get Q-values from the DQN
        q_values = np.zeros_like(rewards, shape=[len(rewards), self._dqn.n])
        for i in range(self._capacity // self._block_size):
            s = slice(i * self._block_size, (i + 1) * self._block_size)
            q_values[s] = self._dqn.predict(states[s])

        q_values_taken = q_values[np.arange(len(q_values)), actions]

        # Start with the 1-step returns
        assert dones[-1]
        returns = rewards.copy()
        # TODO: We should compute the expected value here like Tree Backup does
        returns[:-1] += self._discount * (done_mask * q_values_taken)[1:]

        # Now propagate the traces backwards to generate the multistep returns
        td_errors = returns - q_values_taken
        for i in reversed(range(len(returns) - 1)):
            if not dones[i]:
                pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
                mu = epsilon_greedy_probabilities(q_values[i], mu_epsilons[i])
                trace = self._compute_trace(actions[i], pi, mu)
                returns[i] += self._discount * trace * td_errors[i+1]

        # Save the values needed for sampling minibatches
        self._states, self._actions, self._returns = states, actions, returns

    def sample(self, batch_size):
        assert self._states is not None, "replay cache must be refreshed before sampling"
        j = np.random.randint(len(self._states), size=batch_size)
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
