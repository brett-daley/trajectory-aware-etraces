import numpy as np

from dqn.experience_replay.replay_memory import ReplayMemory
from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities


class OldReplayMemory(ReplayMemory):
    def __init__(self, dqn, capacity, cache_size, discount, lambd, return_estimator,
                 block_size=100):
        super().__init__(dqn, capacity, cache_size, discount, lambd, return_estimator, block_size)
        assert self._cache_size % self._block_size == 0, "cache size must be divisible by block size"
        self._population = 0

    def save(self, state, action, reward, done, mu):
        observation = state[..., -1, None]

        if self._observations is None:
            self._history_len = state.shape[-1]
            self._observations = np.empty(shape=[self._capacity, *observation.shape], dtype=observation.dtype)

            # Allocate memory for the cached states/actions/returns
            self._cached_states = np.empty(shape=[self._cache_size, *state.shape], dtype=state.dtype)
            self._cached_actions = np.empty_like(self._actions[:self._cache_size])
            self._cached_returns = np.empty_like(self._rewards[:self._cache_size])

        self._push((observation, action, reward, done, mu))

        if self._back == self._front:
            # The memory is full; delete the oldest experience
            self._pop()

    def _push(self, transition):
        b = self._back
        self._observations[b], self._actions[b], self._rewards[b], self._dones[b], self._mu_policies[b] = transition
        self._back = (self._back + 1) % self._capacity
        self._population = min(self._population + 1, self._capacity)

    def iterate_cache(self, n_batches, batch_size):
        for j in self._iterate_cache(n_batches, batch_size, actual_cache_size=self._cache_size):
            yield (self._cached_states[j], self._cached_actions[j], self._cached_returns[j])

    def refresh_cache(self, pi_epsilon):
        # Sample blocks and compute returns until we fill up the cache
        for k in range(self._cache_size // self._block_size):
            # Sample a random block
            start = np.random.randint(self._population - self._block_size)
            end = start + self._block_size

            # Add all transitions from the block to the cache
            indices = np.arange(start, end + 1)  # Includes an extra sample for bootstrapping

            # Get Q-values from the DQN
            q_values = self._dqn.predict(self._get_states(indices)).numpy()

            # Compute returns
            returns = self._compute_returns(indices, q_values, pi_epsilon, bootstrap_using_last=True)
            # Slice off the extra sample that was used for bootstrapping
            indices, returns = indices[:-1], returns[:-1]

            # Store states/actions/returns for minibatch sampling later
            sl = slice(k * self._block_size, (k + 1) * self._block_size)
            self._cached_states[sl] = self._get_states(indices)
            self._cached_actions[sl] = self._actions[self._absolute(indices)]
            self._cached_returns[sl] = returns

    def _sample_episodes_for_cache(self):
        raise NotImplementedError

    def _find_episode_boundaries(self):
        raise NotImplementedError
