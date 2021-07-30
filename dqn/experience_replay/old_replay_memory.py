import numpy as np

from dqn.experience_replay.replay_memory import ReplayMemory
from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities


class OldReplayMemory(ReplayMemory):
    def __init__(self, dqn, capacity, cache_size, discount, lambd, return_estimator,
                 history_len=4, block_size=100):
        if return_estimator == 'Peng':
            self._compute_returns = self._pengs_q_lambda
            # Dummy estimator since we won't be using the retrace operator
            return_estimator = 'Qlambda'

        super().__init__(dqn, capacity, cache_size, discount, lambd, return_estimator, history_len, block_size)
        assert self._cache_size % self._block_size == 0, "cache size must be divisible by block size"
        self._population = 0

    def save(self, observation, action, reward, done, mu):
        if self._observations is None:
            self._observations = np.empty(shape=[self._capacity, *observation.shape], dtype=observation.dtype)

            # Allocate memory for the cached states/actions/returns
            self._cached_states = np.empty_like(np.concatenate(
                self._history_len * [self._observations[:self._cache_size]], axis=-1))
            self._cached_actions = np.empty_like(self._actions[:self._cache_size])
            self._cached_returns = np.empty_like(self._rewards[:self._cache_size])

        self._push((observation, action, reward, done, mu))

        if done:
            self._image_stacker.reset()

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

            # Compute returns (slices off the extra sample automatically)
            returns = self._compute_returns(indices, pi_epsilon)
            indices = indices[:-1]

            # Store states/actions/returns for minibatch sampling later
            sl = slice(k * self._block_size, (k + 1) * self._block_size)
            self._cached_states[sl] = self._get_states(indices)
            self._cached_actions[sl] = self._actions[self._absolute(indices)]
            self._cached_returns[sl] = returns

    def _find_episode_boundaries(self):
        raise NotImplementedError

    def _compute_returns(self, indices, pi_epsilon):
        # Get Q-values from the DQN
        q_values = self._dqn.predict(self._get_states(indices)).numpy()

        # Compute the multistep returns:
        # All returns start with the reward
        abs_indices = self._absolute(indices)
        returns = self._rewards[abs_indices]

        # Set up the bootstrap for the last state
        last = abs_indices[-1]
        if not self._dones[last]:
            pi = epsilon_greedy_probabilities(q_values[-1], pi_epsilon)
            returns[-1] = (pi * q_values[-1]).sum()
        else:
            returns[-1] = 0.0

        # For all timesteps except the last, compute the returns
        for i in reversed(range(len(q_values) - 1)):
            x = abs_indices[i]  # Absolute location in replay memory

            if self._dones[x]:
                # This is a terminal transition so we're already done
                continue

            # Compute the target policy probabilities (assuming epsilon-greedy policy)
            pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
            mu = self._mu_policies[x]

            # Add the discounted expected value of the next state
            returns[i] += self._discount * (pi * q_values[i+1]).sum()

            # Recursion: Propagate the discounted future multistep TD error backwards,
            # weighted by the current trace
            trace = self._compute_trace(self._actions[x], pi, mu)
            next_action = self._actions[(x + 1) % self._capacity]
            next_td_error = returns[i+1] - q_values[i+1, next_action]
            returns[i] += self._discount * trace * next_td_error

        # Check for abnormally large returns
        assert (np.abs(returns) < 1e6).all()
        return returns[:-1]

    def _pengs_q_lambda(self, indices, pi_epsilon):
        abs_indices = self._absolute(indices)

        # Get Q-values from the DQN
        q_values = self._dqn.predict(self._states[abs_indices]).numpy()

        # Compute the multistep returns:
        # All returns start with the reward
        returns = self._rewards[abs_indices]

        # Set up the bootstrap for the last state
        last = abs_indices[-1]
        if not self._dones[last]:
            returns[-1] = q_values[-1].max()
        else:
            returns[-1] = 0.0

        # For all timesteps except the last, compute the returns
        for i in reversed(range(len(q_values) - 1)):
            x = abs_indices[i]  # Absolute location in replay memory

            if self._dones[x]:
                # This is a terminal transition so we're already done
                continue

            # Compute the target policy probabilities (assuming epsilon-greedy policy)
            pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
            mu = self._mu_policies[x]

            # Add the discounted expected value of the next state
            returns[i] += self._discount * q_values[i+1].max()

            # Recursion: Propagate the discounted future multistep TD error backwards
            next_td_error = returns[i+1] - q_values[i+1].max()
            returns[i] += self._discount * self._lambd * next_td_error

        # Check for abnormally large returns
        assert (np.abs(returns) < 1e6).all()
        return returns[:-1]
