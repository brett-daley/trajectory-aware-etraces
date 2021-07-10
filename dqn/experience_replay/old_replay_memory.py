import numpy as np

from dqn.experience_replay.replay_memory import ReplayMemory
from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities


class OldReplayMemory(ReplayMemory):
    def __init__(self, dqn, capacity, cache_size, discount, lambd, return_estimator,
                 block_size=100):
        super().__init__(dqn, capacity, cache_size, discount, lambd, return_estimator)
        assert cache_size % block_size == 0, "cache size must be divisible by block size"
        self._block_size = block_size
        self._population = 0

    def save(self, state, action, reward, done, mu):
        if self._states is None:
            self._states = np.empty(shape=[self._capacity, *state.shape], dtype=state.dtype)

        self._push((state, action, reward, done, mu))

        if self._back == self._front:
            # The memory is full; delete the oldest experience
            self._pop()

    def _push(self, transition):
        b = self._back
        self._states[b], self._actions[b], self._rewards[b], self._dones[b], self._mu_policies[b] = transition
        self._back = (self._back + 1) % self._capacity
        self._population = min(self._population + 1, self._capacity)

    def sample(self, batch_size):
        # j = np.random.randint(self._cache_size, size=batch_size)
        # return (self._cached_states[j], self._cached_actions[j], self._cached_returns[j])
        raise NotImplementedError

    def iterate_cache(self, batch_size):
        # Sample without replacement
        indices = np.arange(self._cache_size)
        np.random.shuffle(indices)

        # Yield unique minibatches until the cache is exhausted
        n_batches = self._cache_size // batch_size
        for i in range(n_batches):
            j = indices[i * batch_size : (i + 1) * batch_size]
            yield (self._cached_states[j], self._cached_actions[j], self._cached_returns[j])

    def refresh_cache(self, pi_epsilon):
        # Shorter names to make the code easier to read below
        states, actions, rewards, dones, mu_policies = (
            self._states, self._actions, self._rewards, self._dones, self._mu_policies)

        # Allocate memory for the cached states/actions/returns
        N = self._cache_size
        self._cached_states = np.empty_like(self._states[:N])
        self._cached_actions = np.empty_like(self._actions[:N])
        self._cached_returns = np.empty_like(self._rewards[:N])

        # Sample blocks and compute returns until we fill up the cache
        for k in range(self._cache_size // self._block_size):
            # Sample a random block
            start = np.random.randint(self._population - self._block_size)
            end = start + self._block_size

            # Add all transitions from the block to the cache
            indices = np.arange(start, end + 1)  # Includes an extra sample for bootstrapping
            abs_indices = self._absolute(indices)

            # Get Q-values from the DQN
            q_values = self._dqn.predict(states[abs_indices]).numpy()

            # Compute the multistep returns:
            # All returns start with the reward
            returns = self._rewards[abs_indices]

            # Set up the bootstrap for the last state
            last = abs_indices[-1]
            if not dones[last]:
                pi = epsilon_greedy_probabilities(q_values[-1], pi_epsilon)
                returns[-1] = (pi * q_values[-1]).sum()
            else:
                returns[-1] = 0.0

            # For all timesteps except the last, compute the returns
            for i in reversed(range(len(q_values) - 1)):
                x = abs_indices[i]  # Absolute location in replay memory

                if dones[x]:
                    # This is a terminal transition so we're already done
                    continue

                # Compute the target policy probabilities (assuming epsilon-greedy policy)
                pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
                mu = mu_policies[x]

                # Recursion: Propagate the discounted future multistep TD error backwards,
                # weighted by the current trace
                trace = self._compute_trace(actions[x], pi, mu)
                next_action = actions[(x + 1) % self._capacity]
                next_td_error = returns[i+1] - q_values[i+1, next_action]
                returns[i] += self._discount * trace * next_td_error

                # Add the discounted expected value of the next state
                returns[i] += self._discount * (pi * q_values[i+1]).sum()

            # Store states/actions/returns for minibatch sampling later
            sl = slice(k * self._block_size, (k + 1) * self._block_size)
            self._cached_states[sl] = states[abs_indices[:-1]]
            self._cached_actions[sl] = actions[abs_indices[:-1]]
            self._cached_returns[sl] = returns[:-1]

    def _find_episode_boundaries(self):
        raise NotImplementedError
