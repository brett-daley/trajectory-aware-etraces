import numpy as np

from dqn.experience_replay import eligibility_traces


class ReplayMemory:
    def __init__(self, dqn, capacity, cache_size, block_size, discount, lambd, return_estimator):
        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        assert block_size <= cache_size, "block size cannot be larger than cache size"
        assert 0.0 <= discount <= 1.0, "discount must be in the range [0,1]"
        self._capacity = capacity
        self._cache_size = cache_size
        self._block_size = block_size

        self._dqn = dqn
        self._discount = discount
        self._etrace = getattr(eligibility_traces, return_estimator)(discount, lambd)

        self._states = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float64)
        self._dones = np.empty(capacity, dtype=np.bool)
        self._mu_policies = np.empty([capacity, self._dqn.n], dtype=np.float64)

        self._front = 0  # Points to the oldest experience
        self._back = 0   # Points to the next experience to be overwritten

    def save(self, state, action, reward, done, mu):
        if self._states is None:
            self._states = np.empty(shape=[self._capacity, *state.shape], dtype=state.dtype)

        self._push((state, action, reward, done, mu))

        if self._back == self._front:
            # The memory is full; delete the oldest episode
            while not self._dones[self._front]:
                self._pop()
            assert self._dones[self._front]
            self._pop()
            assert not self._dones[self._front]

    def _push(self, transition):
        b = self._back
        self._states[b], self._actions[b], self._rewards[b], self._dones[b], self._mu_policies[b] = transition
        self._back = (self._back + 1) % self._capacity

    def _pop(self):
        self._front = (self._front + 1) % self._capacity
        if hasattr(self, '_cache_indices'):
            self._cache_indices -= 1
            while self._cache_indices[self._obsolete] < 0:
                self._obsolete += 1

    def sample(self, batch_size):
        assert self._states is not None, "replay cache must be refreshed before sampling"
        j = np.random.randint(low=self._obsolete,
            high=len(self._cache_indices), size=batch_size)
        assert (self._cache_indices[j] >= 0).all()
        x = self._absolute(self._cache_indices[j])
        return (self._states[x], self._actions[x], self._returns[j])

    def _absolute(self, i):
        return (self._front + i) % self._capacity

    def refresh_cache(self, pi_epsilon):
        starts, ends, lengths = self._find_episode_boundaries()

        # Sample episodes randomly until we have enough samples for the cache
        # Save the relative indices of each experience for each episode
        indices = []
        while True:
            # Sample a random episode (let k be its ID)
            k = np.random.randint(len(starts))
            start, end, length = starts[k], ends[k], lengths[k]

            # If adding this episode will make the cache too large, exit the loop
            if len(indices) + length > self._cache_size:
                break

            # Add all transitions from the episode to the cache
            assert self._dones[self._absolute(end)]
            indices.extend(list(range(start, end + 1)))

        indices = sorted(indices)
        self._cache_indices = indices = np.array(indices)
        self._obsolete = 0  # Number of indices that have gone negative, meaning they were deleted

        # Get Q-values from the DQN
        T = len(indices)
        q_values = np.empty(shape=[T, self._dqn.n], dtype=self._rewards.dtype)
        for i in range(self._cache_size // self._block_size):
            s = slice(i * self._block_size, (i + 1) * self._block_size)
            x = self._absolute(indices[s])
            q_values[s] = self._dqn.predict(self._states[x])

        # Compute the multistep returns
        timesteps = np.arange(T)
        abs_indices = self._absolute(indices)

        actions = self._actions[abs_indices]
        dones = self._dones[abs_indices].astype(np.float32)

        behavior_probs = self._mu_policies[abs_indices, actions]
        pi = epsilon_greedy_probabilities(q_values, pi_epsilon)
        target_probs = pi[timesteps, actions]

        taken_q_values = q_values[timesteps, actions]
        td_errors = self._rewards[abs_indices] - taken_q_values
        # Bootstrap from the next state values if non-terminal
        assert dones[-1], "trajectory must end at an episode boundary"
        td_errors[:-1] += (self._discount * (1.0 - dones) * (pi * q_values).sum(axis=1))[1:]

        # Store returns for minibatch sampling later
        updates = self._etrace(td_errors, behavior_probs, target_probs, dones)
        self._returns = taken_q_values + updates

    def _find_episode_boundaries(self):
        # 0th episode "ends" at -1 (relative), since buffer always begins at an episode start
        ends = [-1]
        # Let i be the relative index, and x be the absolute index
        i = 0
        # Scan the buffer to obtain the episode ends
        while True:
            x = self._absolute(i)
            if x == self._back:
                break
            if self._dones[x]:
                ends.append(i)
            i += 1
        ends = np.array(ends)
        # Starts are always 1 timestep after an end
        starts = ends[:-1] + 1
        # Get rid of the -1 "end"
        ends = ends[1:]
        # Compute episode lengths
        lengths = (ends - starts) + 1
        assert lengths.sum() == ends[-1] - starts[0] + 1
        return starts, ends, lengths


def epsilon_greedy_probabilities(q_values, epsilon):
    assert q_values.ndim in {1, 2}, "Q-values must be a 1- or 2-dimensional vector"
    assert 0.0 <= epsilon <= 1.0, "epsilon must be in the interval [0,1]"

    is_1d = (q_values.ndim == 1)
    if is_1d:
        q_values = q_values[None]

    n_actions = q_values.shape[-1]
    probabilities = (epsilon / n_actions) * np.ones_like(q_values)
    probabilities[np.arange(len(q_values)), np.argmax(q_values, axis=-1)] += (1.0 - epsilon)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)

    if is_1d:
        probabilities = probabilities[0]
    return probabilities
