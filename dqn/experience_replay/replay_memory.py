import numpy as np

from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities


class ReplayMemory:
    def __init__(self, dqn, capacity, cache_size, discount, lambd, return_estimator,
                 refresh_split=2):
        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        assert 0.0 <= discount <= 1.0, "discount must be in the range [0,1]"
        assert cache_size % refresh_split == 0, "cache size must be divisible by split"
        self._capacity = capacity
        self._cache_size = cache_size
        self._block_size = cache_size // refresh_split

        self._dqn = dqn
        self._discount = discount
        self._lambd = lambd
        self._compute_trace = get_trace_function(return_estimator, lambd)

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

    def _relative(self, i):
        return (i - self._front) % self._capacity

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

        assert len(indices) > 0
        indices = sorted(indices)
        self._cache_indices = indices = np.array(indices)
        self._obsolete = 0  # Number of indices that have gone negative, meaning they were deleted

        # Shorter names to make the code easier to read below
        states, actions, rewards, dones, mu_policies = (
            self._states, self._actions, self._rewards, self._dones, self._mu_policies)

        # Get Q-values from the DQN
        q_values = np.empty_like(rewards, shape=[len(indices), self._dqn.n])
        for i in range(self._cache_size // self._block_size):
            s = slice(i * self._block_size, (i + 1) * self._block_size)
            x = self._absolute(indices[s])
            q_values[s] = self._dqn.predict(states[x])

        # Compute the multistep returns
        returns = rewards[self._absolute(indices)]  # All returns start with the reward
        assert dones[self._absolute(indices[-1])], "trajectory must end at an episode boundary"
        for i in reversed(range(len(indices))):
            x = self._absolute(indices[i])  # Absolute location in replay memory

            if dones[x]:
                # This is a terminal transition so we're already done
                continue

            # Compute the target policy probabilities (assuming epsilon-greedy policy)
            pi = epsilon_greedy_probabilities(q_values[i], pi_epsilon)
            mu = mu_policies[x]

            # Add the discounted expected value of the next state
            returns[i] += self._discount * (pi * q_values[i+1]).sum()

            # Recursion: Propagate the discounted future multistep TD error backwards,
            # weighted by the current trace
            trace = self._compute_trace(actions[x], pi, mu)
            next_action = actions[(x + 1) % self._capacity]
            next_td_error = returns[i+1] - q_values[i+1, next_action]
            returns[i] += self._discount * trace * next_td_error

        # Store returns for minibatch sampling later
        self._returns = returns

    def _find_episode_boundaries(self):
        # Start by finding episode ends (note these are *relative* to the front)
        if self._back >= self._front:
            dones = self._dones[self._front:self._back]
        else:
            # The buffer has wrapped around; we need to re-order the dones
            dones = np.concatenate([self._dones[self._front:],
                                    self._dones[:self._back]])
        ends, = np.where(dones)

        # Episodes always start 1 timestep after an end (except for the last episode)
        starts = ends[:-1] + 1
        # Prepend 0 to account for the first episode start
        starts = np.insert(starts, obj=0, values=0, axis=0)

        # Compute episode lengths
        lengths = (ends - starts) + 1
        # Sanity check: make sure the episode lengths sum up to the correct value
        assert lengths.sum() == ends[-1] - starts[0] + 1
        # Although episode lengths of 1 are possible for MDPs in general,
        # for Atari games it probably means something went wrong here
        assert (lengths > 1).all()

        return starts, ends, lengths
