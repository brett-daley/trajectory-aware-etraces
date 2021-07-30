import math

import numpy as np

from dqn.experience_replay.traces import get_trace_function, epsilon_greedy_probabilities
from dqn.image_stacker import ImageStacker


class ReplayMemory:
    def __init__(self, dqn, capacity, cache_size, discount, lambd, return_estimator,
                 history_len=4, block_size=16384):
        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        assert 0.0 <= discount <= 1.0, "discount must be in the range [0,1]"
        assert history_len >= 1, "history length must be a positive integer"
        assert block_size >= 1, "block size must be a positive integer"
        self._capacity = capacity
        self._cache_size = cache_size
        self._block_size = block_size

        self._history_len = history_len
        self._image_stacker = ImageStacker(history_len)

        self._dqn = dqn
        self._discount = discount
        self._lambd = lambd
        self._compute_trace = get_trace_function(return_estimator, lambd)

        self._observations = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float64)
        self._dones = np.empty(capacity, dtype=np.bool)
        self._mu_policies = np.empty([capacity, self._dqn.n], dtype=np.float64)

        self._front = 0  # Points to the oldest experience
        self._back = 0   # Points to the next experience to be overwritten

    def get_state(self, observation):
        self._image_stacker.append(observation)
        return self._image_stacker.get_stack()

    def save(self, observation, action, reward, done, mu):
        if self._observations is None:
            self._observations = np.empty(shape=[self._capacity, *observation.shape], dtype=observation.dtype)

        self._push((observation, action, reward, done, mu))

        if done:
            self._image_stacker.reset()

        if self._back == self._front:
            # The memory is full; delete the oldest episode
            while not self._dones[self._front]:
                self._pop()
            assert self._dones[self._front]
            self._pop()
            assert not self._dones[self._front]

    def _push(self, transition):
        b = self._back
        self._observations[b], self._actions[b], self._rewards[b], self._dones[b], self._mu_policies[b] = transition
        self._back = (self._back + 1) % self._capacity

    def _pop(self):
        self._front = (self._front + 1) % self._capacity

    def iterate_cache(self, n_batches, batch_size):
        for j in self._iterate_cache(n_batches, batch_size, actual_cache_size=len(self._cache_indices)):
            i = self._cache_indices[j]
            x = self._absolute(i)
            yield (self._get_states(i), self._actions[x], self._returns[j])

    def _iterate_cache(self, n_batches, batch_size, actual_cache_size):
        # Actual cache size cannot be larger than the nominal size
        assert actual_cache_size <= self._cache_size

        # We must be able to sample at least one minibatch
        assert batch_size <= actual_cache_size

        # Yield minibatches of indices without replacement
        indices = np.arange(actual_cache_size)
        np.random.shuffle(indices)
        start = 0
        for _ in range(n_batches):
            end = start + batch_size

            if end > actual_cache_size:
                # There aren't enough samples for the requested number of minibatches;
                # re-shuffle and start another pass
                np.random.shuffle(indices)
                start, end = 0, batch_size

            assert len(indices[start:end]) == batch_size
            yield indices[start:end]
            start += batch_size

    def _get_states(self, indices):
        states = []
        for j in reversed(range(self._history_len)):
            x = self._absolute(indices - j)
            states.append(self._observations[x])

        mask = np.ones_like(states[0])
        for j in range(1, self._history_len):
            i = indices - j
            x = self._absolute(i)
            mask[self._dones[x]] = 0.0
            mask[np.where(i < 0)] = 0.0
            states[-1 - j] *= mask

        states = np.concatenate(states, axis=-1)
        assert states.shape[0] == len(indices)
        assert (states.shape[-1] % self._history_len) == 0
        return states

    def _absolute(self, i):
        return (self._front + i) % self._capacity

    def _relative(self, i):
        return (i - self._front) % self._capacity

    def refresh_cache(self, pi_epsilon):
        starts, ends, lengths = self._find_episode_boundaries()

        # Sample episodes randomly until we have enough samples for the cache
        # Save the relative indices of each experience for each episode
        indices = []
        almost_full = False
        while not almost_full:
            # Attempt to sample episodes without replacement, but we will repeat when
            # the replay memory is smaller than the cache (early in training only)
            shuffle = np.arange(len(starts))
            np.random.shuffle(shuffle)

            for k in shuffle:
                start, end, length = starts[k], ends[k], lengths[k]

                if len(indices) + length > self._cache_size:
                    # Adding this episode will make the cache too large; skip it but
                    # finish checking the remaining episodes. We won't conduct any more
                    # sweeps after this one (otherwise we bias for short episodes).
                    almost_full = True
                    continue

                # Add all transitions from this episode to the cache
                assert self._dones[self._absolute(end)]
                indices.extend(list(range(start, end + 1)))

        assert len(indices) > 0
        indices = sorted(indices)
        self._cache_indices = indices = np.array(indices)

        # Shorter names to make the code easier to read below
        observations, actions, rewards, dones, mu_policies = (
            self._observations, self._actions, self._rewards, self._dones, self._mu_policies)

        # Get Q-values from the DQN
        q_values = np.empty_like(rewards, shape=[len(indices), self._dqn.n])
        n_batches = math.ceil(len(indices) / self._block_size)
        for i in range(n_batches):
            s = slice(i * self._block_size, (i + 1) * self._block_size)
            states = self._get_states(indices[s])
            q_values[s] = self._dqn.predict(states)

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
