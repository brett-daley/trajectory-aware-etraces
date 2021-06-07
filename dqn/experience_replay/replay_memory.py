from collections import deque

import numpy as np


class ReplayMemory:
    def __init__(self, env, capacity, cache_size, discount, return_estimator):
        assert cache_size <= capacity, "cache size cannot be larger than memory capacity"
        self._size_now = 0
        self._capacity = capacity

        self._discount = discount
        self._return_estimator = return_estimator

        self._cache = ReplayCache(cache_size)
        self._completed_episodes = deque()
        self._current_episode = Episode()

    def save(self, state, action, reward, done):
        self._current_episode.append_transition(state, action, reward, done)
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

    def refresh_cache(self):
        self._cache.refresh(self._completed_episodes)


class ReplayCache:
    def __init__(self, capacity):
        self._capacity = capacity
        self._states = None
        self._actions = None
        self._returns = None

    def refresh(self, episode_list):
        states, actions, rewards, dones = [], [], [], []
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

        # Convert to numpy arrays for efficient return calculation and sampling
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        dones = np.stack(dones).astype(np.float32)
        returns = np.zeros_like(rewards)

        # TODO: Compute Q-values and returns here

        # Save the values needed for sampling minibatches
        self._states, self._actions, self._returns = states, actions, returns

    def sample(self, batch_size):
        assert self._states is not None, "replay cache must be refreshed before sampling"
        j = np.random.randint(len(self._states), size=batch_size)
        return (self._states[j], self._actions[j], self._returns[j])


class Episode:
    def __init__(self):
        self.states, self.actions, self.rewards, self.dones = [], [], [], []
        self._already_done = False

    def __len__(self):
        return len(self.states)

    def append_transition(self, state, action, reward, done):
        assert not self._already_done
        self._already_done = done or self._already_done

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
