import numpy as np

from dqn.experience_replay.replay_memory import ReplayMemory


class AblatedReplayMemory(ReplayMemory):
    def _sample_episodes_for_cache(self):
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
                assert 0 < len(indices) <= self._cache_size
                return np.array(sorted(indices), dtype=np.int32)

            # Add all transitions from this episode to the cache
            assert self._dones[self._absolute(end)]
            indices.extend(list(range(start, end + 1)))
