from collections import deque

import cv2
import gym
from gym.envs.atari.atari_env import AtariEnv
from gym.spaces import Box
import numpy as np

from dqn.auto_monitor import AutoMonitor


def make(game):
    env = AtariEnv(game, frameskip=4, obs_type='image')
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    # To avoid miscounts, monitor must come before episodic life reset and reward clipping
    env = AutoMonitor(env)
    env = EpisodicLifeWrapper(env)
    env = ClippedRewardWrapper(env)
    env = PreprocessImageWrapper(env)
    env = HistoryWrapper(env, history_len=4)
    return env


class ClippedRewardWrapper(gym.RewardWrapper):
    """Clips rewards to be in {-1, 0, +1} based on their signs."""
    def reward(self, reward):
        return np.sign(reward)


class EpisodicLifeWrapper(gym.Wrapper):
    """Signals done when a life is lost, but only resets when the game ends."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # We lost a life, but force a reset only if it's not game over.
            # Otherwise, the environment just handles it automatically.
            done = True
        self.lives = lives
        return self.observation, reward, done, info

    def reset(self):
        if self.was_real_done:
            self.observation = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return self.observation


class FireResetWrapper(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        super().__init__(env)

    def reset(self):
        self.env.reset()
        observation, _, _, _ = self.step(1)
        return observation


class HistoryWrapper(gym.Wrapper):
    """Stacks the previous `history_len` observations along their last axis.
    Pads observations with zeros at the beginning of an episode."""
    def __init__(self, env, history_len=4):
        assert history_len > 1
        super().__init__(env)
        self.history_len = history_len
        self.deque = deque(maxlen=history_len)

        self.shape = self.observation_space.shape
        self.dtype = self.observation_space.dtype
        self.observation_space.shape = (*self.shape[:-1], history_len * self.shape[-1])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.deque.append(observation)
        return self._history(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._clear()
        self.deque.append(observation)
        return self._history()

    def _history(self):
        return np.concatenate(list(self.deque), axis=-1)

    def _clear(self):
        for _ in range(self.history_len):
            self.deque.append(np.zeros(self.shape, dtype=self.dtype))


class PreprocessImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, interpolation='nearest'):
        super().__init__(env)
        self._shape = (84, 84, 1)
        self.observation_space = Box(low=0, high=255, shape=self._shape, dtype=np.uint8)
        self._interpolation = getattr(cv2, 'INTER_' + interpolation.upper())

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        return self._resize(observation).reshape(self._shape)

    def _resize(self, observation):
        return cv2.resize(observation, self._shape[:2][::-1], interpolation=self._interpolation)
