from argparse import ArgumentParser
import itertools
import os

from gym.spaces import Discrete
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from dqn import atari_env
from dqn.deep_q_network import DeepQNetwork
from dqn.experience_replay.replay_memory import ReplayMemory

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DQNAgent:
    def __init__(self, env, return_estimator='Qlambda-0', **kwargs):
        assert isinstance(env.action_space, Discrete)
        self._env = env

        optimizer = Adam(lr=5e-5, epsilon=1e-8)
        self._dqn = DeepQNetwork(env, optimizer)
        self._replay_memory = ReplayMemory(self._dqn, capacity=1_000_000,
                                           cache_size=80_000, block_size=40_000,
                                           discount=0.99, return_estimator=return_estimator)

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

        # Ensure that the cache gets refreshed before training starts
        assert self._prepopulate % self._target_update_freq == 0

        # Parameter for the epsilon-greedy exploration policy
        # Updated whenever the replay memory cache is refreshed
        self._epsilon = self._epsilon_schedule(t=1)

    def policy(self, t, state):
        assert t > 0, "timestep must start at 1"
        # With probability epsilon, take a random action
        if np.random.rand() < self._epsilon:
            return self._env.action_space.sample()
        # Else, take the predicted best action (greedy)
        return self._greedy_action(state)

    def _greedy_action(self, state):
        Q = self._dqn.predict(state[None])[0]
        return np.argmax(Q)

    def _epsilon_schedule(self, t):
        if t <= self._prepopulate:
            return 1.0
        t -= self._prepopulate
        epsilon = 1.0 - 0.9 * (t / 1_000_000)
        return max(epsilon, 0.1)

    def update(self, t, state, action, reward, done, next_state):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done, self._epsilon)

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % self._target_update_freq == 1:
            self._epsilon = self._epsilon_schedule(t)
            self._replay_memory.refresh_cache(self._epsilon)

        if t % self._train_freq == 1:
            minibatch = self._replay_memory.sample(self._batch_size)
            self._dqn.train(*minibatch)


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--return-estimator', type=str, default='Qlambda-0')
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


def train(env, agent, timesteps):
    state = env.reset()

    for t in itertools.count(start=1):
        if t >= timesteps and done:
            env.close()
            break

        action = agent.policy(t, state)
        next_state, reward, done, _ = env.step(action)
        agent.update(t, state, action, reward, done, next_state)
        state = env.reset() if done else next_state


def main(kwargs):
    game = kwargs['game']
    seed = kwargs['seed']
    timesteps = kwargs['timesteps']

    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(game)
    env.seed(seed)
    env.action_space.seed(seed)

    agent = DQNAgent(env, **kwargs)
    train(env, agent, timesteps)


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(kwargs)
