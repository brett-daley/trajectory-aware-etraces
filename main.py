from argparse import ArgumentParser
import itertools
import os

from gym import Wrapper
from gym.spaces import Discrete
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from dqn import atari_env
from dqn.deep_q_network import DeepQNetwork
from dqn.experience_replay import AblatedReplayMemory, ReplayMemory, OldReplayMemory
from dqn.experience_replay.traces import epsilon_greedy_probabilities

os.environ['TF_DETERMINISTIC_OPS'] = '1'


class DQNAgent:
    def __init__(self, env, **kwargs):
        assert isinstance(env.action_space, Discrete)
        self._env = env
        self._timesteps = kwargs['timesteps']

        optimizer = Adam(lr=5e-5, epsilon=1e-8)
        self._dqn = DeepQNetwork(env, optimizer)

        replay_memory_cls = {
            'new': ReplayMemory,
            'old': OldReplayMemory,
            'ablated': AblatedReplayMemory,
        }[kwargs['rmem']]
        self.replay_memory = replay_memory_cls(
            self._dqn, capacity=1_000_000, cache_size=160_000,
            discount=0.99, lambd=kwargs['lambd'], return_estimator=kwargs['return_estimator'])

        self._prepopulate = 50_000
        self._train_freq = 4
        self._batch_size = 32
        self._target_update_freq = 10_000

        # Ensure that the cache gets refreshed before training starts
        assert self._prepopulate % self._target_update_freq == 0

        # Compute number of minibatches to conduct per "epoch" (i.e. target net update)
        assert self._target_update_freq % self._train_freq == 0
        self._batches_per_epoch = self._target_update_freq // self._train_freq

    def policy(self, t, state):
        assert t > 0, "timestep must start at 1"
        epsilon = self._epsilon_schedule(t)
        Q = self._dqn.predict(state[None])[0]
        mu = epsilon_greedy_probabilities(Q, epsilon)

        # With probability epsilon, take a random action
        if np.random.rand() < epsilon:
            return self._env.action_space.sample(), mu
        # Else, take the predicted best action (greedy)
        return np.argmax(Q), mu

    def _epsilon_schedule(self, t):
        # Linear interpolation schedule
        points = [(0, 1.0),
                  (self._prepopulate, 1.0),
                  (self._prepopulate + 1_000_000, 0.1),
                  (self._timesteps, 0.05)]
        segments = zip(points[:-1], points[1:])

        for (t_start, eps_start), (t_end, eps_end) in reversed(list(segments)):
            assert t_end > t_start
            if t >= t_start:
                frac_elapsed = (t - t_start) / (t_end - t_start)
                return eps_start + frac_elapsed * (eps_end - eps_start)
        raise ValueError(f"timestep {t} is not in the schedule")

    def update(self, t, state, action, reward, done, mu):
        assert t > 0, "timestep must start at 1"
        self.replay_memory.save(state, action, reward, done, mu)

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % self._target_update_freq == 1:
            epsilon = self._epsilon_schedule(t)
            self.replay_memory.refresh_cache(epsilon)

            for minibatch in self.replay_memory.iterate_cache(self._batches_per_epoch, self._batch_size):
                self._dqn.train(*minibatch)


def allow_gpu_memory_growth():
    try:
        gpu_list = tf.config.list_physical_devices('GPU')
    except AttributeError:
        gpu_list = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)


def parse_kwargs():
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default='pong')
    parser.add_argument('--rmem', type=str, default='new')
    parser.add_argument('--lambd', type=float, default=0.0)
    parser.add_argument('--return-estimator', type=str, default='Qlambda')
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--seed', type=int, default=0)
    return vars(parser.parse_args())


def setup_env(game, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = atari_env.make(game)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def train(env, agent, timesteps):
    observation = env.reset()

    for t in itertools.count(start=1):
        if t >= timesteps and real_done:
            env.close()
            break

        state = agent.replay_memory.get_state(observation)
        action, mu = agent.policy(t, state)
        next_observation, reward, done, _ = env.step(action)
        agent.update(t, observation, action, reward, done, mu)

        real_done = was_real_done(env, done)
        observation = env.reset() if real_done else next_observation


def was_real_done(env, done):
    if isinstance(env, atari_env.EpisodicLifeWrapper):
        return env.was_real_done
    elif isinstance(env, Wrapper):
        return was_real_done(env.env, done)
    return done


def main(kwargs):
    game = kwargs['game']
    seed = kwargs['seed']
    timesteps = kwargs['timesteps']

    env = setup_env(game, seed)
    agent = DQNAgent(env, **kwargs)
    train(env, agent, timesteps)


if __name__ == '__main__':
    allow_gpu_memory_growth()
    kwargs = parse_kwargs()
    main(kwargs)
