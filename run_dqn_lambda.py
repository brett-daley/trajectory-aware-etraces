from dqn.experience_replay.old_replay_memory import OldReplayMemory
from main import DQNAgent, parse_kwargs, setup_env, train


class OldDQNAgent(DQNAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, replay_memory_cls=OldReplayMemory, **kwargs)

    def update(self, t, state, action, reward, done, mu):
        assert t > 0, "timestep must start at 1"
        self._replay_memory.save(state, action, reward, done, mu)

        if t <= self._prepopulate:
            # We're still pre-populating the replay memory
            return

        if t % self._target_update_freq == 1:
            epsilon = self._epsilon_schedule(t)
            self._replay_memory.refresh_cache(epsilon)

        if t % self._train_freq == 1:
            minibatch = self._replay_memory.sample(self._batch_size)
            self._dqn.train(*minibatch)


def main(kwargs):
    game = kwargs['game']
    seed = kwargs['seed']
    timesteps = kwargs['timesteps']

    env = setup_env(game, seed)
    agent = OldDQNAgent(env, **kwargs)
    train(env, agent, timesteps)


if __name__ == '__main__':
    kwargs = parse_kwargs()
    main(kwargs)
