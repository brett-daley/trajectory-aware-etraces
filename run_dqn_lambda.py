from dqn.experience_replay.old_replay_memory import OldReplayMemory
from main import DQNAgent, parse_kwargs, setup_env, train


class OldDQNAgent(DQNAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, replay_memory_cls=OldReplayMemory, **kwargs)


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
