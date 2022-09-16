import gym
import numpy as np


class EnvSampler:
    def __init__(self, env_id, seed):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.np_random = np.random.RandomState(seed)

    def sample_episodes(self, behavior_policy, n_episodes):
        return tuple(
            self.sample_one_episode(behavior_policy) for _ in range(n_episodes)
        )

    def sample_one_episode(self, behavior_policy):
        state = self.env.reset()
        done = False
        transitions = []
        while not done:
            action = self.np_random.choice(self.env.action_space.n, p=behavior_policy(state))
            next_state, reward, done, _ = self.env.step(action)
            transitions.append( (state, action, reward, next_state, done) )
            state = next_state
        return tuple(transitions)
