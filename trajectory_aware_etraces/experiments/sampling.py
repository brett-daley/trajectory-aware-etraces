import gym
import numpy as np


class EnvSampler:
    def __init__(self, env_id, seed):
        self.env = gym.make(env_id)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.np_random = np.random.RandomState(seed)
        self._state = None

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

    def step(self, behavior_policy):
        state = self.state
        action = self.np_random.choice(self.env.action_space.n, p=behavior_policy(state))
        next_state, reward, done, _ = self.env.step(action)
        if done:
            next_state = self.env.reset()
        transition = (state, action, reward, next_state, done)
        self.state = next_state
        return transition

    @property
    def state(self):
        state = self._state
        if state is None:
            state = self.env.reset()
        return state

    @state.setter
    def state(self, new_state):
        try:
            self._state = new_state.copy()
        except AttributeError:
            self._state = new_state
