from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld


class GridWalk(NoisyGridworld):
    def __init__(self):
        self._goal = (9, 3)
        super().__init__(dims=(10, 7), starts={(0, 3)})

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
