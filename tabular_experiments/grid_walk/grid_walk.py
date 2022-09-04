from gym_classics.envs.abstract.gridworld import Gridworld


class GridWalk(Gridworld):
    def __init__(self):
        self._goal = (9, 1)
        super().__init__(dims=(10, 3), starts={(0, 1)})

    def _move(self, state, action):
        x, y = state
        return {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x,   y)     # Instead of Left, do a No-op
        }[action]

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return next_state == self._goal
