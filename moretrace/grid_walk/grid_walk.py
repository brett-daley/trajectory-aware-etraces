from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld as Gridworld


class GridWalk(Gridworld):
    def __init__(self):
        N = 7
        self._goal = (N-1, N-1)
        super().__init__(n_actions=4, dims=(N, N), starts={(0, 0)})

    def _move(self, state, action):
        x, y = state
        return {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x-1, y)     # Left
        }[action]

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return state == self._goal
