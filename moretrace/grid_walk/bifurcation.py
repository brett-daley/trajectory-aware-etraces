from gym_classics.envs.abstract.noisy_gridworld import NoisyGridworld as Gridworld


class Bifurcation(Gridworld):
    def __init__(self):
        self._goal = (4, 2)
        blocks = {
            # Left rectangle
            (0,1), (0,2), (0,3), (0,4),
            (1,1), (1,2), (1,3), (1,4),
            # Right wall
            (3,1), (3,2), (3,3)
        }
        super().__init__(n_actions=4, dims=(5, 5), starts={(0, 0)}, blocks=blocks)

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
