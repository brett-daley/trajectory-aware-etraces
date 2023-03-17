from gym_classics.envs.abstract.gridworld import Gridworld


class Bifurcation1(Gridworld):
    layout = """
|XX   |
|XX X |
|XX XG|
|XX X |
|S    |
"""

    def __init__(self):
        super().__init__(Bifurcation1.layout)

    def _reward(self, state, action, next_state):
        return 1.0 if self._done(state, action, next_state) else 0.0

    def _done(self, state, action, next_state):
        return state in self._goals
