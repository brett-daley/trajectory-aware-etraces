from gym.envs import register
from gym_classics.envs.abstract.linear_walk import LinearWalk


register(
    id='19WalkNoOp-v0',
    entry_point='walk19_no_op:Walk19NoOp'
)


class Walk19NoOp(LinearWalk):
    def __init__(self):
        self._length = 19
        self._left_reward = -1.0
        self._right_reward = 1.0
        # NOTE: 3 possible actions here, not just 2
        super(LinearWalk, self).__init__(starts={10}, n_actions=3)

    def _next_state(self, state, action):
        # 0 = move left, 1 = move right, 2 = do nothing
        state += [-1, 1, 0][action]
        next_state = min(max(state, 0), self._length - 1)
        return next_state, 1.0
