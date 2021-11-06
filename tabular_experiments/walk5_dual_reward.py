from gym.envs import register
from gym_classics.envs.abstract.linear_walk import LinearWalk


register(
    id='5WalkDualReward-v0',
    entry_point='walk5_dual_reward:Walk5DualReward'
)


class Walk5DualReward(LinearWalk):
    def __init__(self):
        self._length = 5
        self._left_reward = 1.0
        self._right_reward = 1.0
        super(LinearWalk, self).__init__(starts={10}, n_actions=2)
