from gym.envs import register


register(
    id='GridWalk-v0',
    entry_point='grid_walk.grid_walk:GridWalk'
)
