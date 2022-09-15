from gym.envs import register


register(
    id='GridWalk-v0',
    entry_point='moretrace.grid_walk.grid_walk:GridWalk'
)
