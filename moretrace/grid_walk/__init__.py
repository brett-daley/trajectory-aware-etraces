from gym.envs import register


register(
    id='Bifurcation-v0',
    entry_point='moretrace.grid_walk.bifurcation:Bifurcation'
)

register(
    id='GridWalk-v0',
    entry_point='moretrace.grid_walk.grid_walk:GridWalk'
)
