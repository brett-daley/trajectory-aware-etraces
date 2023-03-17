from gym.envs import register


register(
    id='Bifurcation1-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation1:Bifurcation1'
)

register(
    id='Bifurcation2-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation2:Bifurcation2'
)
