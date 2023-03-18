from gym.envs import register


register(
    id='Bifurcation1-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation1:Bifurcation1'
)

register(
    id='Bifurcation2-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation2:Bifurcation2'
)

register(
    id='Bifurcation3-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation3:Bifurcation3'
)

register(
    id='Bifurcation4-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcation4:Bifurcation4'
)
