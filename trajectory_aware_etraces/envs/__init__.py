from gym.envs import register


register(
    id='BifurcatedGridworld-v0',
    entry_point='trajectory_aware_etraces.envs.bifurcated_gridworld:BifurcatedGridworld'
)
