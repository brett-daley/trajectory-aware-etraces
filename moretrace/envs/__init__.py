from gym.envs import register


register(
    id='BifurcatedGridworld-v0',
    entry_point='moretrace.envs.bifurcated_gridworld:BifurcatedGridworld'
)
