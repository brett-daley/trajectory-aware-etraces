from trajectory_aware_etraces.envs.bifurcation1 import Bifurcation1


class Bifurcation4(Bifurcation1):
    layout = """
|       |
|       |
|  XXX  |
|   G   |
|  XXX  |
|       |
|S      |
"""

    def __init__(self):
        super(Bifurcation1, self).__init__(Bifurcation4.layout)
