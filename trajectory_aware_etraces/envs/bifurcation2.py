from trajectory_aware_etraces.envs.bifurcation1 import Bifurcation1


class Bifurcation2(Bifurcation1):
    layout = """
|  G    |
| X X X |
| X X X |
| X X X |
| X X X |
| X X X |
|   S   |
"""

    def __init__(self):
        super(Bifurcation1, self).__init__(Bifurcation2.layout)
