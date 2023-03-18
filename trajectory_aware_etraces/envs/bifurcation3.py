from trajectory_aware_etraces.envs.bifurcation1 import Bifurcation1


class Bifurcation3(Bifurcation1):
    layout = """
|G     |
|G     |
|GXX   |
| X    |
|  S   |
|XX    |
"""

    def __init__(self):
        super(Bifurcation1, self).__init__(Bifurcation3.layout)
