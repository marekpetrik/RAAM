"""
Optimizing approximate dynamic programming (:mod:`raam`)
=============================================================

Core optimization module that handles sample processing and facilitates simulation and optimization.

It directly imports:
    - :mod:`raam.samples`
    - :mod:`raam.matrix`

To determine the package version, call ::
    >>> import raam
    >>> raam.version.full_version

"""
from . import samples
from .samples import *
from . import simulator
from .simulator import *

from . import version
