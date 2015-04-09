"""
=============================================================================
Energy arbitrage with degradable storage (:mod:`raam.examples.inventory`)
=============================================================================

Models an invetory problem with multiple price levels, sales, purchases. The storage
is assumed to degrade with use. This can be used to model battery energy storage
with the battery degrading while in use.
"""
from . import inventory
from .inventory import *
from . import configuration
