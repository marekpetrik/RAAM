# cd Projects/raam/scripts

import raam
import raam.examples
import numpy as np
import configuration

horizon = 600
runs = 5

config = configuration.construct_martingale(np.arange(3), 3)

sim = raam.examples.inventory.Simulator(config)     

 sim.random_policy()