#!/bin/python
"""
Runs a simulation optimization method to compute 2-threshold policies
"""
import numpy as np
import raam
from raam import examples
from raam import crobust

## Configuration

horizon = 2000                  # maximal simulation horizon
runs = 1000                     # maximal number of runs
sample_count = 10000            # total maximal number of samples for constructing models
eval_sample_count = 10000       # total maximal number of samples for constructing models

discount = 0.8      

discretization = 3

extra_exp_samples = 5           # number of additional expectation transitions sampled

print('Simulation parameter bounds: horizon %d, runs %d, samples %d, eval samples %d' \
        % (horizon,runs,sample_count,eval_sample_count))
        
## Optimize thresholds

sim, _, _ = examples.inventory.makesimulator(discretization,discount)
examples.inventory.optimize_independently(sim, horizon=horizon)