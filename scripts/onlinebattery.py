"""
Runs online optimization of the degradable battery problem
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

discretization = 3
discount = 0.8

print('Simulation parameter bounds: horizon %d, runs %d, samples %d, eval samples %d' \
        % (horizon,runs,sample_count,eval_sample_count))

## Solve offline with extra samples

# construct an initial random policy
sim, create_srmdp, create_policy = examples.inventory.makesimulator(discretization,discount)

policy = sim.random_policy()

print('Sampling ...')
# transition limit makes sure that the simulation is fair in terms of the number of samples
samples = sim.simulate(horizon, policy, runs, transitionlimit=sample_count)
print('Return from samples:', samples.statistics(sim.discount)['mean_return'])

srmdp = create_srmdp(samples)
rmdp = srmdp.rmdp

print('Solving aggregated MDP...')
valfun,policy_vec,residual,iterations = rmdp.mpi_jac(300,maxresidual=1e-4,stype=0)

print('Constructing policies ...')
policy = create_policy(srmdp,policy_vec)

#TODO: Why is this consistently lower than the first value?
print('Evaluating ...')
samples = sim.simulate(horizon, policy, runs, probterm=1-sim.discount, \
                        transitionlimit=eval_sample_count)
print('Evaluation:', samples.statistics(1)['mean_return'])

## Solve online with limiting weights

# construct an initial random policy
sim, create_srmdp, create_policy = examples.inventory.makesimulator(discretization,discount)

policy = sim.random_policy()

for i in range(5):
    print('Sampling ...')
    # transition limit makes sure that the simulation is fair in terms of the number of samples
    samples = sim.simulate(horizon, policy, runs, transitionlimit=sample_count)
    print('Return from samples:', samples.statistics(sim.discount)['mean_return'])
    
    srmdp = create_srmdp(samples)
    rmdp = srmdp.rmdp
    
    print('Solving aggregated MDP...')
    valfun,policy_vec,residual,iterations = rmdp.mpi_jac(300,maxresidual=1e-4,stype=0)
    
    print('Constructing policies ...')
    policy = create_policy(srmdp,policy_vec)
    
    #TODO: Why is this consistently lower than the first value?
    print('Evaluating ...')
    samples = sim.simulate(horizon, policy, runs, probterm=1-sim.discount, \
                            transitionlimit=eval_sample_count)
    print('Evaluation:', samples.statistics(1)['mean_return'])

## Solve online with occupancy frequency weights

# construct an initial random policy
sim, create_srmdp, create_policy = examples.inventory.makesimulator(discretization,discount)

policy = sim.random_policy()

for i in range(5):
    print('Sampling ...')
    # transition limit makes sure that the simulation is fair in terms of the number of samples
    samples = sim.simulate(horizon, policy, runs, probterm=1-sim.discount, \
                        transitionlimit=sample_count)
    print('Return from samples:', samples.statistics(1)['mean_return'])
    
    srmdp = create_srmdp(samples)
    rmdp = srmdp.rmdp
    
    print('Solving aggregated MDP...')
    valfun,policy_vec,residual,iterations = rmdp.mpi_jac(300,maxresidual=1e-4,stype=0)
    
    print('Constructing policies ...')
    policy = create_policy(srmdp,policy_vec)
    
    print('Evaluating ...')
    samples = sim.simulate(horizon, policy, runs, probterm=1-sim.discount, \
                            transitionlimit=eval_sample_count)
    print('Evaluation:', samples.statistics(1)['mean_return'])

## Optimize thresholds

sim, _, _ = examples.inventory.makesimulator(discretization,discount)
examples.inventory.optimize_independently(sim, horizon=100)