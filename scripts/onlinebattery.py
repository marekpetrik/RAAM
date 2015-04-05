"""
Runs online optimization of the degradable battery problem
"""

## Configuration
# cd Projects/raam/scripts

import raam
import raam.examples
import numpy as np
import configuration
from raam import crobust

import online

horizon = 3000              # maximal simulation horizon
runs = 5                    # maximal number of runs
sample_count = 4000         # total maximal number of samples

## Construct simulator

print('Simulation parameter bounds: horizon %d, runs %d, samples %d' % (horizon,runs,sample_count))

# problem configuration
config = configuration.construct_martingale(np.arange(5), 5)
config['change_capacity'] = True
sim = raam.examples.inventory.Simulator(config,action_step=0.1,discount=0.8)

# construct the set of possible actions to map them to indexes
all_actions = sim.all_actions()

## Construct aggregation parameters

def decmap(s):
    soc,capacity,priceindex = s
    assert type(priceindex) == int or type(priceindex) == np.int64
    return (soc,priceindex)
    
def expmap(s):
    soc,capacity,priceindex,reward = s
    return (soc,capacity,reward)

epsilon = 1e-6

discretization = 3

#** for narrow samples
# decision state: soc, priceindex
decagg_big = raam.features.GridAggregation( \
                    ((0,config['initial_capacity']+epsilon), (0,config['price_sell'].shape[0])),\
                    (discretization, config['price_sell'].shape[0]) )

# expectation state: soc, priceindex, reward 
expagg = raam.features.GridAggregation( \
                    ((0,config['initial_capacity']+epsilon), (0,config['price_sell'].shape[0]), (-5,5)), \
                    (3*discretization, config['price_sell'].shape[0], 200) )

# decision states
dab = lambda x: decagg_big.classify(x,True)     
# expectation states
ea = lambda x: expagg.classify(x,True)
# used to represent the worst case for decision states
das = lambda x: 0
# used to represent the worst case for decision states
aa = lambda x : np.argmin(np.abs(all_actions - x))

policy = sim.random_policy()

## Solve by regular simulation

# * simulate

print('Sampling ...')
samples = sim.simulate(horizon, policy, runs)
print('Return from samples:', samples.statistics(sim.discount)['mean_return'])

narrow_samples = raam.samples.SampleView(samples,
                    decmap = decmap,
                    expmap = expmap ,
                    actmap=raam.identity)


srmdp = crobust.SRoMDP(0,sim.discount)
srmdp.from_samples(narrow_samples,decagg_big=dab,decagg_small=das,expagg=ea,actagg=aa)
rmdp = srmdp.rmdp

print('Solving aggregated MDP...')
valfun,policy_vec,residual,iterations = rmdp.mpi_jac(300,maxresidual=1e-4,stype=0)

print('Constructing policies ...')
policy_vec_dec = srmdp.decpolicy(len(decagg_big), policy_vec)
policy = raam.vec2policy(policy_vec_dec, all_actions, lambda x: dab(decmap(x)),0)


## Solve by discounted simulation

# * simulate

print('Sampling ...')
samples = sim.simulate(1000, policy, runs, probterm=1-sim.discount)
print('Return from samples:', samples.statistics(1)['mean_return'])

narrow_samples = raam.samples.SampleView(samples,
                    decmap = decmap,
                    expmap = expmap ,
                    actmap=raam.identity)


srmdp = crobust.SRoMDP(0,sim.discount)
srmdp.from_samples(narrow_samples,decagg_big=dab,decagg_small=das,expagg=ea,actagg=aa)
rmdp = srmdp.rmdp

print('Solving aggregated MDP...')
valfun,policy_vec,residual,iterations = rmdp.mpi_jac(300,maxresidual=1e-4,stype=0)

print('Constructing policies ...')
policy_vec_dec = srmdp.decpolicy(len(decagg_big), policy_vec)
policy = raam.vec2policy(policy_vec_dec, all_actions, lambda x: dab(decmap(x)),0)


## Optimize thresholds

import optimize_thresholds

optimize_thresholds.optimize_independently(sim, horizon=100)