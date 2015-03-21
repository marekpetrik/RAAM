# cd Projects/raam/scripts

import raam
import raam.examples
import numpy as np
import configuration

from raam import crobust

## Get Samples

horizon = 600
runs = 5

config = configuration.construct_martingale(np.arange(3), 3)

# Assume that the battery capacity does not change
config['change_capacity'] = False

sim = raam.examples.inventory.Simulator(config)     

random_samples = sim.simulate(horizon, sim.random_policy(), runs)

print('Random policy:', random_samples.statistics(sim.discount)['mean_return'])

samples = random_samples

## Build Approximation Properties

def decmap(s):
    soc,capacity,priceindex = s
    return (soc,priceindex)
    
def expmap(s):
    soc,capacity,priceindex,reward = s
    return (soc,capacity,reward)

narrow_samples = raam.samples.SampleView(samples,
                    decmap = decmap,
                    expmap = expmap ,
                    actmap=raam.identity)

discretization = 50

#** for narrow samples
# decision state: soc, priceindex
decagg_big = raam.features.GridAggregation( \
                    ((0,config['initial_capacity']), (0,config['price_sell'].shape[0])),\
                    (discretization, config['price_sell'].shape[0]) )

# expectation state: soc, priceindex, reward 
expagg = raam.features.GridAggregation( \
                    ((0,config['initial_capacity']), (0,config['price_sell'].shape[0]), (-2.5,2.5)), \
                    (3*discretization, config['price_sell'].shape[0], 200) )


## Build SRMDP

dab = lambda x: decagg_big.classify(x,True)     

das = lambda x: 0

ea = lambda x: expagg.classify(x,True)
aa = lambda x : x

srmdp = crobust.SRoMDP(0,sim.discount)
srmdp.from_samples(narrow_samples,dab,das,ea,aa)

## Solve RMDP

rmdp = srmdp.rmdp

print('Optimizing robust policy ...')
rmdp.set_uniform_distributions(1.0)
rob_valfun,rob_policy_vec,rob_residual,rob_iterations = rmdp.mpi_jac_l1(300,maxresidual=1e-4,stype=0)
print('Residual', rob_residual)

print('Optimizing average policy ...')
rmdp.set_uniform_distributions(0.0)
avg_valfun,avg_policy_vec,avg_residual,avg_iterations = rmdp.mpi_jac(300,maxresidual=1e-4, stype=2)
print('Residual', avg_residual)

print('Optimizing optimistic policy ...')
rmdp.set_uniform_distributions(1.0)
opt_valfun,opt_policy_vec,opt_residual,opt_iterations = rmdp.mpi_jac_l1(300,maxresidual=1e-4, stype=1)
print('Residual', opt_residual)


print('Constructing policies ...')
rob_policy_vec_dec = srmdp.decpolicy(len(decagg_big), rob_policy_vec)
opt_policy_vec_dec = srmdp.decpolicy(len(decagg_big), opt_policy_vec)
avg_policy_vec_dec = srmdp.decpolicy(len(decagg_big), avg_policy_vec)

rob_policy = raam.vec2policy(rob_policy_vec_dec, [0,1], lambda x: dab(decmap(x)),0)
opt_policy = raam.vec2policy(opt_policy_vec_dec, [0,1], lambda x: dab(decmap(x)),0)
avg_policy = raam.vec2policy(avg_policy_vec_dec, [0,1], lambda x: dab(decmap(x)),0)
 
## Evaluate the computed policy

optimized_samples = sim.simulate(horizon, rob_policy, runs)

print('Optimized policy:', optimized_samples.statistics(sim.discount)['mean_return'])