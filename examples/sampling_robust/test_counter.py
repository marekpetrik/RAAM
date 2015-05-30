import raam
from raam import robust
from raam import crobust
from raam import features
from raam import examples
import numpy as np
from counter_modes import CounterModes

# define counter
rewards = np.array([-1,1,2,3,2,1,-1,-2,-3,3,4,5])
modecount = 3
poscount = len(rewards)
np.random.seed(0)
counter = CounterModes(rewards,modecount,0.2)
actions = counter.actions(counter.initstates().__next__())
decstatecount = poscount * modecount

## Define Aggregation

# define state aggregation functions
def decstatenum(x): 
    pos,mode = x
    return pos + mode * poscount
    
def expstatenum(x):    
    (pos,mode),action = x
    numaction = actions.index(action)
    assert numaction >= 0
    return pos + mode * poscount + decstatecount*numaction
    
zero = lambda x: 0
dict_si = {}
def sampleindex(x):
    x = expstatenum(x)
    index = dict_si.get(x,0)
    dict_si[x] = index + 1
    return index

## Compute Optimal Policy

# get samples
samples = counter.simulate(1000, counter.random_policy(),30)

# build the sampled MDP    
r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())
r.rmdp.set_uniform_distributions(0.0)


# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac_l1(5000,stype=robust.SolutionType.Robust.value)

optdecvalue = r.decvalue(decstatecount, v)
optdecpolicy = r.decpolicy(decstatecount, pol)

print('Optimal value function', optdecvalue.reshape(modecount, poscount).T)
print('Optimal policy', optdecpolicy.reshape(modecount, poscount).T)

returneval = counter.simulate(200, raam.vec2policy(optdecpolicy, actions, decstatenum),30)
print('Optimal return', returneval.statistics(counter.discount)['mean_return'])

## Compute the baseline policy

# get samples
counter.set_acceptable_modes(1)
samples = counter.simulate(2000, counter.random_policy(),6)

# build the sampled MDP    
r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())
r.rmdp.set_uniform_distributions(0.0)

# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac_l1(1000,stype=robust.SolutionType.Robust.value)

basedecvalue = r.decvalue(decstatecount, v)
basedecpolicy = r.decpolicy(decstatecount, pol).reshape(modecount, poscount)

for i in range(1, modecount):
    basedecpolicy[i,:] = basedecpolicy[0,:] 

basedecpolicy = basedecpolicy.reshape(-1)

print('Baseline value function', basedecvalue.reshape(modecount, poscount).T)
print('Policy', basedecpolicy.reshape(modecount, poscount).T)

returneval = counter.simulate(200, raam.vec2policy(basedecpolicy, actions, decstatenum),30)
print('Baseline optimal return', returneval.statistics(counter.discount)['mean_return'])