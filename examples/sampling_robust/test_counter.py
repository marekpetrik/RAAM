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
success = 0.5
poscount = len(rewards)
np.random.seed(0)
counter = CounterModes(rewards,modecount,success)
actions = counter.actions(counter.initstates().__next__())
decstatecount = poscount * modecount

# define test parameters
test_steps = 20
test_counts = 20

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

## Compute optimal policy

# get samples
np.random.seed(0)
samples = counter.simulate(1000, counter.random_policy(),20)

# build the sampled MDP    
r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())
#r.rmdp.set_uniform_distributions(0.0)


# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac(5000,stype=robust.SolutionType.Average.value)

optdecvalue = r.decvalue(decstatecount, v)
optdecpolicy = r.decpolicy(decstatecount, pol)

print('Optimal value function', optdecvalue.reshape(modecount, poscount).T)
print('Optimal policy', optdecpolicy.reshape(modecount, poscount).T)

returneval = counter.simulate(50, raam.vec2policy(optdecpolicy, actions, decstatenum),120)
print('Optimal return', returneval.statistics(counter.discount)['mean_return'])

## Compute the baseline policy

# get samples
counter.set_acceptable_modes(1)
samples = counter.simulate(1000, counter.random_policy(),20)
counter.set_acceptable_modes(modecount)

# build the sampled MDP    
r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())
#r.rmdp.set_uniform_distributions(0.0)

# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac(1000,stype=robust.SolutionType.Average.value)

basedecvalue = r.decvalue(decstatecount, v)
basedecpolicy = r.decpolicy(decstatecount, pol).reshape(modecount, poscount)

for i in range(1, modecount):
    basedecpolicy[i,:] = basedecpolicy[0,:] 

basedecpolicy = basedecpolicy.reshape(-1)

print('Baseline value function', basedecvalue.reshape(modecount, poscount).T)
print('Policy', basedecpolicy.reshape(modecount, poscount).T)

returneval = counter.simulate(50, raam.vec2policy(basedecpolicy, actions, decstatenum),120)
print('Baseline optimal return', returneval.statistics(counter.discount)['mean_return'])

# TODO: write a test that takes samples and then computes the outcomes across the states to make sure that they are equal

## Compute a regular solution 


np.random.seed(0)
samples = counter.simulate(test_steps, counter.random_policy(),test_counts)

r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())

# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac(1000,stype=robust.SolutionType.Average.value)

expdecvalue = r.decvalue(decstatecount, v)
expdecpolicy = r.decpolicy(decstatecount, pol)
# use a default action when there were no samples for it
expdecpolicy[np.where(expdecpolicy < 0)] = 1

# compute the number of samples for each expectation state
returneval = counter.simulate(50, raam.vec2policy(expdecpolicy, actions, decstatenum),120)
print('Expected value of the expectation policy', expdecvalue[0])
print('Return of expectation policy', returneval.statistics(counter.discount)['mean_return'])

## Compute a reward adjusted solution

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.1 / np.sqrt(samples)

maxr = np.max(rewards)

np.random.seed(0)
samples = counter.simulate(test_steps, counter.random_policy(),test_counts)

r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())

# compute the number of samples for each expectation state
expstateinds = r.expstate_numbers()
samplecounts = [r.rmdp.outcome_count(es,0) for es in expstateinds]

# adjust the rewards *******
#   every transition is now treated as an outcome (= one sample per outcome)
#   every expectation state has only one action
for es, scount in zip(expstateinds, samplecounts):
    for out in range(r.rmdp.outcome_count(es,0)):
        r.rmdp.set_reward(es,0,out,0, 
            r.rmdp.get_reward(es,0,out,0) - 
            maxr * counter.discount / (1 - counter.discount) * err(scount) )

# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac(1000,stype=robust.SolutionType.Average.value)

expadjdecvalue = r.decvalue(decstatecount, v)
expadjdecpolicy = r.decpolicy(decstatecount, pol)
# use a default action when there were no samples for it
expadjdecpolicy[np.where(expadjdecpolicy < 0)] = 1

# compute the number of samples for each expectation state
returneval = counter.simulate(50, raam.vec2policy(expadjdecpolicy, actions, decstatenum),120)
print('Expected value of the expectation policy', expadjdecvalue[0])
print('Return of expectation policy', returneval.statistics(counter.discount)['mean_return'])

## Compute a robust solution

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.4 / np.sqrt(samples)


np.random.seed(0)
samples = counter.simulate(test_steps, counter.random_policy(),test_counts)

r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=features.IdCache())

# compute the number of samples for each expectation state
expstateinds = r.expstate_numbers()
decstateinds = r.decstate_numbers()
samplecounts = [r.rmdp.outcome_count(es,0) for es in expstateinds]

r.rmdp.set_uniform_distributions(1.0)

# *******
# add transitions to all decision states (even if not sampled)
#   IMPORTANT: needs to come after getting sample counts
for es,scount in zip(expstateinds, samplecounts):
    dist = [1.0] * scount
    for ds in decstateinds:
        r.rmdp.add_transition(es,0,r.rmdp.outcome_count(es,0),ds,0.0,0.0)
        dist.append(0)
    dist = np.array(dist)
    dist = dist / dist.sum()
    r.rmdp.set_distribution(es,0,dist,err(scount))
    
# solve sampled MDP
v,pol,_,_ = r.rmdp.mpi_jac_l1(100,stype=robust.SolutionType.Robust.value)

robdecvalue = r.decvalue(decstatecount, v)
robdecpolicy = r.decpolicy(decstatecount, pol)
# use a default action when there were no samples for it
robdecpolicy[np.where(robdecpolicy < 0)] = 1

# compute the number of samples for each expectation state
returneval = counter.simulate(50, raam.vec2policy(robdecpolicy, actions, decstatenum),200)
print('Expected value of the expectation policy', robdecvalue[0])
print('Return of expectation policy', returneval.statistics(counter.discount)['mean_return'])

## Compute the jointly optimized solution


