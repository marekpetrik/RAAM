import raam
from raam import robust
from raam import crobust
from raam import features
from raam import examples
import numpy as np
from counter_modes import CounterModes
import random

import srobust

# define counter
rewards = np.array([-1,1,2,3,2,1,-1,-2,-3,3,4,5])
modecount = 3
success = [0.1,0.8,0.3]
poscount = len(rewards)
np.random.seed(0)
counter = CounterModes(rewards,modecount,success)
actions = counter.actions(counter.initstates().__next__())
decstatecount = poscount * modecount

# define test parameters
test_steps = 50
test_counts = 40

# Define state numbering
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

def actionagg(act):
    return actions.index(act)

## Optimal policy

# get samples
np.random.seed(20)
random.seed(20)
samples = counter.simulate(1000, counter.random_policy(),20)

optdecvalue, optdecpolicy = srobust.solve_expectation(samples, counter.discount, decstatecount, \
                            decagg_big=decstatenum, decagg_small=zero, 
                            expagg_big=expstatenum, expagg_small=sampleindex, 
                            actagg=actionagg)

optrv = counter.simulate(50, raam.vec2policy(optdecpolicy, actions, decstatenum), \
                                200).statistics(counter.discount)['mean_return']

print('Optimal value function\n', optdecvalue.reshape(modecount, poscount).T)
print('Optimal policy', optdecpolicy.reshape(modecount, poscount).T)
print('Optimal return', optrv)

## Baseline policy

# get samples
counter.set_acceptable_modes(1)
np.random.seed(30)
random.seed(30)
samples = counter.simulate(1000, counter.random_policy(),10)
counter.set_acceptable_modes(modecount)

basedecvalue, basedecpolicy = srobust.solve_expectation(samples, counter.discount, decstatecount, \
                            decagg_big=decstatenum, decagg_small=zero, 
                            expagg_big=expstatenum, expagg_small=sampleindex, 
                            actagg=actionagg)

# extend the baseline policy back to all states
basedecpolicy = basedecpolicy.reshape(modecount, -1)
for i in range(1, modecount):
    basedecpolicy[i,:] = basedecpolicy[0,:] 
basedecpolicy = basedecpolicy.reshape(-1)
# remove the transitions between modes
basedecpolicy[np.where(basedecpolicy > 1)] = 1

# construct baseline policy
baselinepol_fun = raam.vec2policy(basedecpolicy, actions, decstatenum)

baserv = counter.simulate(50, baselinepol_fun, 200).statistics(counter.discount)['mean_return']

print('Baseline value function\n', basedecvalue.reshape(modecount, -1).T)
print('Policy', basedecpolicy.reshape(modecount, -1).T)
print('Baseline optimal return', baserv)

# TODO: write a test that takes samples and then sums the number of outcomes across states to make sure that they are equal

## Evaluate the regular solution 

np.random.seed(0)
random.seed(0)

for count in range(1,test_counts):
    samples = counter.simulate(test_steps, counter.random_policy(),count)
    
    expdecvalue, expdecpolicy = srobust.solve_expectation(samples, counter.discount, decstatecount, \
                                decagg_big=decstatenum, decagg_small=zero, 
                                expagg_big=expstatenum, expagg_small=sampleindex, 
                                actagg=actionagg)
    expdecpolicy[np.where(expdecpolicy < 0)] = 1
    
    # compute the number of samples for each expectation state
    exprv = counter.simulate(50, raam.vec2policy(expdecpolicy, actions, decstatenum),200).statistics(counter.discount)['mean_return']
    print('Expected value of the expectation policy', expdecvalue[0])
    print('Return of expectation policy', exprv)

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
                actagg=actionagg)

# compute the number of samples for each expectation state
_, expstateinds = r.expstate_numbers()
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
v,pol,_,_,_ = r.rmdp.mpi_jac(1000,stype=robust.SolutionType.Average.value)

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
    return 0.3 / np.sqrt(samples)

np.random.seed(0)
samples = counter.simulate(test_steps, counter.random_policy(),test_counts)

r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=actionagg)

# compute the number of samples for each expectation state
_,expstateinds = r.expstate_numbers()
_,decstateinds = r.decstate_numbers()
samplecounts = [r.rmdp.outcome_count(es,0) for es in expstateinds]

r.rmdp.set_uniform_distributions(1.0)

# *******
# add transitions to all decision states (even if not sampled)
#   IMPORTANT: needs to come after getting sample counts
for es,scount in zip(expstateinds, samplecounts):
    dist = [1.0] * scount
    for ds in decstateinds:
        # TODO: the zero reward here is not exactly right
        r.rmdp.add_transition(es,0,r.rmdp.outcome_count(es,0),ds,0.0,0.0)
        dist.append(0)
    dist = np.array(dist)
    dist = dist / dist.sum()
    r.rmdp.set_distribution(es,0,dist,err(scount))
    
# solve sampled MDP
v,pol,_,_,_ = r.rmdp.mpi_jac_l1(100,stype=robust.SolutionType.Robust.value)

robdecvalue = r.decvalue(decstatecount, v)
robdecpolicy = r.decpolicy(decstatecount, pol)
# use a default action when there were no samples for it
robdecpolicy[np.where(robdecpolicy < 0)] = 1

# compute the number of samples for each expectation state
returneval = counter.simulate(50, raam.vec2policy(robdecpolicy, actions, decstatenum),200)
print('Expected value of the expectation policy', robdecvalue[0])
print('Return of expectation policy', returneval.statistics(counter.discount)['mean_return'])

## Compute the jointly optimized solution (simple baseline approach)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.3 / np.sqrt(samples)

np.random.seed(0)
samples = counter.simulate(test_steps, counter.random_policy(),test_counts)

r = crobust.SRoMDP(1,counter.discount)
r.from_samples(samples, decagg_big=decstatenum, decagg_small=zero,
                expagg_big=expstatenum, expagg_small=sampleindex,
                actagg=actionagg)

# compute the number of samples for each expectation state
_,expstateinds = r.expstate_numbers()
decstatenums,decstateinds = r.decstate_numbers()
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


# set policies according to the baseline transitions
# TODO: this is a quite bit of a hack and may not lead to the optimal solution    
rmdp = r.rmdp.copy()
for ds,ind in zip(decstatenums, decstateinds):
    a = basedecpolicy[ds] 
    # get the expectation state that corresponds to the baseline policy
    # assumes that all transitions are to the same state
    expind = rmdp.get_toid(ind,a,0,0)
    # set the threshold to be small for this transition
    rmdp.set_threshold(expind,0,0.0)

# solve sampled MDP
v,pol,_,_,_ = rmdp.mpi_jac_l1(100,stype=robust.SolutionType.Robust.value)

robdecvalue = r.decvalue(decstatecount, v)
robdecpolicy = r.decpolicy(decstatecount, pol)
# use a default action when there were no samples for it
robdecpolicy[np.where(robdecpolicy < 0)] = 0

# compute the number of samples for each expectation state
returneval = counter.simulate(50, raam.vec2policy(robdecpolicy, actions, decstatenum),200)

print(robdecvalue)

print('Expected value of the expectation policy', robdecvalue[0])
print('Return of expectation policy', returneval.statistics(counter.discount)['mean_return'])


# ********
## now compute the baseline optimistic policy

baseline_rmdp = r.rmdp.copy()

# make all transitions that do not represent the baseline policy appear bad
for ds,ind in zip(decstatenums, decstateinds):
    for a in range(baseline_rmdp.action_count(ind)):
        # add a bad reward if the action is not a baseline action
        if basedecpolicy[ds] != a:
            baseline_rmdp.set_reward(ind,a,0,0,-1000)

v,pol,_,_,_ = baseline_rmdp.mpi_jac(100,stype=robust.SolutionType.Average.value)

optdecvalue = r.decvalue(decstatecount, v)

print(optdecvalue)

## iterate the value function

# compute baseline action outcome probabilities

# set robust transition probabilities for baseline actions 

# set baseline transitions probabilities (for baseline actions)

# TODO: a simple first step - just set the baseline actions to have 0 error