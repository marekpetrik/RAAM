import raam
from raam import robust
from raam import crobust
from raam import features
from raam import examples
import numpy as np
from counter_modes import CounterModes
import random
import matplotlib.pyplot as pp
import srobust
import imp
imp.reload(srobust)

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
experiment_runs = 5

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

## Optimal and baseline policy

# get samples
np.random.seed(20)
random.seed(20)
samples = counter.simulate(1000, counter.random_policy(),20)

optdecvalue, optdecpolicy = srobust.solve_expectation(samples, counter.discount, decstatecount, \
                            decagg_big=decstatenum, decagg_small=zero, 
                            expagg_big=expstatenum, expagg_small=sampleindex, 
                            actagg=actionagg)
np.random.seed(0)
random.seed(0)
optrv = counter.simulate(50, raam.vec2policy(optdecpolicy, actions, decstatenum), \
                                400).statistics(counter.discount)['mean_return']

print('Optimal value function\n', optdecvalue.reshape(modecount, poscount).T)
print('Optimal policy', optdecpolicy.reshape(modecount, poscount).T)
print('Optimal return', optrv)

# Baseline policy

# get samples
counter.set_acceptable_modes(1)
np.random.seed(20)
random.seed(20)
samples = counter.simulate(1000, counter.random_policy(),30)
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

np.random.seed(0)
random.seed(0)
baserv = counter.simulate(50, baselinepol_fun, 400).statistics(counter.discount)['mean_return']

print('Baseline value function\n', basedecvalue.reshape(modecount, -1).T)
print('Policy', basedecpolicy.reshape(modecount, -1).T)
print('Baseline optimal return', baserv)

# TODO: write a test that takes samples and then sums the number of outcomes across states to make sure that they are equal

## Regular solution 

np.random.seed(0)
random.seed(0)

expectation_results = np.empty((test_counts-1, experiment_runs))

for count in range(1,test_counts):
    for erun in range(experiment_runs):
        samples = counter.simulate(test_steps, counter.random_policy(),count)
        
        expdecvalue, expdecpolicy = srobust.solve_expectation(samples, counter.discount, decstatecount, \
                                    decagg_big=decstatenum, decagg_small=zero, 
                                    expagg_big=expstatenum, expagg_small=sampleindex, 
                                    actagg=actionagg)
        expdecpolicy[np.where(expdecpolicy < 0)] = 1

        # compute the number of samples for each expectation state
        np.random.seed(0)
        random.seed(0)
        exprv = counter.simulate(50, raam.vec2policy(expdecpolicy, actions, decstatenum),400).statistics(counter.discount)['mean_return']
        
        #print('Expected value of the expectation policy', expdecvalue[0])
        #print('Return of expectation policy', exprv)
        
        expectation_results[count-1, erun] = exprv

xvals = np.arange(1,test_counts)*test_steps
pp.plot(xvals, expectation_results.mean(1),label='Expectation')
pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
pp.xlabel('Number of samples')
pp.ylabel('Return')
pp.grid()
pp.show()

## Reward-adjusted solution

imp.reload(srobust)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.1 / np.sqrt(samples)

maxr = np.max(rewards)

np.random.seed(0)
random.seed(0)

rewadj_results = np.empty((test_counts-1, experiment_runs))

for count in range(1,test_counts):
    for erun in range(experiment_runs):

        samples = counter.simulate(test_steps, counter.random_policy(),count)
        
        expadjdecvalue, expadjdecpolicy = srobust.solve_reward_adjusted(samples, counter.discount, decstatecount, \
                                    decagg_big=decstatenum, decagg_small=zero, 
                                    expagg_big=expstatenum, expagg_small=sampleindex, 
                                    actagg=actionagg, maxr=maxr, err=err, baselinepol=basedecpolicy, 
                                    baselineval=basedecvalue, baselinerv=baserv)
                                    
                                    
        np.random.seed(0)
        random.seed(0)
        rewadjrv = counter.simulate(50, raam.vec2policy(expadjdecpolicy, actions, decstatenum),400).statistics(counter.discount)['mean_return']
        #print('Expected value of the expectation policy', expadjdecvalue[0])
        #print('Return of expectation policy', rewadjrv)

        rewadj_results[count-1, erun] = rewadjrv

xvals = np.arange(1,test_counts)*test_steps
pp.plot(xvals, expectation_results.mean(1),label='Expectation')
pp.plot(xvals, rewadj_results.mean(1),label='Reward Adj')

pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
pp.xlabel('Number of samples')
pp.ylabel('Return')
pp.grid()
pp.show()


## Robust solution

imp.reload(srobust)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.25 / np.sqrt(samples)

robust_results = np.empty((test_counts-1, experiment_runs))

np.random.seed(0)
random.seed(0)

for count in range(1,test_counts):
    for erun in range(experiment_runs):

        samples = counter.simulate(test_steps, counter.random_policy(),count)
        
        robdecvalue, robdecpolicy = srobust.solve_reward_adjusted(samples, counter.discount, decstatecount, \
                                    decagg_big=decstatenum, decagg_small=zero, 
                                    expagg_big=expstatenum, expagg_small=sampleindex, 
                                    actagg=actionagg, maxr=maxr, err=err, baselinepol=basedecpolicy, 
                                    baselineval=basedecvalue, baselinerv=baserv)
        
        # compute the number of samples for each expectation state
        np.random.seed(0)
        random.seed(0)
        robrv = counter.simulate(50, raam.vec2policy(robdecpolicy, actions, decstatenum),400).statistics(counter.discount)['mean_return']
                
        #print('Expected value of the robust policy', robdecvalue[0])
        #print('Return of robust policy', robrv)
        robust_results[count-1, erun] = robrv

xvals = np.arange(1,test_counts)*test_steps
pp.plot(xvals, expectation_results.mean(1),label='Expectation')
pp.plot(xvals, rewadj_results.mean(1),label='Reward Adj')
pp.plot(xvals, robust_results.mean(1),label='Robust')

pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
pp.xlabel('Number of samples')
pp.ylabel('Return')
pp.grid()
pp.show()

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