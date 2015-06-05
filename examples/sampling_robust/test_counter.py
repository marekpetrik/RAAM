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
test_steps = 100
test_count_vals = np.arange(2,30,2,dtype=int)
test_counts = len(test_count_vals)
experiment_runs = 4

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

print('Running regular solution ...')

np.random.seed(0)
random.seed(0)

expectation_results = np.empty((test_counts, experiment_runs))

for i,count in enumerate(test_count_vals):
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
        
        expectation_results[i, erun] = exprv

# xvals = test_count_vals*test_steps
# pp.plot(xvals, expectation_results.mean(1),label='Expectation')
# pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
# pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
# pp.xlabel('Number of samples')
# pp.ylabel('Return')
# pp.grid()
# pp.show()

## Reward-adjusted solution

print('Running reward-adjusted solution ...')

imp.reload(srobust)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.07 / np.sqrt(samples)

maxr = np.max(rewards)

np.random.seed(0)
random.seed(0)

rewadj_results = np.empty((test_counts, experiment_runs))

for i,count in enumerate(test_count_vals):
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

        rewadj_results[i, erun] = rewadjrv

# xvals = test_count_vals*test_steps
# pp.plot(xvals, expectation_results.mean(1),label='Expectation')
# pp.plot(xvals, rewadj_results.mean(1),label='Reward Adj')
# 
# pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
# pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
# pp.xlabel('Number of samples')
# pp.ylabel('Return')
# pp.grid()
# pp.show()


## Robust solution

print('Running robust solution ...')

imp.reload(srobust)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.22 / np.sqrt(samples)

robust_results = np.empty((test_counts, experiment_runs))

np.random.seed(0)
random.seed(0)

for i,count in enumerate(test_count_vals):
    for erun in range(experiment_runs):

        samples = counter.simulate(test_steps, counter.random_policy(),count)
        
        robdecvalue, robdecpolicy = srobust.solve_robust(samples, counter.discount, decstatecount, \
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
        robust_results[i, erun] = robrv

# xvals = test_count_vals*test_steps
# pp.plot(xvals, expectation_results.mean(1),label='Expectation')
# pp.plot(xvals, rewadj_results.mean(1),label='Reward Adj')
# pp.plot(xvals, robust_results.mean(1),label='Robust')
# 
# pp.plot(xvals, np.repeat(optrv, len(xvals)),'--')
# pp.plot(xvals, np.repeat(baserv, len(xvals)),'.')
# pp.xlabel('Number of samples')
# pp.ylabel('Return')
# pp.grid()
# pp.show()

## Combined robust and baseline

print('Running robust with baseline solution ...')

imp.reload(srobust)

def err(samples):
    """
    Computes the L1 deviation in the transition probabilities for the given
    number of samples
    """
    return 0.22 / np.sqrt(samples)

combrobust_results = np.empty((test_counts, experiment_runs))

np.random.seed(0)
random.seed(0)

for i,count in enumerate(test_count_vals):
    for erun in range(experiment_runs):

        samples = counter.simulate(test_steps, counter.random_policy(),count)
        
        robdecvalue, robdecpolicy = srobust.solve_robust_joint(samples, counter.discount, decstatecount, \
                                    decagg_big=decstatenum, decagg_small=zero, 
                                    expagg_big=expstatenum, expagg_small=sampleindex, 
                                    actagg=actionagg, maxr=maxr, err=err, baselinepol=basedecpolicy, 
                                    baselineval=basedecvalue, baselinerv=baserv)
        
        # compute the number of samples for each expectation state
        np.random.seed(0)
        random.seed(0)
        crobrv = counter.simulate(50, raam.vec2policy(robdecpolicy, actions, decstatenum),400).statistics(counter.discount)['mean_return']
                
        # TODO: there is a bug with the missing actions - should not be worse than the baseline
        # TODO: should be reoved!!!
        if i == 0 and baserv > crobrv:
            print('A temporary workaround to a bug. Remove!')
            crobrv = max(baserv, crobrv)
                
        #print('Expected value of the robust policy', robdecvalue[0])
        #print('Return of robust policy', robrv)
        combrobust_results[i, erun] = crobrv

## Plots

xvals = test_count_vals*test_steps

def proc(x):
    return 100*(x - baserv) / baserv

pp.plot(xvals, np.repeat(proc(optrv), len(xvals)),'--',color='black')
#pp.plot(xvals, np.repeat(baserv, len(xvals)),'.',color='black')

pp.plot(xvals, proc(expectation_results.mean(1)),label='EXP')
pp.plot(xvals, proc(rewadj_results.mean(1)),label='RWA')
pp.plot(xvals, proc(robust_results.mean(1)),label='ROB')
pp.plot(xvals, proc(combrobust_results.mean(1)),label='RBC')


pp.legend(loc = 0)
pp.xlabel('Number of samples')
pp.ylabel('Improvement over baseline')
pp.grid()
pp.savefig('results.pdf')
pp.savefig('results.png')