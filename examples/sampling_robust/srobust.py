"""
Baseline methods for computing solutions that are guaranteed to (likely!) improve
on a provided baseline policy.
"""

import raam
from raam import robust
from raam import crobust
from raam import features
from raam import examples
import numpy as np

debugoutput = True

def solve_expectation(samples, discount, decstatecount, decagg_big, decagg_small, 
                        expagg_big, expagg_small, actagg):
    """ Solves the regular expected MDP """

    # build the sampled MDP    
    r = crobust.SRoMDP(1,discount)
    
    r.from_samples(samples, decagg_big=decagg_big, decagg_small=decagg_small,
                    expagg_big=expagg_big, expagg_small=expagg_small,
                    actagg=actagg)
    
    # solve sampled MDP
    v,pol,_,_,_ = r.rmdp.mpi_jac(5000,stype=robust.SolutionType.Average.value)
    
    decvalue = r.decvalue(decstatecount, v)
    decpolicy = r.decpolicy(decstatecount, pol)
    
    return decvalue, decpolicy
    
def solve_reward_adjusted(samples, discount, decstatecount, decagg_big, decagg_small, 
                        expagg_big, expagg_small, actagg, maxr, err, baselinepol, 
                        baselineval, baselinerv):
    """ 
    Solves the MDP but adjusts the rewards to be a lower bound. Only returns 
    non-baseline when it is better than the baseline return.
    """  

    r = crobust.SRoMDP(1,discount)
    r.from_samples(samples, decagg_big=decagg_big, decagg_small=decagg_small,
                    expagg_big=expagg_big, expagg_small=expagg_small,
                    actagg=actagg)
    
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
                maxr * discount / (1 - discount) * err(scount) )
    
    # solve sampled MDP
    v,pol,_,_,_ = r.rmdp.mpi_jac(1000,stype=robust.SolutionType.Average.value)
    
    expadjdecvalue = r.decvalue(decstatecount, v)
    expadjdecpolicy = r.decpolicy(decstatecount, pol)
    # use a default action when there were no samples for it
    expadjdecpolicy[np.where(expadjdecpolicy < 0)] = 1
    
    # compute the number of samples for each expectation state
    
    # TODO assumes that the first state is the initial one
    if expadjdecvalue[0] >= baselinerv:
        return expadjdecvalue, expadjdecpolicy
    else:
        return baselinepol, baselinepol


def solve_robust(samples, discount, decstatecount, decagg_big, decagg_small, 
                        expagg_big, expagg_small, actagg, maxr, err, baselinepol, 
                        baselineval, baselinerv):

    r = crobust.SRoMDP(1,discount)
    r.from_samples(samples, decagg_big=decagg_big, decagg_small=decagg_small,
                    expagg_big=expagg_big, expagg_small=expagg_small,
                    actagg=actagg)
    
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
    
    # TODO assumes that the first state is the initial one
    if robdecvalue[0] >= baselinerv:
        return robdecvalue, robdecpolicy
    else:
        return baselineval, baselinepol
    
    
def solve_robust_joint(samples, discount, decstatecount, decagg_big, decagg_small, 
                        expagg_big, expagg_small, actagg, maxr, err, baselinepol, 
                        baselineval, baselinerv):


    r = crobust.SRoMDP(1,discount)
    r.from_samples(samples, decagg_big=decagg_big, decagg_small=decagg_small,
                    expagg_big=expagg_big, expagg_small=expagg_small,
                    actagg=actagg)

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
    # TODO: better methods are commented out below
    rmdp = r.rmdp.copy()
    for ds,ind in zip(decstatenums, decstateinds):
        a = baselinepol[ds] 
        # get the expectation state that corresponds to the baseline policy
        # assumes that all transitions are to the same state
        if a < rmdp.action_count(ind):
            # TODO: IMPORTANT, just skips the baseline action when not sampled; this is wrong, make it better
            expind = rmdp.get_toid(ind,a,0,0)
            # set the threshold to be small for this transition
            rmdp.set_threshold(expind,0,0.0)
        else:
            # TODO: a hack - just add the missing baseline action with a good return -> take it
            rmdp.add_transition(ind,a,0,ind,1.0,5.0)
            dist = np.array([1.0])
            dist = dist / dist.sum()
            r.rmdp.set_distribution(ind,a,dist,0.0)

    
    # solve sampled MDP
    v,pol,_,_,_ = rmdp.mpi_jac_l1(100,stype=robust.SolutionType.Robust.value)
    
    robdecvalue = r.decvalue(decstatecount, v)
    robdecpolicy = r.decpolicy(decstatecount, pol)
    # use a default action when there were no samples for it
    robdecpolicy[np.where(robdecpolicy < 0)] = 0
        
    # # ********
    # # computes the baseline optimistic policy
    # 
    # baseline_rmdp = r.rmdp.copy()
    # 
    # # make all transitions that do not represent the baseline policy appear bad
    # for ds,ind in zip(decstatenums, decstateinds):
    #     for a in range(baseline_rmdp.action_count(ind)):
    #         # add a bad reward if the action is not a baseline action
    #         if baselinepol[ds] != a:
    #             baseline_rmdp.set_reward(ind,a,0,0,-1000)
    # 
    # # TODO: this is average to make the hack above work
    # v,pol,_,_,_ = baseline_rmdp.mpi_jac(100,stype=robust.SolutionType.Average.value)
    # 
    # optdecvalue = r.decvalue(decstatecount, v)
    
    
    # probably no need to compare with the baseline policy; if it was better then
    # the actions are simply taken in the optimization itself
    
    return robdecvalue, robdecpolicy
    
    
    # TODO: The REAL solution should proceed as follows
    # # *** Iterative solution method for the value function
    # 
    # # compute baseline action outcome probabilities
    # 
    # # set robust transition probabilities for baseline actions 
    # 
    # # set baseline transitions probabilities (for baseline actions)