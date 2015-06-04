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
    
    
