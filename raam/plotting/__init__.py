"""
Convenient plotting functions.

See plot_grid for more details.
"""

import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_grid(aggregation,value=None,policy=None,ticks=(60,60),labels=('x','y')):
    """
    Plots value function and policy for a gridded approximation for a 2D
    state space.
    
    Note that the solution of an MDP may be shorter than the number of states
    in the aggregation when it is constructed from samples  and if states with 
    high indexes are never sampled. You need to append a sufficient number of 
    some values to the solutionin order to match the number of aggregate states.
    
    metplotlib.pyplot.show() needs to be used to actually show the plot.
    
    Parameters
    ----------
    aggregation : raam.features.GridAggregation
        Aggregation class that was used to construct the value function and policy
    value : array, one value for each aggregate state, (optional)
        Maps states to the value function.
    policy_vec : array, one value for each aggregate state, (optional)
        Maps states to the policy value (must be a scalar)
    ticks : tuple, optional
        x,y resolution of the plot
    labels : tuple, optional
        Labels for the plot
    """

    if value is None and policy is None:
        raise ValueError('Must provide either the value of the policy')
    
    X,Y = aggregation.meshgrid(ticks)
    
    if value is not None and policy is not None:
        fig = pp.figure(1,figsize=(15,7))
    else:
        fig = pp.figure(1,figsize=(8,7))
        ax = fig.add_subplot(111, projection='3d')
    
    # value function
    if value is not None and policy is not None:
        ax = fig.add_subplot(121, projection='3d')

    if value is not None:
        Z = aggregation.eval_function(zip(X.flat, Y.flat), value)
        Z = np.array(Z).reshape(X.shape)
        
        ax.plot_surface(X,Y,Z)
        pp.xlabel(labels[0])
        pp.ylabel(labels[1])
        pp.title('Value Function')
    
    # policy
    if value is not None and policy is not None:
        ax = fig.add_subplot(122,projection='3d')

    if policy is not None:
        P = aggregation.eval_function(zip(X.flat, Y.flat), policy)
        P = np.array(P).reshape(X.shape)
    
        ax.plot_surface(X,Y,P)
        pp.xlabel(labels[0])
        pp.ylabel(labels[1])
        pp.title('Policy')