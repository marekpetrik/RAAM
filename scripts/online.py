import raam
from raam import crobust


def onlineopt(simulator, sample_count, sim_param, agg_param, init_policy=None, iterations=10):
    """
    Runs online policy optimization, alternatively optimizing and simulating
    the problem.
    
    Parameters
    ----------
    simulator : raam.Simulator
        Simulator used to get samples
    sample_count : int
        Approximate number of samples to gather during simulation. 
    sim_param : dict
        Named parameters for sample simulation (see raam.Simulator.simulate)
    agg_param : dict
        Named parameters for the aggregation function
    init_policy : function (state->action), optional
        Initial policy to get samples. If omitted (None), then the random
        policy from the simulator is used
    iterations : int, optional
        Number of iterations. 
    """
    pass
    