"""
Global configuration for the problem settings
"""

import numpy as np
from scipy import stats

horizon = 300
runs = 40

DefaultConfiguration = {
    "price_buy" : [1.2,2.1,3.3],
    "price_sell" : [1,2,3],
    "price_probabilities" : np.array([[0.8, 0.1, 0.1],[0.1, 0.8, 0.1],[0.1, 0.1, 0.8]]),
    "initial_capacity" : 1,
    "initial_inventory" : 0.5,
    "degradation" : {"fun":"polynomial","charge":[0.0,0,0.01],
                            "discharge":[0.01,-0.02,0.01]  },
    "capacity_cost" : 1,
    "change_capacity" : False   # assume that the capacity does not change
    }

def construct_martingale(prices, variance):
    """
    Constructs a definitions with a martingale definitoin of transition probabilities. 
    The change in price is modeled as a normal distribution with zero mean and 
    the specified variance. 
    
    The capacity of the battery does in fact change
    
    Parameters
    ----------
    prices : array
        **Sell** prices that correspond to states in the Martingale price state 
        process. **Buy** prices are 10% higher. 
    variance : float
        Variance of the normal distribution
        
    Returns
    -------
    out : dict
        Configuration that corresponds to the martingale
    """
    states = len(prices)
    
    # defines over how many states the probability is spread over
    spread = min(5,states-1)
    
    if type(prices) is not np.ndarray:
        prices = np.array(prices)
    
    # relative transition probabilities
    p = stats.norm(0,variance).pdf(np.arange(-spread,spread+1))
    p = p / p.sum()
    # add extra 0s to both ends of p
    p = np.concatenate((np.zeros(states-spread-1), p, np.zeros(states-spread-1)))
    
    P = [p[states-i-1:2*states-i-1] for i in range(states)]
    P = np.array(P)
    
    P = np.diag(1/P.sum(1)).dot(P)
    
    configuration = {
        "price_buy" : 1.1 * prices,
        "price_sell" : prices,
        "price_probabilities" : P,
        "initial_capacity" : 1,
        "initial_inventory" : 0.5,
        "degradation" : {"fun":"polynomial","charge":[0.0,0,0.01],
                                "discharge":[0.01,0.02,0.01]  },
        "capacity_cost" : 1,
        "change_capacity" : True   # assume that the capacity does not change
        }
        
    return configuration

def construct_massdata(degrade):
    """
    Returns a problem definition on what is described in the experimental 
    section of the paper

    This uses a simple uniform quantization of energy prices
    
    Paramaters
    ----------
    degrade : bool
        Whether the battery degrades
    """    
    
    prices = np.array([25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0])
    
    P = np.array([[  8.15584416e-01,   1.76623377e-01,   5.19480519e-03,
          2.59740260e-03,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  4.70114171e-02,   8.72397582e-01,   7.25319006e-02,
          7.38750839e-03,   0.00000000e+00,   6.71591672e-04,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.19904077e-03,   1.31894484e-01,   7.79376499e-01,
          6.95443645e-02,   1.43884892e-02,   3.59712230e-03,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  0.00000000e+00,   4.24528302e-02,   2.83018868e-01,
          5.14150943e-01,   1.22641509e-01,   2.35849057e-02,
          9.43396226e-03,   0.00000000e+00,   0.00000000e+00,
          4.71698113e-03],
       [  0.00000000e+00,   2.15053763e-02,   9.67741935e-02,
          2.68817204e-01,   4.30107527e-01,   1.29032258e-01,
          4.30107527e-02,   1.07526882e-02,   0.00000000e+00,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   3.22580645e-02,
          2.58064516e-01,   3.54838710e-01,   1.93548387e-01,
          9.67741935e-02,   6.45161290e-02,   0.00000000e+00,
          0.00000000e+00],
       [  0.00000000e+00,   7.14285714e-02,   1.42857143e-01,
          0.00000000e+00,   7.14285714e-02,   2.14285714e-01,
          2.85714286e-01,   1.42857143e-01,   7.14285714e-02,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   1.42857143e-01,
          0.00000000e+00,   2.85714286e-01,   0.00000000e+00,
          0.00000000e+00,   2.85714286e-01,   2.85714286e-01,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   2.50000000e-01,   2.50000000e-01,
          2.50000000e-01,   0.00000000e+00,   2.50000000e-01,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]])

    if degrade:
        degradation = {"fun":"polynomial","charge" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00142857142857143],
                       "discharge" : [0.0, 0.00500000000000000, -0.00750000000000000, 0.00500000000000000, -0.00125000000000000]  }
    else:
        degradation = {"fun":"polynomial","charge" : [0.0],
                                        "discharge" : [0.0]  }

    configuration = {
        "price_buy" : 1.05 * prices,
        "price_sell" : 0.95 * prices,
        "price_probabilities" : P,
        "initial_capacity" : 1,
        "initial_inventory" : 0.5,
        "degradation" : degradation,
        "capacity_cost" : 20000,
        "change_capacity" : True   
        }
        
    return configuration
