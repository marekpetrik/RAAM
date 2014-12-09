import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int choice(double[:] probs):
    """ 
    Chooses one of the options using the probabilities.

    Parameters
    ----------
    cumprobs : arraylike
        Density, should sum to 1, but it is not checked
    """
    
    cdef double r, s = 0
    r = np.random.rand()
    cdef int i, l = len(probs)
    for i in range(l):
        s += probs[i]
        if r <= s:
            return i
    return len(probs) - 1