======================================================
RAAM: Robust and Approximate Markov Decision Processes
======================================================


Overview
--------

This is a library of methods and computational infrastructure for solving 

* Plain Markov decision processes (MDPs)
* Some types of robust MDPs
* Approximately solving MDPs with many states through value function approximation

Installation
------------

Requirements:

- Python 3.3+ (Python 2 is NOT supported)
- Setuptools 7.0
- Numpy 1.8+
- Scipy 0.13 
- Cython 0.21+
- GCC 4.7+ or LLVM with C++11 support

Optional dependencies:

- Matplotlib 1.0+ for plotting support

To install, simply execute:

.. code:: bash

    $ python setup.py install

To install in a development mode, execute:

.. code:: bash

    $ python setup.py develop

The development installation will not copy the files to ``site-packages``---any changes to the Python code will be reflected without the need to reinstall.

To test the installation, run the following python code:
    
.. code:: python

    import raam
    import raam.test
    
    raam.test.test()
    
It is also possible to run the tests from the command line as:
    
.. code:: bash

    python -mraam.test