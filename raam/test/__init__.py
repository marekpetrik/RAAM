from .test import *

def test(suite='raam.test',verbosity=2):
    """
    Runs all tests from the module.
    
    Parameters
    ----------
    verbosity : int (optional)
        Test output verbosity
    """
    suite = unittest.TestLoader().loadTestsFromNames([suite])
    #suite = unittest.TestLoader().loadTestsFromTestCase(raam.test.MatrixConstruction)
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

