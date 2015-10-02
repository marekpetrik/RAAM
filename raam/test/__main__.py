import raam
import raam.test
import sys

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        raam.test.test(sys.argv[1])
    else:
        raam.test.test()
