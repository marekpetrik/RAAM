#!/usr/bin/python
from setuptools import setup
import subprocess

try:
    from Cython.Build import cythonize
    from Cython.Distutils import Extension
    from Cython.Distutils import build_ext  
    import numpy
except:
    print('ERROR: Setup requires Cython and numpy.')
    raise

# read the version of the package
with open('raam/version.py') as f:
    code = compile(f.read(), "raam/version.py", 'exec')
    exec(code, globals(), locals())

ext_modules = [
    Extension(
        "raam.crobust",
        ["raam/crobust_src/crobust.pyx"],
        extra_compile_args = ['-std=c++11','-fopenmp','-Ofast','-march=native'],
        extra_link_args=['-fopenmp'],
        include_dirs = [numpy.get_include(),'craam/include']),
    Extension(
        "raam.examples.fastrandom",
        ["raam/examples/fastrandom.pyx"],
        extra_compile_args = ['-O3'],
        include_dirs = [numpy.get_include()])]

setup(
    name='raam',
    version=version,
    author='Marek Petrik',
    author_email='marekpetrik@gmail.com',
    packages=['raam','raam.test','raam.examples', 'raam.plotting'],
    scripts=[],
    url='http://watson.ibm.com',
    license='LICENSE',
    description='Algorithms for solving robust and approximate (and plain) Markov decision processes',
    install_requires=[
        "numpy >= 1.8.0",
        "cython >= 0.21.0",
        "scipy >= 0.13.0"
    ],
    cmdclass = {'build_ext' : build_ext},
    ext_modules = cythonize(ext_modules),    
)

