#!/usr/bin/python3
from setuptools import setup
import subprocess

# read the version of the package
with open('raam/version.py') as f:
    code = compile(f.read(), "raam/version.py", 'exec')
    exec(code, globals(), locals())

setup(
    name='raam',
    version=version,
    author='Marek Petrik',
    author_email='marekpetrik@gmail.com',
    packages=['raam','raam.test','raam.examples', 'raam.examples.inventory', 'raam.plotting'],
    scripts=[],
    url='',
    license='LICENSE',
    description='Algorithms for solving robust and approximate (and plain) Markov decision processes',
    install_requires=[
        "numpy >= 1.8.0",
        "scipy >= 0.13.0"
    ]
)

