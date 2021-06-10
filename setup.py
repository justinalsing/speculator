#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

install_requires = ["tqdm>=4.41.1", "numpy", "sklearn"]

if platform.machine() == 'arm64':
    try:
        import tensorflow
    except ImportError:
        raise ImportError('install tensorflow manually')
else:
    install_requires.append('tensorflow>=2.3.0')

setup(name='speculator',
      version='v0.2',
      description='SPS emulation',
      author='Justin Alsing',
      url='https://github.com/justinalsing/speculator',
      packages=find_packages(),
      install_requires=install_requires)
