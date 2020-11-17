#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='speculator',
      version='v0.2',
      description='SPS emulation',
      author='Justin Alsing',
      url='https://github.com/justinalsing/speculator',
      packages=find_packages(),
      install_requires=[
          "tqdm>=4.41.1",
          "tensorflow>=2.3.0",
          "sklearn"
      ])
