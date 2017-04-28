This repository contains code for simulation experiments in Leung (2017), "A Weak Law for Moments of Pairwise-Stable Networks." 

Dependencies: numpy, scipy, pandas, snap (snap.stanford.edu).

Coded for Python 2.7.

Contents:
* run.py: simulate networks and estimate identified set
* estimation\_module.py: functions for estimating identified set
* gen\_net\_module.py: functions for simulating networks

The output of run.py will be placed in a directory 'output' in this folder. The directory will be created by run.py. Within 'output' there will be two folders. The folder 'id\_set' will contain csvs listing parameters in the identified set for each simulation. The folder 'stats' will contain summary statistics of networks for each simulation.

