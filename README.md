This repository contains code for simulation experiments in Leung (2019), "A Weak Law for Moments of Pairwise Stable Networks." 

Dependencies: numpy, scipy, pandas, snap (snap.stanford.edu).

Coded for Python 2.7.

Contents:
* monte\_carlo.py: run simulations
* gen\_net\_module.py: functions for simulating networks

First run monte\_carlo.py with variable simulate\_only set to True. This simulates the limiting expected degree and clustering. The values obtained should then be hard-coded into the variables limit\_cc and limit\_deg. Then rerun with simulate\_only set to False. If the variable subcritical is set to True, this corresponds to the subcritical design. Otherwise, this corresponds to the supercritical design.

The output of run.py will be placed in file titled either 'results\_sub.txt' or 'results\_super.txt', depending on the value of the subcritical variable.
 
