# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:42:06 2019

@author: ben91
"""
from SimulationClasses import *
import numpy as np
'''
Add time stepping methods to this library. Currently only supports explicit
methods
Parameters:
    ss: coefficients in front of new flux at each substep
    cff: coefficients in front of old solution values from previous substeps

See http://www.cfm.brown.edu/people/jansh/page5/page10/page40/assets/Zhong_Talk.pdf
slide 12 for format the code uses. Note that SSPRK2 and SSPRK3 match this
'''
def ExplicitEuler():
    ss = np.array([1])
    cff = np.array([[1]])
    return TimeSteppingMethod(ss, cff)

def SSPRK2():
    ss = np.array([1, 1/2])
    cff = np.array([[1,0],[1/2,1/2]])
    return TimeSteppingMethod(ss, cff)

def SSPRK3():
    ss = np.array([1, 1/4, 2/3])
    cff = np.array([[1,0,0],[3/4,1/4,0],[1/3,0,2/3]])    
    return TimeSteppingMethod(ss, cff)

