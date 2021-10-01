# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:52:26 2019

@author: ben91
"""

from SimulationClasses import *
import numpy as np

def LaxFriedrichs(F,alpha):
    def Lp(u):
        return 0.5*(F(u) + alpha*u)
    def Lm(u):
        return 0.5*(F(u) - alpha*u)
    FS = FluxSplittingMethod(Lp, Lm)
    return FS

def dontSplit(F):
    def Lp(u):
        return F(u)
    def Lm(u):
        return 0
    FS = FluxSplittingMethod(Lp, Lm)
    return FS