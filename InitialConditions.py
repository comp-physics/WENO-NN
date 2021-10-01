# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:10:35 2019

@author: ben91
"""
import numpy as np

def step1():
    def fxn(x):
        ons = np.ones(np.shape(x))
        return np.greater(x,ons)
    return fxn

def flat():
    def fxn(x):
        zur = np.zeros(np.shape(x))
        return zur
    return fxn

def smoothedSquare(k,P):
    def fxn(x):
        zur = np.tanh(k*(x-P/4)) - np.tanh(k*(x-3*P/4))
        return zur
    return fxn

def makeIC(L):
    def fxn(x):
        #np.random.seed(N)
        f = 0*x
        for j in range(0,5):
            f = f + np.random.rand(1)*np.sin(2*j*np.pi*(x-np.random.rand(1)))
        f = f + (x>(L/2))*(5-10*np.random.rand(1))
        return f
    return fxn


def randomData():
    def fxn(x):
        np.random.seed(7)
        return np.random.rand(*np.shape(x))
    return fxn

def step2():
    def fxn(x):
        ons = np.ones(np.shape(x))
        res = np.greater(x,ons)/2
        avg = (res+np.roll(res,1))
        return avg
    return fxn

def sinu(A,k):
    def fxn(x):
        return A*np.sin(2*np.pi*k*x)#create the initial condition, step function
    return fxn

def cosu(A,k):
    def fxn(x):
        return A*np.cos(2*np.pi*k*x)#create the initial condition, step function
    return fxn

def gaussian(k,s):
    def fxn(x):
        return np.exp(-k*np.power(x-s,2))
    return fxn

def gaussian2(k,s,A):
    def fxn(x):
        return A*np.exp(-k*np.power(x-s,2))
    return fxn

def idkThatoneThinglol(k,s):
    def fxn(x):
        return 2*k*(x-s)*np.exp(-k*np.power(x-s,2))
    return fxn

def sod():#initial conditions for the sod problem (for Euler equations)
    def fxn(x):
        g = 1.4
        rho0l = 1#initial density
        rho0r = 0.125
        p0l = 1#initial pressure
        p0r = 0.1
        u0l = 0#initial velocity
        u0r = 0
        E0l = p0l/(g-1)+0.5*rho0l*np.power(u0l,2)#initial energy
        E0r = p0r/(g-1)+0.5*rho0r*np.power(u0r,2)
        
        u = np.zeros((np.size(x),3))
        B = 0.5
        u[:,0] = rho0l*np.greater(B,x)+rho0r*np.greater_equal(x,B)
        u[:,1] = rho0l*u0l*np.greater(B,x)+rho0r*u0r*np.greater_equal(x,B)
        u[:,2] = E0l*np.greater(B,x)+E0r*np.greater_equal(x,B)
        return u
    return fxn

def stand():#initial conditions for the sod problem (for Euler equations)
    def fxn(x):
        g = 1.4
        rho0l = 1#initial density
        rho0r = 0.125
        p0l = 1#initial pressure
        p0r = 0.1
        u0l = -1#initial velocity
        u0r = -1
        E0l = p0l/(g-1)+0.5*rho0l*np.power(u0l,2)#initial energy
        E0r = p0r/(g-1)+0.5*rho0r*np.power(u0r,2)
        
        u = np.zeros((np.size(x),3))
        B = 0.5
        u[:,0] = rho0l*np.greater(B,x)+rho0r*np.greater_equal(x,B)
        u[:,1] = rho0l*u0l*np.greater(B,x)+rho0r*u0r*np.greater_equal(x,B)
        u[:,2] = E0l*np.greater(B,x)+E0r*np.greater_equal(x,B)
        return u
    return fxn

def shuOsher():#initial conditions for the sod problem (for Euler equations)
    def fxn(x):
        g = 1.4
        rho0l = 3.857143#initial density
        rho0r = 1 + 0.2*np.sin(5*x)
        p0l = 10.3333#initial pressure
        p0r = 1
        u0l = 2.629369#initial velocity
        u0r = 0
        E0l = p0l/(g-1)+0.5*rho0l*np.power(u0l,2)#initial energy
        E0r = p0r/(g-1)+0.5*rho0r*np.power(u0r,2)
        
        u = np.zeros((np.size(x),3))
        B = 1
        u[:,0] = rho0l*np.greater(B,x)+rho0r*np.greater_equal(x,B)
        u[:,1] = rho0l*u0l*np.greater(B,x)+rho0r*u0r*np.greater_equal(x,B)
        u[:,2] = E0l*np.greater(B,x)+E0r*np.greater_equal(x,B)
        return u
    return fxn