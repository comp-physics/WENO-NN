# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 22:50:54 2019

@author: ben91
"""
import numpy as np
import math
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, nx, nt, L, T, RK, FS, FVM, IC):
        self.nx = nx#number of gridpoints
        self.nt = nt#number of timesteps
        self.L = L#domain length
        self.T = T#time to simulate for
        self.RK = RK#timestepping method the sim will use
        self.FS = FS#flux splitting method the sim will use
        self.FVM = FVM#finite volume method the sim will use
        self.IC = IC#initial condition of the simulation
        
    def run(self):
        x = np.linspace(0,self.L,self.nx,endpoint = False)
        t = np.linspace(0,self.T,self.nt,endpoint = True)
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        u_all = np.zeros((int(self.nx),int(self.nt)))
        u_all[:,0] = self.IC(x)
        for i in range(0, int(self.nt-1)):
            u_all[:,i+1] = self.RK.stepIt(u_all[:,i], self.FS.flux, self.FVM, dt, dx)
        return u_all        
    
class eulerSimulation:
    def __init__(self, nx, nt, L, T, RK, flux, IC, neq):
        self.nx = nx#number of gridpoints
        self.nt = nt#number of timesteps
        self.L = L#domain length
        self.T = T#time to simulate for
        self.RK = RK#timestepping method the sim will use
        self.flux = flux#finite volume method the sim will use
        self.IC = IC#initial condition of the simulation
        self.neq = neq#number of equations in system of pdes
        
    def runEuler(self):
        x = np.linspace(0,self.L,self.nx,endpoint = False)
        t = np.linspace(0,self.T,self.nt,endpoint = True)
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        u_all = np.zeros((self.nx,self.nt,self.neq))
        u_all[:,0,:] = self.IC(x)
        for i in range(0, self.nt-1):#TODO: not a big deal but why nt-1?
            print(i)
            u_all[:,i+1,:] = self.RK.stepItEuler(u_all[:,i,:], self.flux, dt, dx, self.neq)
        return u_all        
        
class TimeSteppingMethod:#assumes explicit method
    def __init__(self, ss, cff):
        self.ss = ss#substep coefficients
        self.cff = cff#coefficients of the timesteping method, should be lower triangular
        self.nss = ss.size#number of substeps the method has
        
    def stepIt(self, u, flux, FVM, dt, dx):#should work for a vector of inputs u
        n = u.size
        u_all = np.zeros((n,self.nss+1))
        u_all[:,0] = u
        for i in range(0,self.nss):
            for j in range(0,self.nss):
                u_all[:,i+1] += self.cff[i,j]*u_all[:,j]
            u_all[:,i+1] -= self.ss[i]*flux(u_all[:,i], FVM)*dt/dx#minus because HCL is du/dt = -df(u)/dx
        return u_all[:,-1]
    
    def stepItEuler(self, u, flux, dt, dx, neq):#FVM is now inside of the flux equation
        n = np.shape(u)[0]
        u_all = np.zeros((n,self.nss+1,neq))
        u_all[:,0,:] = u
        for i in range(0,self.nss):
            for j in range(0,self.nss):
                u_all[:,i+1,:] += self.cff[i,j]*u_all[:,j,:]
                #print(u_all[:,i+1,:])
            fl = flux(u_all[:,i,:])
            #print(fl)
            u_all[:,i+1,:] -= self.ss[i]*(fl)*dt/dx#minus because HCL is du/dt = -df(u)/dx
            #u_all[:,i+1,:] -= self.ss[i]*(fl)*dt/dx#minus because HCL is du/dt = -df(u)/dx
        return u_all[:,-1]
    
class FluxSplittingMethod:
    def __init__(self, Lp, Lm):
        self.Lp = Lp#positive flux. Note that this defines the PDE
        self.Lm = Lm#negative flux
        
    def flux(self, u, FVM):
        u_int_og = FVM.evalF(u)#normal velocity
        u_int_fl = np.roll(np.flip(FVM.evalF(np.flip(u))),-1)#mirror velocity
        fp = self.Lp(u_int_og)
        fm = self.Lm(u_int_fl)
        
        '''
        plt.figure()
        plt.plot(u_int_og)
        plt.figure()
        plt.plot(u_int_fl)
        '''
        f = (fp+fm-np.roll(fp+fm,1))#flux in minus flux out
        return f
    
class FiniteVolumeMethod:
    def __init__(self, ss, L):
        self.ss = ss#stencil size
        self.L = L#function that gives interpolated value
        
    def partU(self, u):#partition u into the stencil#TODO: get this going for systems of HCL (nx5x3 for euler eqn). pretend it works for now
        n = len(u)
        u_all = np.zeros((n,self.ss))
        
        for i in range(0,self.ss):#assume scheme is upwind or unbiased
            u_all[:,i] = np.roll(u,math.floor(self.ss/2)-i)
        return u_all
        
    def evalF(self, u):
        u_part = self.partU(u)
        u_int = self.L(u_part)
        #tv = np.sum(np.abs(u-np.roll(u, 1, axis = 0)),axis = 0)
        return u_int
    
class FiniteVolumeMethodEuler:
    def __init__(self, ss, L):
        self.ss = ss#stencil size
        self.L = L#function that gives interpolated value
        
    def partU(self, u):#partition u into the stencil#TODO: get this going for systems of HCL (nx5x3 for euler eqn). pretend it works for now
        m,n = np.shape(u)
        u_all = np.zeros((m,n,self.ss))
        
        for i in range(0,self.ss):#assume scheme is upwind or unbiased
            u_all[:,:,i] = np.roll(u,math.floor(self.ss/2)-i,axis = 0)
        return u_all
        
    def evalF(self, f):
        '''
        inputs:
            f: characteristic fluxes (nx3x5)
        outputs:
            u_int: interpolated flux
        '''
        f_int = self.L(f)
        return f_int
        