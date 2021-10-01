# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 19:32:05 2019

@author: ben91
"""

from SimulationClasses import *
import numpy as np

def cons_to_char(u,f):
    '''
    transform the conservative variables to the characteristic variables
    inputs:
        u: vector of conservative variables (n x 3 matrix)
        f: vector of conservative fluxes (n x 3 matrix)
    outputs:
        w: vector of characteristic variables (n x 3 matrix)
    '''
    n,m,s = np.shape(u)
    g = 1.4#ratio of specific heats for air
    us = 0.5*(u[:,:,int(np.floor(s/2))] + u[:,:,int(np.ceil(s/2))])#average to find the state
    rho = us[:,0]#density
    vel = us[:,1]/us[:,0]#velocity
    E = us[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    beta = 1/(np.sqrt(2)*rho*c)
    L = np.zeros((n,m,m))#vector of 3x3 transformation matrices
    
    L[:,0,0] = 1-(g-1)/2*np.power(vel/c,2)
    L[:,0,1] = (g-1)*vel/np.power(c,2)#TODO: change syntax to python
    L[:,0,2] = -(g-1)/np.power(c,2)
    L[:,1,0] = beta*(0.5*(g-1)*np.power(vel,2)-c*vel)
    L[:,1,1] = beta*(c-(g-1)*vel)
    L[:,1,2] =  beta*(g-1)
    L[:,2,0] = beta*(0.5*(g-1)*np.power(vel,2)+c*vel)
    L[:,2,1] = -beta*(c+(g-1)*vel)
    L[:,2,2] = beta*(g-1)
    
    w = np.zeros_like(u)
    fw = np.zeros_like(f)
    for i in range(0,n):
        w[i,:,:] = np.matmul(L[i,:,:],u[i,:,:])#TODO: check if we need  any sort of transposes here
        fw[i,:,:] = np.matmul(L[i,:,:],f[i,:,:])
    return w,fw


def char_to_cons(fc,u):
    '''
    transform the conservative variables to the characteristic variables
    inputs:
        fc: characteristic fluxes (n x 3 matrix)
        u: vector of conserved variables (n x 3 matrix)
    outputs:
        f_cons: vector of conserved fluxes (n x 3 matrix)
    '''
    us = 0.5*(u + np.roll(u,-1,axis = 0))#average to find the state#TODO: verify this is right
    
    #TODO: not really srue whats going on in spencers project_to_cons code
    
    g = 1.4#ratio of specific heats for air
    rho = us[:,0]#density
    vel = us[:,1]/us[:,0]#velocity
    E = us[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    alpha = rho/(np.sqrt(2)*c)
    n,m = np.shape(u)
    R = np.zeros((n,m,m))#vector of 3x3 transformation matrices
    R[:,0,0] = 1
    R[:,0,1] = alpha#TODO: change syntax to python
    R[:,0,2] = alpha
    R[:,1,0] = vel
    R[:,1,1] = alpha*(vel+c)
    R[:,1,2] = alpha*(vel-c)
    R[:,2,0] = 0.5*np.power(vel,2)
    R[:,2,1] = alpha*(0.5*np.power(vel,2)+np.power(c,2)/(g-1)+c*vel)
    R[:,2,2] = alpha*(0.5*np.power(vel,2)+np.power(c,2)/(g-1)-c*vel)
    f_cons = np.zeros_like(fc)
    for i in range(0,n):
        f_cons[i,:] = np.matmul(R[i,:,:],fc[i,:])#TODO: check if we need  any sort of transposes here
        
    return f_cons

def flux(u):
    g =  1.4
    u1 = u[:,0]
    u2 = u[:,1]
    u3 = u[:,2]
    f = np.zeros_like(u)        
    f[:,0] = u2
    f[:,1] = 0.5*(3-g)*np.power(u2,2)/u1+(g-1)*u3
    f[:,2] = g*u2*u3/u1-0.5*(g-1)*np.power(u2,3)/np.power(u1,2)
    return f

def spds(ws):
    g = 1.4#ratio of specific heats for air
    rho = ws[:,0]#density
    vel = ws[:,1]/ws[:,0]#velocity
    E = ws[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    sps = np.zeros_like(ws)
    sps[:,0] = vel
    sps[:,1] = vel + c
    sps[:,2] = vel - c
    return sps

def leftBC(u):
    g = 1.4#ratio of specific heats for air
    rho = u[:,0]#density
    vel = u[:,1]/u[:,0]#velocity
    E = u[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    n,m = np.shape(u)
    
    leig1 = vel[0] - c[0]
    
    L1 = leig1*((-3*p[0]+4*p[1]-p[2])/2-rho[0]*c[0]*(-3*vel[0]+4*vel[1]-vel[2])/2)
    
    d1 = L1/(2*np.power(c[0],2))
    d2 = 0.5*L1
    d3 = -L1/(2*rho[0]*c[0])
    
    rhs = np.zeros(3)
    
    rhs[0] = -d1
    rhs[1] = -(vel[0]*d1+rho[0]*d3)
    rhs[2] = -(0.5*np.power(vel[0],2)*d1+d2/(g-1.0)+rho[0]*vel[0]*d3)
    return rhs

def rightBC(u):
    g = 1.4#ratio of specific heats for air
    rho = u[:,0]#density
    vel = u[:,1]/u[:,0]#velocity
    E = u[:,2]#energy
    p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
    c = np.sqrt(g*p/rho)#sound speed
    n,m = np.shape(u)
    
    leig2 = vel[-1]
    leig3 = vel[-1] + c[-1]
    
    L1 = 0
    L2 = leig2*(np.power(c[-1],2)*(3*rho[-1]-4*rho[-2]+rho[-3])/2-(3*p[-1]-4*p[-2]+p[-3])/2)
    L3 = leig3*((3*p[-1]-4*p[-2]+p[-3])/2+rho[-1]*c[-1]*(3*vel[-1]-4*vel[-2]+vel[-3])/2)
    
    d1 = 1/(np.power(c[-1],2))*(L2+0.5*L3)
    d2 = 0.5*L3
    d3 = 1/(2*rho[-1]*c[-1])*L3
    
    rhs = np.zeros(3)
    
    rhs[0] = -d1
    rhs[1] = -(vel[-1]*d1+rho[-1]*d3)
    rhs[2] = -(0.5*np.power(vel[-1],2)*d1+d2/(g-1)+rho[-1]*vel[-1]*d3)
    return rhs

def WENOtheBC(f,G):
    '''
    inputs:
        f: 3x5 fluxes to be weno'd
        B: 3 zeros or ones that decide which stencils to ignore
    outputs:
        fl: interpolated flux
    '''
    ep = 1E-6
    #compute fluxes on sub stencils
    f1 = 1/3*f[:,0]-7/6*f[:,1]+11/6*f[:,2]
    f2 =-1/6*f[:,1]+5/6*f[:,2]+ 1/3*f[:,3]
    f3 = 1/3*f[:,2]+5/6*f[:,3]- 1/6*f[:,4]
    #compute smoothness indicators
    B1 = 13/12*np.power(f[:,0]-2*f[:,1]+f[:,2],2) + 1/4*np.power(f[:,0]-4*f[:,1]+3*f[:,2],2)
    B2 = 13/12*np.power(f[:,1]-2*f[:,2]+f[:,3],2) + 1/4*np.power(f[:,1]-f[:,3],2)
    B3 = 13/12*np.power(f[:,2]-2*f[:,3]+f[:,4],2) + 1/4*np.power(3*f[:,2]-4*f[:,3]+f[:,4],2)
    #assign linear weights
    g1 = G[0]*1/10
    g2 = G[1]*3/5
    g3 = G[2]*3/10
    #compute the unscaled nonlinear weights
    wt1 = g1/np.power(ep+B1,2)
    wt2 = g2/np.power(ep+B2,2)
    wt3 = g3/np.power(ep+B3,2)
    wts = wt1 + wt2 + wt3
    #scale the nonlinear weights
    w1 = wt1/wts
    w2 = wt2/wts
    w3 = wt3/wts
    #compute the flux
    fl = f1*w1+f2*w2+f3*w3
    return fl

def getEulerFlux(FVM):#this contains the flux splitting in it so no nead to do more flux splitting
    def full_flux(u):
        '''
        inputs:
            u: cell average conservative variables
        outputs:
            flux: the flux at the boundary
        '''
        fu = flux(u)#compute flux from conserved variables cell averages (the temporary one)
        c = spds(u)#compute wave speeds from cell averages
        alph = np.max(np.abs(c),axis=0)#flux splitting coefficient
        #TODO: this is where we would compute the flux at half points with WENO5. Figure out how to redo code to get this
            #I think I have it, at least good enough for alpha version of code
        u_part = FVM.partU(u)
        fu_part = FVM.partU(fu)
            
        wp,fwp = cons_to_char(u_part,fu_part)#project to characteristic variables
        wm,fwm = cons_to_char(np.flip(u_part,axis = 2),np.flip(fu_part,axis = 2))#project to characteristic variables
        
        n,m,s = np.shape(wp)
        f_pos = np.zeros_like(fwp)
        f_neg = np.zeros_like(fwm)
        for i in range(0,m):#TODO: look into vectorizing this for a speed boost
            f_pos[:,i,:] = 0.5*(fwp[:,i,:] + alph[i]*wp[:,i,:])#positive half of flux
            f_neg[:,i,:] = 0.5*(fwm[:,i,:] - alph[i]*wm[:,i,:])#negative half of flux
        f_neg = np.roll(f_neg,-1,axis = 0)
        #interior point stuff
        f_half_pos = FVM.evalF(f_pos)#find the characteristic values at cell faces
        f_half_neg = FVM.evalF(f_neg)#find the characteristic values at cell faces
        #get the fluxes near the boundaries
        
        f_half_pos[0,:] = WENOtheBC(f_pos[0,:,:],np.array([0,0,1]))
        f_half_pos[1,:] = WENOtheBC(f_pos[1,:,:],np.array([0,1,1]))
        f_half_neg[0,:] = WENOtheBC(f_neg[0,:,:],np.array([1,1,0]))
        
        f_half_neg[-2,:] = WENOtheBC(f_neg[-2,:,:],np.array([0,0,1]))
        f_half_neg[-3,:] = WENOtheBC(f_neg[-3,:,:],np.array([0,1,1]))
        f_half_pos[-2,:] = WENOtheBC(f_pos[-2,:,:],np.array([1,1,0]))
        
        #f_half_pos[-2,:] = WENOtheBC(f_pos[-2,:,:],np.array([0,0,1]))
        #f_half_pos[-1,:] = WENOtheBC(f_pos[-1,:,:],np.array([0,1,1]))
        #f_half_neg[-1,:] = WENOtheBC(f_neg[-1,:,:],np.array([1,1,0]))
        
        #f_half_neg[1,:] = WENOtheBC(f_neg[1,:,:],np.array([0,1,1]))
        #f_half_pos[-2,:] = WENOtheBC(f_pos[-2,:,:],np.array([1,1,0]))
        #f_half_pos[-2,:] = WENOtheBC(f_pos[-2,:,:],np.array([1,1,0]))
        #f_half_neg[1,:] = WENOtheBC(f_neg[1,:,:],np.array([0,1,1]))
        #f_half_neg[-3,:] = WENOtheBC(f_neg[-3,:,:],np.array([1,1,0]))
        #f_half_neg[-2,:] = WENOtheBC(f_neg[-2,:,:],np.array([1,0,0]))
        
        #add the fluxes together
        flux_char = f_half_pos + f_half_neg
        flux_cons = char_to_cons(flux_char,u)
        
        net_flux = flux_cons-np.roll(flux_cons,1,axis=0)
        
        lbc = leftBC(u)
        rbc = rightBC(u)#TODO: was figuring out where to put these values. Like its probably flux_cons or something but idk for sure
        net_flux[0,:] = -lbc
        net_flux[-1:,:] = -rbc
        return net_flux
    return full_flux