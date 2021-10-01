# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:08:01 2019

@author: ben91
"""

from SimulationClasses import *
import numpy as np

def adv():    
    def R(u):
        mat = np.zeros((u.shape(),1))
        mat[:,:,1] = [1]#TODO: gonna be annoying to get all of the indexing stuff right in this file
        return mat
        
    def lambd(u):
        mat = np.zeros((u.shape(),1))
        mat[:,:,1] = [1]
        return mat

    def Rinv(u):
        mat = np.zeros((u.shape(),1))
        mat[:,:,1] = [1]
        return mat        

    def flux(u):
        return u
    
    return flux

def invBurg():
    def flux(u):
        return 0.5*np.power(u,2)
    return flux

def euler(g):#should probably contain the characteristic decomposition    
    def R(u):
        mat = np.zeros((u.shape(),3))
        mat[:,:,1] = [1,u-a,H-u*a]
        mat[:,:,2] = [1,u,np.power(u,2)]
        mat[:,:,3] = [1,u+a,H+u*a]
        return mat
        
    def lambd(u):
        mat = np.zeros((u.shape(),3))
        mat[:,:,1] = [u-a,0,0]
        mat[:,:,2] = [0, u ,0]
        mat[:,:,3] = [0,0,a+u]
        return mat

    def Rinv(u):
        mat = np.zeros((u.shape(),3))
        mat[:,:,1] = [(g-1)/4*np.power(u/a,2)+u/(2*a),-(g-1)*u/np.power(a,3)-1/(2*a),(g-1)/(2*np.power(a,2))]
        mat[:,:,2] = [1-(g-1)/2*np.power(u/a,2),(g-1)*u/np.power(a,2),-(g-1)/np.power(a,2)]
        mat[:,:,3] = [(g-1)/4*np.power(u/a,2)-u/(2*a),-(g-1)*u/np.power(a,3)+1/(2*a),(g-1)/(2*np.power(a,2))]
        return mat
    
    def cons_to_char(u,f,uta):
        '''
        transform the conservative variables to the characteristic variables
        inputs:
            u: vector of conservative variables (n x 3 matrix)
            f: vector of conservative fluxes (n x 3 matrix)
            uta: vector of conservative variables to average
        outputs:
            w: vector of characteristic variables (n x 3 matrix)
        '''
        g = 1.4#ratio of specific heats for air
        us = 0.5*(u + np.roll(u,1,axis = 0))#average to find the state#TODO: verify this is right
        rho = us[0,:]#density
        vel = us[1,:]/us[0,:]#velocity
        E = us[2,:]#energy
        p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
        c = np.sqrt(g*p/rho)#sound speed
        alpha = rho/(np.sqrt(2)*c)
        beta = 1/(sqrt(2)*rho*c)
        n,m = np.shape(u)
        L = np.zeros((n,m,m))#vector of 3x3 transformation matrices
        L[:,1,1] = 1-(g-1)/2*np.power(u/c,2)
        L[:,1,2] = (g-1)*u/np.power(c,2)#TODO: change syntax to python
        L[:,1,3] = -(g-1)/np.power(c,2)
        L[:,2,1] = beta*(0.5*(g-1)*np.power(u,2)-c*u)
        L[:,2,2] = beta*(c-(g-1)*u)
        L[:,2,3] =  beta*(g-1)
        L[:,3,1] = beta*(0.5*(g-1)*np.power(u,2)+c*u)
        L[:,3,2] = -beta*(c+(g-1)*u)
        L[:,3,3] = beta*(g-1)
        
        w = L*u#TODO: see if need to put this in loop or something
        fw = L*f
        return w,fw
    
    
    def char_to_cons(fc,w):
        '''
        transform the conservative variables to the characteristic variables
        inputs:
            u: vector of conservative variables (n x 3 matrix)
        outputs:
            w: vector of characteristic variables (n x 3 matrix)
        '''
        ws = 0.5*(w + np.roll(w,1,axis = 0))#average to find the state#TODO: verify this is right
        
        #TODO: not really srue whats going on in spencers project_to_cons code
        
        g = 1.4#ratio of specific heats for air
        rho = ws[0,:]#density
        vel = ws[1,:]/ws[0,:]#velocity
        E = ws[2,:]#energy
        p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
        c = np.sqrt(g*p/rho)#sound speed
        alpha = rho/(np.sqrt(2)*c)
        beta = 1/(sqrt(2)*rho*c)
        n,m = np.shape(u)
        R = np.zeros((n,m,m))#vector of 3x3 transformation matrices
        R[:,0,0] = 1
        R[:,0,1] = alpha#TODO: change syntax to python
        R[:,0,2] = alpha
        R[:,1,0] = u
        R[:,1,1] = alpha*(u+c)
        R[:,1,2] = alpha*(u-c)
        R[:,2,0] = 0.5*u^2
        R[:,2,1] = alpha*(0.5*np.power(u,2)+np.power(c,2)/(gamma-1)+c*u)
        R[:,2,2] = alpha*(0.5*np.power(u,2)+np.power(c,2)/(gamma-1)-c*u)
        
        f_cons = R*fc#TODO: see if need to put this in loop or something
        return f_cons
    
    def flux(u):
        u1 = u[:,0]
        u2 = u[:,1]
        u3 = u[:,2]
        f = np.zeros_like(u)        
        f[:,0] = u2
        f[:,1] = 0.5*(3-g)*np.power(u2,2)/u1+(g-1)*u3
        f[:,2] = g*u2*u3/u1-0.5*(g-1)*np.power(u2,3)/np.power(u1,2)
        return f
    
    def spds(u):
        g = 1.4#ratio of specific heats for air
        rho = ws[0,:]#density
        vel = ws[1,:]/ws[0,:]#velocity
        E = ws[2,:]#energy
        p = (g-1)*(E-0.5*rho*np.power(vel,2))#pressure
        c = np.sqrt(g*p/rho)#sound speed
        sps = np.zeros_like(u)
        sps[:,0] = u
        sps[:,1] = u + c
        sps[:,2] = u - c
        return sps
        
    def full_flux(u):
        '''
        inputs:
            u: cell average conservative variables (n x 3 matrix)
        outputs:
            flux: the flux at the boundary
        '''
        fu = flux(u)#compute flux from conserved variables cell averages (the temporary one)
        c = spds(u)#compute wave speeds from cell averages
        alph = max(abs(c))#flux splitting coefficient
        #TODO: this is where we would compute the flux at half points with WENO5. Figure out how to redo code to get this
        w,fw = cons_to_char(u,fu,uta)#project to characteristic variables
        f_pos = 0.5*(fw + alph*w)#positive half of flux
        f_neg = np.flip(0.5*(fw - alph*w))#negative half of flux
        w_half_pos = WENO5(f_pos)#find the characteristic values at cell faces
        w_half_neg = np.roll(np.flip(WENO5(f_neg)),-1)#find the characteristic values at cell faces

        flux_char = w_half_pos + w_half_neg
        flux_cons = char_to_cons(flux_char)
        return flux_cons

    return full_flux