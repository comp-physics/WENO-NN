# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:49:22 2019

@author: ben91
"""

from SimulationClasses import *
import numpy as np

def Hook5():
    def scheme(u):
        fl = 1/30*u[:,0]-13/60*u[:,1]+47/60*u[:,2]+9/20*u[:,3]-1/20*u[:,4]
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def WENO5():
    def scheme(u):
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(u[:,0]-2*u[:,1]+u[:,2],2) + 1/4*np.power(u[:,0]-4*u[:,1]+3*u[:,2],2)
        B2 = 13/12*np.power(u[:,1]-2*u[:,2]+u[:,3],2) + 1/4*np.power(u[:,1]-u[:,3],2)
        B3 = 13/12*np.power(u[:,2]-2*u[:,3]+u[:,4],2) + 1/4*np.power(3*u[:,2]-4*u[:,3]+u[:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
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
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def ENO3():
    def scheme(u):
        ep = 1E-6
        
        f1 =-1/2*u[:,0]+3/2*u[:,1]
        f2 = 1/2*u[:,1]+1/2*u[:,2]
        
        B1 = (u[:,0]-u[:,1])**2
        B2 = (u[:,1]-u[:,2])**2
        
        #WENO3 stuff
        g1 = 1/3
        g2 = 2/3
        
        wt1 = g1/(ep+B1**2)
        wt2 = g2/(ep+B2**2)
        wts = wt1 + wt2
        
        w1 = wt1/wts
        w2 = wt2/wts
        
        fl = f1*w1+f2*w2
        #return fl

        fl = np.zeros_like(f1)
        for i in range(0,len(fl)):
            if(B1[i]>B2[i]):
                fl[i] = f2[i]
            else:
                fl[i] = f1[i]
        #fl[B1>B2] = f2[B1>B2]
        #fl[B1<=B2] = f1[B1<=B2]

        return fl
    FVM = FiniteVolumeMethod(3, scheme)
    return FVM

def upwind():
    def scheme(u):
        return u[:,2]
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def NNWENO5():
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(u[:,0]-2*u[:,1]+u[:,2],2) + 1/4*np.power(u[:,0]-4*u[:,1]+3*u[:,2],2)
        B2 = 13/12*np.power(u[:,1]-2*u[:,2]+u[:,3],2) + 1/4*np.power(u[:,1]-u[:,3],2)
        B3 = 13/12*np.power(u[:,2]-2*u[:,3]+u[:,4],2) + 1/4*np.power(3*u[:,2]-4*u[:,3]+u[:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
        #compute the unscaled nonlinear weights
        wt1 = g1/np.power(ep+B1,2)
        wt2 = g2/np.power(ep+B2,2)
        wt3 = g3/np.power(ep+B3,2)
        wts = wt1 + wt2 + wt3
        #scale the nonlinear weights
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts
        #compute the coefficients
        c1 = np.transpose(np.array([1/3*w1,-7/6*w1-1/6*w2,11/6*w1+5/6*w2+1/3*w3,1/3*w2+5/6*w3,-1/6*w3]))
        A1 = np.array([[-0.94130915, -0.32270527, -0.06769955],
        [-0.37087336, -0.05059665,  0.55401474],
        [ 0.40815187, -0.5602299 , -0.01871526],
        [ 0.56200236, -0.5348897 , -0.04091108],
        [-0.6982639 , -0.49512517,  0.52821904]])
        b1 = np.array([-0.04064859,  0.        ,  0.        ])
        c2 = np.maximum(np.matmul(c1,A1)+b1,0)
    
        A2 = np.array([[ 0.07149544,  0.9637294 ,  0.41981453],
        [ 0.75602794, -0.0222342 , -0.95690656],
        [ 0.07406807, -0.41880417, -0.4687035 ]])
        b2 = np.array([-0.0836111 , -0.00330033, -0.01930024])
        c3 = np.maximum(np.matmul(c2,A2)+b2,0)
        
        A3 = np.array([[ 0.8568574 , -0.5809458 ,  0.04762125],
        [-0.26066098, -0.23142155, -0.6449008 ],
        [ 0.7623346 ,  0.81388015, -0.03217626]])
        b3 = np.array([-0.0133561 , -0.05374921,  0.        ])
        c4 = np.maximum(np.matmul(c3,A3)+b3,0)
 
        A4 = np.array([[-0.2891752 , -0.53783405, -0.17556567, -0.7775279 ,  0.69957024],
        [-0.12895434,  0.13607207,  0.12294354,  0.29842544, -0.00198237],
        [ 0.5356503 ,  0.09317833,  0.5135357 , -0.32794708,  0.13765627]])
        b4 = np.array([ 0.00881096,  0.01138764,  0.00464343,  0.0070305 , -0.01644066])
        dc = np.matmul(c4,A4)+b4
        ct = c1 - dc
        
        Ac = np.array([[-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2]])
        bc = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        dc2 = np.matmul(ct,Ac)+bc
        C = ct + dc2
        fl = u[:,0]*C[:,0] + u[:,1]*C[:,1] + u[:,2]*C[:,2] + u[:,3]*C[:,3] + u[:,4]*C[:,4]

        
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value        
        
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def NNWENO5dx(dx):
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(u[:,0]-2*u[:,1]+u[:,2],2) + 1/4*np.power(u[:,0]-4*u[:,1]+3*u[:,2],2)
        B2 = 13/12*np.power(u[:,1]-2*u[:,2]+u[:,3],2) + 1/4*np.power(u[:,1]-u[:,3],2)
        B3 = 13/12*np.power(u[:,2]-2*u[:,3]+u[:,4],2) + 1/4*np.power(3*u[:,2]-4*u[:,3]+u[:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
        #compute the unscaled nonlinear weights
        wt1 = g1/np.power(ep+B1,2)
        wt2 = g2/np.power(ep+B2,2)
        wt3 = g3/np.power(ep+B3,2)
        wts = wt1 + wt2 + wt3
        #scale the nonlinear weights
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts
        #compute the coefficients
        c1 = np.transpose(np.array([1/3*w1,-7/6*w1-1/6*w2,11/6*w1+5/6*w2+1/3*w3,1/3*w2+5/6*w3,-1/6*w3]))
        A1 = np.array([[-0.94130915, -0.32270527, -0.06769955],
        [-0.37087336, -0.05059665,  0.55401474],
        [ 0.40815187, -0.5602299 , -0.01871526],
        [ 0.56200236, -0.5348897 , -0.04091108],
        [-0.6982639 , -0.49512517,  0.52821904]])
        b1 = np.array([-0.04064859,  0.        ,  0.        ])
        c2 = np.maximum(np.matmul(c1,A1)+b1,0)
    
        A2 = np.array([[ 0.07149544,  0.9637294 ,  0.41981453],
        [ 0.75602794, -0.0222342 , -0.95690656],
        [ 0.07406807, -0.41880417, -0.4687035 ]])
        b2 = np.array([-0.0836111 , -0.00330033, -0.01930024])
        c3 = np.maximum(np.matmul(c2,A2)+b2,0)
        
        A3 = np.array([[ 0.8568574 , -0.5809458 ,  0.04762125],
        [-0.26066098, -0.23142155, -0.6449008 ],
        [ 0.7623346 ,  0.81388015, -0.03217626]])
        b3 = np.array([-0.0133561 , -0.05374921,  0.        ])
        c4 = np.maximum(np.matmul(c3,A3)+b3,0)
 
        A4 = np.array([[-0.2891752 , -0.53783405, -0.17556567, -0.7775279 ,  0.69957024],
        [-0.12895434,  0.13607207,  0.12294354,  0.29842544, -0.00198237],
        [ 0.5356503 ,  0.09317833,  0.5135357 , -0.32794708,  0.13765627]])
        b4 = np.array([ 0.00881096,  0.01138764,  0.00464343,  0.0070305 , -0.01644066])
        dc = np.matmul(c4,A4)+b4
        ct = c1 - dc
        
        Ac = np.array([[-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2]])
        bc = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        dc2 = np.matmul(ct,Ac)+bc
        C = ct + dc2
        fl = u[:,0]*C[:,0] + u[:,1]*C[:,1] + u[:,2]*C[:,2] + u[:,3]*C[:,3] + u[:,4]*C[:,4]

        
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value        
        
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM


def NNWENO5_NoScale():
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        u_str = np.zeros_like(u)
        for i in range(0,5):
            u_str[:,i] = u[:,i]
            #u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        
        ep = 1E-6
        #compute fluxes on sub stencils
        #f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        #f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        #f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(u[:,0]-2*u[:,1]+u[:,2],2) + 1/4*np.power(u[:,0]-4*u[:,1]+3*u[:,2],2)
        B2 = 13/12*np.power(u[:,1]-2*u[:,2]+u[:,3],2) + 1/4*np.power(u[:,1]-u[:,3],2)
        B3 = 13/12*np.power(u[:,2]-2*u[:,3]+u[:,4],2) + 1/4*np.power(3*u[:,2]-4*u[:,3]+u[:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
        #compute the unscaled nonlinear weights
        wt1 = g1/np.power(ep+B1,2)
        wt2 = g2/np.power(ep+B2,2)
        wt3 = g3/np.power(ep+B3,2)
        wts = wt1 + wt2 + wt3
        #scale the nonlinear weights
        w1 = wt1/wts
        w2 = wt2/wts
        w3 = wt3/wts
        #compute the coefficients
        c1 = np.transpose(np.array([1/3*w1,-7/6*w1-1/6*w2,11/6*w1+5/6*w2+1/3*w3,1/3*w2+5/6*w3,-1/6*w3]))
        A1 = np.array([[-0.94130915, -0.32270527, -0.06769955],
        [-0.37087336, -0.05059665,  0.55401474],
        [ 0.40815187, -0.5602299 , -0.01871526],
        [ 0.56200236, -0.5348897 , -0.04091108],
        [-0.6982639 , -0.49512517,  0.52821904]])
        b1 = np.array([-0.04064859,  0.        ,  0.        ])
        c2 = np.maximum(np.matmul(c1,A1)+b1,0)
    
        A2 = np.array([[ 0.07149544,  0.9637294 ,  0.41981453],
        [ 0.75602794, -0.0222342 , -0.95690656],
        [ 0.07406807, -0.41880417, -0.4687035 ]])
        b2 = np.array([-0.0836111 , -0.00330033, -0.01930024])
        c3 = np.maximum(np.matmul(c2,A2)+b2,0)
        
        A3 = np.array([[ 0.8568574 , -0.5809458 ,  0.04762125],
        [-0.26066098, -0.23142155, -0.6449008 ],
        [ 0.7623346 ,  0.81388015, -0.03217626]])
        b3 = np.array([-0.0133561 , -0.05374921,  0.        ])
        c4 = np.maximum(np.matmul(c3,A3)+b3,0)
 
        A4 = np.array([[-0.2891752 , -0.53783405, -0.17556567, -0.7775279 ,  0.69957024],
        [-0.12895434,  0.13607207,  0.12294354,  0.29842544, -0.00198237],
        [ 0.5356503 ,  0.09317833,  0.5135357 , -0.32794708,  0.13765627]])
        b4 = np.array([ 0.00881096,  0.01138764,  0.00464343,  0.0070305 , -0.01644066])
        dc = np.matmul(c4,A4)+b4
        ct = c1 - dc
        
        Ac = np.array([[-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2]])
        bc = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        dc2 = np.matmul(ct,Ac)+bc
        C = ct + dc2
        #fl = u[:,0]*C[:,0] + u[:,1]*C[:,1] + u[:,2]*C[:,2] + u[:,3]*C[:,3] + u[:,4]*C[:,4]
        fl = u_str[:,0]*C[:,0] + u_str[:,1]*C[:,1] + u_str[:,2]*C[:,2] + u_str[:,3]*C[:,3] + u_str[:,4]*C[:,4]

        
        fl = fl.flatten()
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value        
        
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM


def NNMethod(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def NNMethod_noScale(model):    
    def scheme(u):
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        #for i in range(0,5):
            #u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        fl = model.predict(u)#compute \Delta u
        fl = fl.flatten()
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        fl[const_n] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethod(5, scheme)
    return FVM

def WENO5euler():
    def scheme(f):
        '''
        inputs:
            f: variables to WENO5 (nx3x5)
        outputs:
            fl: WENO5 flux
        '''
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*f[:,:,0]-7/6*f[:,:,1]+11/6*f[:,:,2]
        f2 =-1/6*f[:,:,1]+5/6*f[:,:,2]+ 1/3*f[:,:,3]
        f3 = 1/3*f[:,:,2]+5/6*f[:,:,3]- 1/6*f[:,:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(f[:,:,0]-2*f[:,:,1]+f[:,:,2],2) + 1/4*np.power(f[:,:,0]-4*f[:,:,1]+3*f[:,:,2],2)
        B2 = 13/12*np.power(f[:,:,1]-2*f[:,:,2]+f[:,:,3],2) + 1/4*np.power(f[:,:,1]-f[:,:,3],2)
        B3 = 13/12*np.power(f[:,:,2]-2*f[:,:,3]+f[:,:,4],2) + 1/4*np.power(3*f[:,:,2]-4*f[:,:,3]+f[:,:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
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
    FVM = FiniteVolumeMethodEuler(5, scheme)
    return FVM

def NNWENO5euler():
    def scheme(f):
        '''
        inputs:
            f: variables to WENO5 (nx3x5)
        outputs:
            fl: WENO5 flux
        '''
        ep = 1E-6
        #compute fluxes on sub stencils
        f1 = 1/3*f[:,:,0]-7/6*f[:,:,1]+11/6*f[:,:,2]
        f2 =-1/6*f[:,:,1]+5/6*f[:,:,2]+ 1/3*f[:,:,3]
        f3 = 1/3*f[:,:,2]+5/6*f[:,:,3]- 1/6*f[:,:,4]
        #compute smoothness indicators
        B1 = 13/12*np.power(f[:,:,0]-2*f[:,:,1]+f[:,:,2],2) + 1/4*np.power(f[:,:,0]-4*f[:,:,1]+3*f[:,:,2],2)
        B2 = 13/12*np.power(f[:,:,1]-2*f[:,:,2]+f[:,:,3],2) + 1/4*np.power(f[:,:,1]-f[:,:,3],2)
        B3 = 13/12*np.power(f[:,:,2]-2*f[:,:,3]+f[:,:,4],2) + 1/4*np.power(3*f[:,:,2]-4*f[:,:,3]+f[:,:,4],2)
        #assign linear weights
        g1 = 1/10
        g2 = 3/5
        g3 = 3/10
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
    FVM = FiniteVolumeMethodEuler(5, scheme)
    return FVM

def NNEuler(model):    
    def scheme(u_all):
        n,m,s = np.shape(u_all)
        fl = np.zeros((n,m))
        u = np.zeros((n,s))
        for i in range(0,m):
            u[:,:] = u_all[:,i,:]
            min_u = np.amin(u,1)
            max_u = np.amax(u,1)
            const_n = min_u==max_u
            #print('u: ', u)
            u_tmp = np.zeros_like(u[:,2])
            u_tmp[:] = u[:,2]
            for j in range(0,5):
                u[:,j] = (u[:,j]-min_u)/(max_u-min_u)
            fl[:,i] = model.predict(u).flatten()#compute \Delta u
            #fl = fl.flatten()
            fl[:,i] = np.multiply(fl[:,i],(max_u-min_u))+min_u
            fl[const_n,i] = u_tmp[const_n]#if const across stencil, set to that value
        #print('fl: ', fl)
        return fl
    FVM = FiniteVolumeMethodEuler(5, scheme)
    return FVM