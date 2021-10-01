# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:05:51 2019

@author: ben91
"""
from SimulationClasses import *
from TimeSteppingMethods import *
from FiniteVolumeSchemes import *
from FluxSplittingMethods import *
from InitialConditions import *
from Equations import *
from wholeNetworks import *
from LoadDataMethods import *

from keras import *
from keras.models import *
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib import style
from matplotlib import rcParams
import math
style.use('fivethirtyeight')
rcParams.update({'figure.autolayout': True})
'''

# Import modules/packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.close('all') # close all open figures

# Define and set custom LaTeX style
styleNHN = {
        "pgf.rcfonts":False,
        "pgf.texsystem": "pdflatex",   
        "text.usetex": False,     #TODO: might need to change this to false           
        "font.family": "serif"
        }
mpl.rcParams.update(styleNHN)
xx = np.linspace(0,1,100)
yy = xx**2
# Plotting defaults
ALW = 0.75  # AxesLineWidth
FSZ = 12    # Fontsize
LW = 2      # LineWidth
MSZ = 5     # MarkerSize
SMALL_SIZE = 8    # Tiny font size
MEDIUM_SIZE = 10  # Small font size
BIGGER_SIZE = 14  # Large font size
plt.rc('font', size=FSZ)         # controls default text sizes
plt.rc('axes', titlesize=FSZ)    # fontsize of the axes title
plt.rc('axes', labelsize=FSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSZ)   # fontsize of the x-tick labels
plt.rc('ytick', labelsize=FSZ)   # fontsize of the y-tick labels
plt.rc('legend', fontsize=FSZ)   # legend fontsize
plt.rc('figure', titlesize=FSZ)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = ALW    # sets the default axes lindewidth to ``ALW''
plt.rcParams["mathtext.fontset"] = 'cm' # Computer Modern mathtext font (applies when ``usetex=False'')

def discTrackStep(c,x,t,u,P,title, a, b, err):
    '''
    Assume shocks are at middle and end of the x domain at start
    
    Inputs:
        c: shock speed
        x: x coordinates
        y: y coordinates
        u: velocity
        P: periods advected for
        err: plot error if True, otherwise plot solution
    '''
    u = np.transpose(u)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - c*tg
    plt.figure()
    if err:
        ons = np.ones_like(xp)
        eex = np.greater(xp%L,ons)
        er = eex-u
        '''
        plt.contourf(xp,tg,u)
        
        plt.colorbar()
        plt.title(title)
        
        plt.figure()
        plt.contourf(xp,tg,eex)
        '''
        for i in range(-2,int(P)):
            plt.contourf(xp+i*L,tg,abs(er),np.linspace(0,0.7,20))
        plt.xlim(a,b)
        plt.xlabel('x-ct')
        plt.ylabel('t')
        plt.colorbar()
        plt.title(title)
    else:
        for i in range(-2,int(P)+1):
            plt.contourf(xp+i*L,tg,u,np.linspace(-0.2,1.2,57))
        plt.xlim(a,b)
        plt.xlabel('x-ct')
        plt.ylabel('t')
        plt.colorbar()
        plt.title(title)
        
def intError(c,x,t,u,title):
    L = x[-1] - x[0] + x[1] - x[0]
    dx = x[1] - x[0]
    nx = np.size(x)
    xg, tg = np.meshgrid(t,x)
    xp = xg - c*tg
    ons = np.ones_like(xp)
    #eex = np.roll(np.greater(ons,xp%L),-1,axis = 0)
    
    eex1 = xp/dx
    eex1[eex1>=1] = 1
    eex1[eex1<=0] = 0
    
    eex2 = (-xp%L-L/2)/dx
    eex2[eex2>=1] = 1
    eex2[eex2<=0] = 0
    
    eex3 = (-xp%L-L/2)/dx
    eex3[eex3>(nx/2-1)] = -(eex3[eex3>(nx/2-1)]-nx/2)
    eex3[eex3>=1] = 1
    eex3[eex3<=0] = 0
    
    er = eex3-u
    ers = np.power(er,2)
    ers0 = np.expand_dims(ers[0,:],axis = 0)
    ers_aug = np.concatenate((ers,ers0), axis = 0)
    err_int = np.trapz(ers_aug, dx = dx, axis = 0)
    plt.plot(t,np.sqrt(err_int),'.')
    #plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    #plt.ylim([0,0.02])

def totalVariation(t,u,title):#plot total variation over time
    us = np.roll(u, 1, axis = 0)
    tv = np.sum(np.abs(u-us),axis = 0)
    #plt.figure()
    plt.plot(t,tv,'.')
    #plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Total Variation')
    #plt.ylim((1.999,2.01))
    
def totalEnergy(t,u, dx, title):#plot total energy
    u0 = np.expand_dims(u[0,:],axis = 0)
    u_aug = np.concatenate((u,u0), axis = 0)
    energy = 0.5*np.trapz(np.power(u_aug,2), dx = dx, axis = 0)
    plt.figure()
    plt.plot(t,energy)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('1/2*integral(u^2)')
    plt.ylim([0,np.max(energy)*1.1])
    
def mwn(FVM):
    '''
    plot modified wavenumber of a finite volume scheme
    Inputs:
        FVM: finite volume method object to test
    '''
    nx = 100
    nt = 10
    L = 2
    T = 0.00001
    x = np.linspace(0,L,nx,endpoint=False)
    t = np.linspace(0,T,nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    sigma = T/dx
    EQ = adv()
    FS = LaxFriedrichs(EQ, 1)
    RK = SSPRK3()
        
    NK = int((np.size(x)-1)/2)
    mwn = np.zeros(NK,dtype=np.complex_)
    wn = np.zeros(NK)
    A = 1
    for k in range(2,NK):
        IC = cosu(A,k/L)
        testCos = Simulation(nx, nt, L, T, RK, FS, FVM, IC)
        u_cos = testCos.run()
        u_f_cos = u_cos[:,0]
        u_l_cos = u_cos[:,-1]
        
        IC = sinu(A,k/L)
        testSin = Simulation(nx, nt, L, T, RK, FS, FVM, IC)
        u_sin = testSin.run()
        u_f_sin = u_sin[:,0]
        u_l_sin = u_sin[:,-1]
        
        u_h0 =np.fft.fft(u_f_cos+complex(0,1)*u_f_sin)
        u_h = np.fft.fft(u_l_cos+complex(0,1)*u_l_sin)
        v_h0 = u_h0[k]
        v_h = u_h[k]
        mwn[k] = -1/(complex(0,1)*sigma)*np.log(v_h/v_h0)
        wn[k] = 2*k*np.pi/nx
        
    plt.plot(wn,np.real(mwn))
    #plt.hold
    plt.plot(wn,wn)
    plt.xlabel('\phi')
    plt.ylabel('Modified Wavenumber (real part)')
    plt.figure()
    plt.plot(wn,np.imag(mwn))
    plt.xlabel('\phi')
    plt.ylabel('Modified Wavenumber (imaginary part)')
    
    plt.figure()
    plt.semilogy(wn,abs(wn-np.real(mwn)))
    return wn
    
def animateSim(x,t,u,pas):
    '''
    Assume shocks are at middle and end of the x domain at start
    
    Inputs:
        x: x coordinates
        t: t coordinates
        u: velocity
        pas: how long to pause between frames
    '''
    for i in range(0,len(t)):
        plt.plot(x,u[:,i])
        plt.pause(pas)
        plt.clf()        
    plt.plot(x,u[:,-1])
    
def specAnalysis(model, u, RKM,WENONN, NNNN, h, giveModel, makePlots):
    '''
    perform spectral analysis of a finite volume method when operating on a specific waveform
    
    Finds eigenvalues, and then uses this to compute max
    
    Inputs:
        Model: WENO5 neural network that will be analyzed
        u: the data that is the input to the method
        RKM: time stepping method object to analyze for space-time coupling
        wenoName: name of layer in model that gives WENO5 coefficicents
        NNname: name of layer in model that gives NN coefficients
        giveModel: whether or not we are passing layer names or model names
    '''
    if(giveModel):
        pass
    else:
        WENONN = Model(inputs=model.input, outputs = model.get_layer(WENONN).output)
        NNNN = Model(inputs=model.input, outputs = model.get_layer(NNNN).output)
        adm = optimizers.adam(lr=0.0001)
        WENONN.compile(optimizer=adm,loss='mean_squared_error')
        NNNN.compile(optimizer=adm,loss='mean_squared_error')
    N = np.size(u)
    M = 5#just assume stencil size is 5 for now
    
    sortedU = np.zeros((N,M)) + 1j*np.zeros((N,M))
    for i in range(0,M):#assume scheme is upwind or unbiased
        sortedU[:,i] = np.roll(u,math.floor(M/2)-i)
    
    def scale(sortedU, NNNN):
        min_u = np.amin(sortedU,1)
        max_u = np.amax(sortedU,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(sortedU[:,2])
        u_tmp[:] = sortedU[:,2]
        #for i in range(0,5):
        #    sortedU[:,i] = (sortedU[:,i]-min_u)/(max_u-min_u)
        cff = NNNN.predict(sortedU)#compute \Delta u

        cff[const_n,:] = np.array([1/30,-13/60,47/60,9/20,-1/20])
        #print('fl: ', fl)
        return cff
    if(np.sum(np.iscomplex(u))>=1):    
        wec = WENONN.predict(np.real(sortedU)) + WENONN.predict(np.imag(sortedU))*1j
        nnc = scale(np.real(sortedU), NNNN) + scale(np.imag(sortedU), NNNN)*1j
        op_WENO5 = np.zeros((N,N)) + np.zeros((N,N))*1j
        op_NN = np.zeros((N,N)) + np.zeros((N,N))*1j
    else:
        wec = WENONN.predict(np.real(sortedU))
        nnc = scale(np.real(sortedU), NNNN)
        op_WENO5 = np.zeros((N,N))
        op_NN = np.zeros((N,N))
        
    for i in range(0,N):
        for j in range(0,M):
            op_WENO5[i,(i+j-int(M/2))%N] -= wec[i,j] 
            op_WENO5[i,(i+j-int(M/2)-1)%N] += wec[(i-1)%N,j]
            op_NN[i,(i+j-int(M/2))%N] -= nnc[i,j]
            op_NN[i,(i+j-int(M/2)-1)%N] += nnc[(i-1)%N,j]
        #print(i,': ', op_WENO5[i,:])
            
    WEeigs, WEvecs = np.linalg.eig(op_WENO5)
    NNeigs, NNvecs = np.linalg.eig(op_NN)
    
    con_nn = np.linalg.solve(NNvecs, u)
    
    #now do some rungekutta stuff
    x = np.linspace(-3,3,301)
    y = np.linspace(-3,3,301)
    X,Y = np.meshgrid(x,y)
    Z = X + Y*1j
    g = abs(1 + Z + np.power(Z,2)/2 + np.power(Z,3)/6)
    g_we = abs(1 + (h*WEeigs) + np.power(h*WEeigs,2)/2 + np.power(h*WEeigs,3)/6)
    g_nn = abs(1 + (h*NNeigs) + np.power(h*NNeigs,2)/2 + np.power(h*NNeigs,3)/6)
    #do some processing for that plot of the contributions vs the amplification factor
    c_abs = np.abs(con_nn)
    ords = np.argsort(c_abs)
    g_sort = g_nn[ords]
    c_sort = con_nn[ords]    
    c_norm = c_sort/np.linalg.norm(c_sort,1)
    c_abs2 = np.abs(c_norm) 
    #do some processing for the most unstable mode
    ordsG = np.argsort(g_nn)
    unstb = NNvecs[:,ordsG[-1]]
    
    if(makePlots>=1):
        plt.figure()
        plt.plot(np.sort(g_we),'.')
        plt.plot(np.sort(g_nn),'.')
        plt.legend(('WENO5','NN'))
        plt.title('CFL = '+ str(h))
        plt.xlabel('index')
        plt.ylabel('|1+HL+(HL^2)/2+(HL^3)/6|')
        plt.ylim([0,1.2])
        
        plt.figure()
        plt.plot(np.real(WEeigs),np.imag(WEeigs),'.')
        plt.plot(np.real(NNeigs),np.imag(NNeigs),'.')
        plt.title('Eigenvalues')
        plt.legend(('WENO5','NN'))
        
        plt.figure()
        plt.plot(g_nn,abs(con_nn),'.')
        plt.xlabel('Amplification Factor')
        plt.ylabel('Contribution')

        
        print('Max WENO g: ',np.max(g_we))
        print('Max NN g: ',np.max(g_nn))
    if(makePlots>=2):
        plt.figure()
        sml = 1E-2
        plt.contourf(X, Y, g, [1-sml,1+sml])
        
        plt.figure()
        plt.plot(g_sort,c_abs2,'.')
        plt.xlabel('Scaled Amplification Factor')
        plt.ylabel('Contribution')
        
    return g_nn, con_nn, unstb
    #return np.max(g_we), np.max(g_nn)
    #plt.contourf(xp+i*L,tg,abs(er),np.linspace(0,0.025,20))
def specAnalysisData(model, u, RKM,WENONN, NNNN, CFL, giveModel):
    nx, nt = np.shape(u)
    if(giveModel):
        pass
    else:
        WENONN = Model(inputs=model.input, outputs = model.get_layer(WENONN).output)
        NNNN = Model(inputs=model.input, outputs = model.get_layer(NNNN).output)
        adm = optimizers.adam(lr=0.0001)
        WENONN.compile(optimizer=adm,loss='mean_squared_error')
        NNNN.compile(optimizer=adm,loss='mean_squared_error')
    
    maxWe = np.zeros(nt)
    maxNN = np.zeros(nt)

    for i in range(0,nt):
        print(i)
        maxWe[i], maxNN[i] = specAnalysis(model, u[:,i], RKM, WENONN, NNNN, CFL, True, False)
        
    plt.figure()
    plt.plot(maxWe)
    plt.figure()
    plt.plot(maxNN)
        
    return maxWe, maxNN

def eigenvectorProj(model, u, WENONN, NNNN):
    nx = np.shape(u)
    
    WENONN = Model(inputs=model.input, outputs = model.get_layer(WENONN).output)
    NNNN = Model(inputs=model.input, outputs = model.get_layer(NNNN).output)
    adm = optimizers.adam(lr=0.0001)
    WENONN.compile(optimizer=adm,loss='mean_squared_error')
    NNNN.compile(optimizer=adm,loss='mean_squared_error')
    
    
def evalPerf(x,t,P,u,eex):
    '''
    Assume shocks are at middle and end of the x domain at start
    
    Inputs:
        x: x coordinates
        y: y coordinates
        P: periods advected for
        u: velocity
    Outputs:
        tvm: max total variation in solution
        swm: max shock width in solution
    '''
    us = np.roll(u, 1, axis = 0)
    tv = np.sum(np.abs(u-us),axis = 0)
    tvm = np.max(tv)
    
    u = np.transpose(u)
    
    er = np.abs(eex-u)
    wdth = np.sum(np.greater(er,0.005),axis=1)
    swm = np.max(wdth)
    print(tvm)
    print(swm)
    return tvm, swm
'''
def plotDiscWidth(x,t,P,u,u_WE):
'''
    #plot width of discontinuity over time for neural network and WENO5
    
'''
    us = np.roll(u, 1, axis = 0)

    
    u = np.transpose(u)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    ons = np.ones_like(xp)
    eex = np.greater(xp%L,ons)
    er = np.abs(eex-u)
    wdth = np.sum(np.greater(er,0.005),axis=1)
    swm = np.max(wdth)
    print(tvm)
    print(swm)
    return tvm, swm
'''
def plotDiscWidth(x,t,P,u,u_WE):
    '''
    plot width of discontinuity over time for neural network and WENO5
    
    '''
    u = np.transpose(u)
    u_WE = np.transpose(u_WE)

    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    ons = np.ones_like(xp)

    dx = x[1]-x[0]
    '''
    eex = (-xp%L-L/2)/dx
    eex[eex>49] = -(eex[eex>49]-50)
    eex[eex>=1] = 1
    eex[eex<=0] = 0
    '''
    eex = np.greater(xp%L,ons)
    er = np.abs(eex-u)
    er_we = np.abs(eex-u_WE)
    wdth = np.sum(np.greater(er,0.01),axis=1)*dx/2
    wdth_we = np.sum(np.greater(er_we,0.01),axis=1)*dx/2
    
    plt.figure()
    plt.plot(t,wdth)
    plt.plot(t,wdth_we)
    plt.legend(('Neural Network','WENO5'))
    plt.xlabel('t')
    plt.ylabel('Discontinuity Width')
    
    
def convStudy():
    '''
    Test order of accuracy of an FVM
    '''
    nr = 21
    errNN = np.zeros(nr)
    errWE = np.zeros(nr)
    errEN = np.zeros(nr)
    dxs = np.zeros(nr)
    
    for i in range(0,nr):
        print(i)    
        nx = 10*np.power(10,0.1*i)
        L = 2
        x = np.linspace(0,L,int(nx),endpoint=False)
        dx = x[1]-x[0]
            
        FVM1 = NNWENO5dx(dx)
        FVM2 = WENO5()
        FVM3 = ENO3()
        
        u = np.sin(4*np.pi*x) + np.cos(4*np.pi*x)
        du = 4*np.pi*(np.cos(4*np.pi*x)-np.sin(4*np.pi*x))
        resNN = FVM1.evalF(u)
        resWE = FVM2.evalF(u)
        resEN = FVM3.evalF(u)
        
        du_EN = (resNN-np.roll(resEN,1))/dx
        du_NN = (resNN-np.roll(resNN,1))/dx
        du_WE = (resWE-np.roll(resWE,1))/dx
        
        errNN[i] = np.linalg.norm(du_NN-du,ord = 2)/np.sqrt(nx)
        errEN[i] = np.linalg.norm(du_EN-du,ord = 2)/np.sqrt(nx)
        errWE[i] = np.linalg.norm(du_WE-du,ord = 2)/np.sqrt(nx)
        dxs[i] = dx

    nti = 6
    toRegDx = np.ones((nti,2))
    toRegDx[:,1] = np.log10(dxs[-nti:])

    toRegWe = np.log10(errWE[-nti:])
    toRegNN = np.log10(errNN[-nti:])
    toRegEN = np.log10(errEN[-nti:])

    c_we, m_we = np.linalg.lstsq(toRegDx, toRegWe, rcond=None)[0]
    c_nn, m_nn = np.linalg.lstsq(toRegDx, toRegNN, rcond=None)[0]
    c_en, m_en = np.linalg.lstsq(toRegDx, toRegEN, rcond=None)[0]
    
    print('WENO5 slope: ',m_we)
    print('NN slope: ',m_nn)
    print('ENO3 slope: ',m_en)

    plt.loglog(dxs,errNN,'o')
    plt.loglog(dxs,errWE,'o')
    plt.loglog(dxs,errEN,'o')

    plt.loglog(dxs,(10**c_we)*(dxs**m_we))
    plt.loglog(dxs,(10**c_nn)*(dxs**m_nn))
    plt.loglog(dxs,(10**c_en)*(dxs**m_en))

    plt.legend(['WENO-NN','WENO5-JS','ENO3'])
    plt.xlabel('$\Delta x$')
    plt.ylabel('$E$')
    
def plot_visc(x,t,uv,FVM,P,NN,contours):
    nx, nt = np.shape(uv)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    def scheme(u,NN):
        ust = np.zeros_like(u)
        ust = ust + u
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        ep = 1E-6
        #compute fluxes on sub stencils (similar to derivatives I guess)
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute derivatives on sub stencils
        justU = 1/30*ust[:,0]-13/60*ust[:,1]+47/60*ust[:,2]+9/20*ust[:,3]-1/20*ust[:,4]
        dudx = 0*ust[:,0]+1/12*ust[:,1]-5/4*ust[:,2]+5/4*ust[:,3]-1/12*ust[:,4]
        dudx = (dudx - np.roll(dudx,1))
        d2udx2 = -1/4*ust[:,0]+3/2*ust[:,1]-2*ust[:,2]+1/2*ust[:,3]+1/4*ust[:,4]
        d2udx2 = (d2udx2 - np.roll(d2udx2,1))
        d3udx3 = 0*ust[:,0]-1*ust[:,1]+3*ust[:,2]-3*ust[:,3]+1*ust[:,4]
        d3udx3 = (d3udx3 - np.roll(d3udx3,1))

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
        
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        if(NN):
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
            Cons = C[:,0] + C[:,1] + C[:,2] + C[:,3] + C[:,4]
            C_visc = -5/2*C[:,0] - 3/2*C[:,1] - 1/2*C[:,2] + 1/2*C[:,3] + 3/2*C[:,4]
            C_visc2 = 19/6*C[:,0] + 7/6*C[:,1] + 1/6*C[:,2] + 1/6*C[:,3] + 7/6*C[:,4]
            C_visc3 = -65/24*C[:,0] - 5/8*C[:,1] - 1/24*C[:,2] + 1/24*C[:,3] + 5/8*C[:,4]
            C_visc = C_visc.flatten()
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        else:
            Cons = c1[:,0] + c1[:,1] + c1[:,2] + c1[:,3] + c1[:,4]
            C_visc = (-5/2*c1[:,0] - 3/2*c1[:,1] - 1/2*c1[:,2] + 1/2*c1[:,3] + 3/2*c1[:,4])
            C_visc2 = (19/6*c1[:,0] + 7/6*c1[:,1] + 1/6*c1[:,2] + 1/6*c1[:,3] + 7/6*c1[:,4])
            C_visc3 = (-65/24*c1[:,0] - 5/8*c1[:,1] - 1/24*c1[:,2] + 1/24*c1[:,3] + 5/8*c1[:,4])
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        return Cons,-C_visc,-C_visc2,-C_visc3, dudx, d2udx2, d3udx3
    C_ = np.zeros_like(uv)
    C_i = np.zeros_like(uv)
    C_ii = np.zeros_like(uv)
    C_iii = np.zeros_like(uv)
    d_i = np.zeros_like(uv)
    d_ii = np.zeros_like(uv)
    d_iii = np.zeros_like(uv)
    for i in range(0,nt):
         u_part = FVM.partU(uv[:,i])
         C_[:,i],C_i[:,i],C_ii[:,i],C_iii[:,i],d_i[:,i],d_ii[:,i],d_iii[:,i] = scheme(u_part,NN)
         
    dx = x[1]-x[0]
    C_ = np.transpose(C_)
    C_i = np.transpose(C_i)*dx
    C_ii = np.transpose(C_ii)*dx**2
    C_iii = np.transpose(C_iii)*dx**3
    d_i = np.transpose(d_i)/(dx**2)
    d_ii = np.transpose(d_ii)/(dx**3)
    d_iii = np.transpose(d_iii)/(dx**4)

    indFirst = 100#ignore 1st few timesteps for scaling plots due to disconintuity
    if(contours):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        
        
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                first = ax1.contourf(xtp[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(-0.3,0.3,100))
                #first = ax1.contourf(xtp[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
                #first = ax1.contourf(xp+i*L,tg,C_i*np.abs(d_i),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
                np.savetxt('firstXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('firstTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('firstVisc'+str(i)+'.csv',C_i[indsTP,:]*np.abs(d_i[indsTP,:]))
        ax1.set_title('(A)')
        ax1.set_xlim(x[0],x[-1])
        ax1.set_xlabel('$x-ct$')
        ax1.set_ylabel('$t$')
    
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                second = ax2.contourf(xtp[indsTP,:],tg[indsTP,:],C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]),np.linspace(-0.3,0.3,100))
                #second = ax2.contourf(xp+i*L,tg,C_ii*np.abs(d_ii),np.linspace(np.min((C_ii*np.abs(d_ii))[indFirst:,:]),np.max((C_ii*np.abs(d_ii))[indFirst:,:]),100))
                np.savetxt('secondXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('secondTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('secondVisc'+str(i)+'.csv',C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]))
        ax2.set_title('(B)')
        ax2.set_xlim(x[0],x[-1])
        ax2.set_xlabel('$x-ct$')
    
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                third = ax3.contourf(xtp[indsTP,:],tg[indsTP,:],C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]),np.linspace(-0.3,0.3,100))
                np.savetxt('thirdXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('thirdTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('thirdVisc'+str(i)+'.csv',C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]))
                #third = ax3.contourf(xp+i*L,tg,C_iii*np.abs(d_iii),np.linspace(np.min((C_iii*np.abs(d_iii))[indFirst:,:]),np.max((C_iii*np.abs(d_iii))[indFirst:,:]),100))
        ax3.set_title('(C)')
        ax3.set_xlim(x[0],x[-1])
        ax3.set_xlabel('$x-ct$')
        f.subplots_adjust(right=0.8)
        #cbar_ax1 = f.add_axes([.72, 0.15, 0.05, 0.7])
        #cbar_ax2 = f.add_axes([.82, 0.15, 0.05, 0.7])
        #cbar_ax3 = f.add_axes([.92, 0.15, 0.05, 0.7])
        #f.colorbar(first, cax=cbar_ax1)
        #f.colorbar(second, cax=cbar_ax2)
        #f.colorbar(third, cax=cbar_ax3)
        #f.colorbar(first, ax=ax1)
        #f.colorbar(second, ax=ax2)
        #f.colorbar(third, ax=ax3)
        f.tight_layout()
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([.82, 0.15, 0.05, 0.7])
        f.colorbar(third, cax=cbar_ax)
        #f.tight_layout()
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
        ax1.plot(x,C_i[150,:]*np.abs(d_i[150,:]))
        ax1.plot(x,C_i[1500,:]*np.abs(d_i[1500,:]))
        ax1.plot(x,C_i[3750,:]*np.abs(d_i[3750,:]))
        ax1.plot(x,C_i[7500,:]*np.abs(d_i[7500,:]))
        ax1.set_title('(A)')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$t$')
        ax2.plot(x,C_ii[150,:]*np.abs(d_ii[150,:]))
        ax2.plot(x,C_ii[1500,:]*np.abs(d_ii[1500,:]))
        ax2.plot(x,C_ii[3750,:]*np.abs(d_ii[3750,:]))
        ax2.plot(x,C_ii[7500,:]*np.abs(d_ii[7500,:]))
        ax2.set_title('(B)')
        ax2.set_xlabel('$x$')
        ax3.plot(x,C_iii[150,:]*np.abs(d_iii[150,:]))
        ax3.plot(x,C_iii[1500,:]*np.abs(d_iii[1500,:]))
        ax3.plot(x,C_iii[3750,:]*np.abs(d_iii[3750,:]))
        ax3.plot(x,C_iii[7500,:]*np.abs(d_iii[7500,:]))
        ax3.set_title('(C)')
        ax3.set_xlabel('$x$')
        ax3.legend(('$t=2$','$t=20$','$t=50$','$t=100$'))
    
  
def plot_visc_new(x,t,uv,FVM,P,NN,contours):
    nx, nt = np.shape(uv)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    def scheme(u,NN):
        ust = np.zeros_like(u)
        ust = ust + u
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        ep = 1E-6
        #compute fluxes on sub stencils (similar to derivatives I guess)
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute derivatives on sub stencils
        justU = 1/30*ust[:,0]-13/60*ust[:,1]+47/60*ust[:,2]+9/20*ust[:,3]-1/20*ust[:,4]
        dudx = 0*ust[:,0]+1/12*ust[:,1]-5/4*ust[:,2]+5/4*ust[:,3]-1/12*ust[:,4]
        deriv2 = (dudx - np.roll(dudx,1))
        d2udx2 = -1/4*ust[:,0]+3/2*ust[:,1]-2*ust[:,2]+1/2*ust[:,3]+1/4*ust[:,4]
        deriv3 = (d2udx2 - np.roll(d2udx2,1))
        d3udx3 = 0*ust[:,0]-1*ust[:,1]+3*ust[:,2]-3*ust[:,3]+1*ust[:,4]
        deriv4 = (d3udx3 - np.roll(d3udx3,1))

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
        
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        if(NN):
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
            Cons = C[:,0] + C[:,1] + C[:,2] + C[:,3] + C[:,4]
            C_visc = -5/2*C[:,0] - 3/2*C[:,1] - 1/2*C[:,2] + 1/2*C[:,3] + 3/2*C[:,4]
            C_visc2 = 19/6*C[:,0] + 7/6*C[:,1] + 1/6*C[:,2] + 1/6*C[:,3] + 7/6*C[:,4]
            C_visc3 = -65/24*C[:,0] - 5/8*C[:,1] - 1/24*C[:,2] + 1/24*C[:,3] + 5/8*C[:,4]
            C_visc = C_visc.flatten()
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        else:
            Cons = c1[:,0] + c1[:,1] + c1[:,2] + c1[:,3] + c1[:,4]
            C_visc = (-5/2*c1[:,0] - 3/2*c1[:,1] - 1/2*c1[:,2] + 1/2*c1[:,3] + 3/2*c1[:,4])
            C_visc2 = (19/6*c1[:,0] + 7/6*c1[:,1] + 1/6*c1[:,2] + 1/6*c1[:,3] + 7/6*c1[:,4])
            C_visc3 = (-65/24*c1[:,0] - 5/8*c1[:,1] - 1/24*c1[:,2] + 1/24*c1[:,3] + 5/8*c1[:,4])
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        C_visc = (C_visc+np.roll(C_visc,1))/2
        C_visc2 = (C_visc2+np.roll(C_visc2,1))/2
        C_visc3 = (C_visc3+np.roll(C_visc3,1))/2
        return Cons,-C_visc,-C_visc2,C_visc3, deriv2, deriv3, deriv4
    C_ = np.zeros_like(uv)
    C_i = np.zeros_like(uv)
    C_ii = np.zeros_like(uv)
    C_iii = np.zeros_like(uv)
    d_i = np.zeros_like(uv)
    d_ii = np.zeros_like(uv)
    d_iii = np.zeros_like(uv)
    for i in range(0,nt):
         u_part = FVM.partU(uv[:,i])
         C_[:,i],C_i[:,i],C_ii[:,i],C_iii[:,i],d_i[:,i],d_ii[:,i],d_iii[:,i] = scheme(u_part,NN)
         
    dx = x[1]-x[0]
    C_ = np.transpose(C_)
    C_i = np.transpose(C_i)*(dx**2)
    C_ii = np.transpose(C_ii)*(dx**3)
    C_iii = np.transpose(C_iii)*(dx**4)
    d_i = np.transpose(d_i)/(dx**2)
    d_ii = np.transpose(d_ii)/(dx**3)
    d_iii = np.transpose(d_iii)/(dx**4)

    indFirst = 100#ignore 1st few timesteps for scaling plots due to disconintuity
    if(contours):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        num_cont = 10
        cobarlim = 0.003
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                first = ax1.contourf(xtp[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                np.savetxt('firstXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('firstTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('firstVisc'+str(i)+'.csv',C_i[indsTP,:]*np.abs(d_i[indsTP,:]))
                #first = ax1.contourf(xtp[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
                #first = ax1.contourf(xp+i*L,tg,C_i*np.abs(d_i),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
        ax1.set_title('(A)')
        ax1.set_xlim(x[0],x[-1])
        ax1.set_xlabel('$x-ct$')
        ax1.set_ylabel('$t$')
    
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                second = ax2.contourf(xtp[indsTP,:],tg[indsTP,:],C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                np.savetxt('secondXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('secondTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('secondVisc'+str(i)+'.csv',C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]))
                #second = ax2.contourf(xp+i*L,tg,C_ii*np.abs(d_ii),np.linspace(np.min((C_ii*np.abs(d_ii))[indFirst:,:]),np.max((C_ii*np.abs(d_ii))[indFirst:,:]),100))
        ax2.set_title('(B)')
        ax2.set_xlim(x[0],x[-1])
        ax2.set_xlabel('$x-ct$')
    
        for i in range(-2,int(P)+1):
            xtp = xp+i*L
            maxes = np.amax(xtp,axis = 1)
            mines = np.amin(xtp,axis = 1)
            gdi = mines<=2
            gda = maxes>=0
            indsTP = gdi & gda
            if(np.sum(indsTP)>0):
                third = ax3.contourf(xtp[indsTP,:],tg[indsTP,:],C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                np.savetxt('thirdXTP'+str(i)+'.csv',xtp[indsTP,:])
                np.savetxt('thirdTP'+str(i)+'.csv',tg[indsTP,:])
                np.savetxt('thirdVisc'+str(i)+'.csv',C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]))
                #third = ax3.contourf(xp+i*L,tg,C_iii*np.abs(d_iii),np.linspace(np.min((C_iii*np.abs(d_iii))[indFirst:,:]),np.max((C_iii*np.abs(d_iii))[indFirst:,:]),100))
        ax3.set_title('(C)')
        ax3.set_xlim(x[0],x[-1])
        ax3.set_xlabel('$x-ct$')
        #f.subplots_adjust(right=0.8)
        #cbar_ax1 = f.add_axes([.72, 0.15, 0.05, 0.7])
        #cbar_ax2 = f.add_axes([.82, 0.15, 0.05, 0.7])
        #cbar_ax3 = f.add_axes([.92, 0.15, 0.05, 0.7])
        #f.colorbar(first, cax=cbar_ax1)
        #f.colorbar(second, cax=cbar_ax2)
        #f.colorbar(third, cax=cbar_ax3)
        #f.colorbar(first, ax=ax1)
        #f.colorbar(second, ax=ax2)
        #f.colorbar(third, ax=ax3)
        f.tight_layout()
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([.82, 0.15, 0.05, 0.7])
        f.colorbar(third, cax=cbar_ax)
        #f.tight_layout()
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
        ax1.plot(x,C_i[150,:]*np.abs(d_i[150,:]))
        ax1.plot(x,C_i[1500,:]*np.abs(d_i[1500,:]))
        ax1.plot(x,C_i[3750,:]*np.abs(d_i[3750,:]))
        ax1.plot(x,C_i[7500,:]*np.abs(d_i[7500,:]))
        ax1.set_title('(A)')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$t$')
        ax2.plot(x,C_ii[150,:]*np.abs(d_ii[150,:]))
        ax2.plot(x,C_ii[1500,:]*np.abs(d_ii[1500,:]))
        ax2.plot(x,C_ii[3750,:]*np.abs(d_ii[3750,:]))
        ax2.plot(x,C_ii[7500,:]*np.abs(d_ii[7500,:]))
        ax2.set_title('(B)')
        ax2.set_xlabel('$x$')
        ax3.plot(x,C_iii[150,:]*np.abs(d_iii[150,:]))
        ax3.plot(x,C_iii[1500,:]*np.abs(d_iii[1500,:]))
        ax3.plot(x,C_iii[3750,:]*np.abs(d_iii[3750,:]))
        ax3.plot(x,C_iii[7500,:]*np.abs(d_iii[7500,:]))
        ax3.set_title('(C)')
        ax3.set_xlabel('$x$')
        ax3.legend(('$t=2$','$t=20$','$t=50$','$t=100$'))
        
def plot_visc_Even_newer(x,t,uv,FVM,P,NN,contours):
    nx, nt = np.shape(uv)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    def scheme(u,NN):
        ust = np.zeros_like(u)
        ust = ust + u
        min_u = np.amin(u,1)
        max_u = np.amax(u,1)
        const_n = min_u==max_u
        #print('u: ', u)
        u_tmp = np.zeros_like(u[:,2])
        u_tmp[:] = u[:,2]
        for i in range(0,5):
            u[:,i] = (u[:,i]-min_u)/(max_u-min_u)
        
        ep = 1E-6
        #compute fluxes on sub stencils (similar to derivatives I guess)
        f1 = 1/3*u[:,0]-7/6*u[:,1]+11/6*u[:,2]
        f2 = -1/6*u[:,1]+5/6*u[:,2]+1/3*u[:,3]
        f3 = 1/3*u[:,2]+5/6*u[:,3]-1/6*u[:,4]
        #compute derivatives on sub stencils
        justU = 1/30*ust[:,0]-13/60*ust[:,1]+47/60*ust[:,2]+9/20*ust[:,3]-1/20*ust[:,4]
        dudx = 0*ust[:,0]+1/12*ust[:,1]-5/4*ust[:,2]+5/4*ust[:,3]-1/12*ust[:,4]
        deriv2 = (dudx - np.roll(dudx,1))
        d2udx2 = -1/4*ust[:,0]+3/2*ust[:,1]-2*ust[:,2]+1/2*ust[:,3]+1/4*ust[:,4]
        deriv3 = (d2udx2 - np.roll(d2udx2,1))
        d3udx3 = 0*ust[:,0]-1*ust[:,1]+3*ust[:,2]-3*ust[:,3]+1*ust[:,4]
        deriv4 = (d3udx3 - np.roll(d3udx3,1))

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
        
        #fl = np.multiply(fl,(max_u-min_u))+min_u
        if(NN):
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
            Cons = C[:,0] + C[:,1] + C[:,2] + C[:,3] + C[:,4]
            C_visc = -5/2*C[:,0] - 3/2*C[:,1] - 1/2*C[:,2] + 1/2*C[:,3] + 3/2*C[:,4]
            C_visc2 = 19/6*C[:,0] + 7/6*C[:,1] + 1/6*C[:,2] + 1/6*C[:,3] + 7/6*C[:,4]
            C_visc3 = -65/24*C[:,0] - 5/8*C[:,1] - 1/24*C[:,2] + 1/24*C[:,3] + 5/8*C[:,4]
            C_visc = C_visc.flatten()
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        else:
            Cons = c1[:,0] + c1[:,1] + c1[:,2] + c1[:,3] + c1[:,4]
            C_visc = (-5/2*c1[:,0] - 3/2*c1[:,1] - 1/2*c1[:,2] + 1/2*c1[:,3] + 3/2*c1[:,4])
            C_visc2 = (19/6*c1[:,0] + 7/6*c1[:,1] + 1/6*c1[:,2] + 1/6*c1[:,3] + 7/6*c1[:,4])
            C_visc3 = (-65/24*c1[:,0] - 5/8*c1[:,1] - 1/24*c1[:,2] + 1/24*c1[:,3] + 5/8*c1[:,4])
            C_visc[const_n] = 0#if const across stencil, there was no viscosity
            C_visc2[const_n] = 0#if const across stencil, there was no viscosity
            C_visc3[const_n] = 0#if const across stencil, there was no viscosity
        C_visc = (C_visc+np.roll(C_visc,1))/2
        C_visc2 = (C_visc2+np.roll(C_visc2,1))/2
        C_visc3 = (C_visc3+np.roll(C_visc3,1))/2
        return Cons,-C_visc,-C_visc2,C_visc3, deriv2, deriv3, deriv4
    C_ = np.zeros_like(uv)
    C_i = np.zeros_like(uv)
    C_ii = np.zeros_like(uv)
    C_iii = np.zeros_like(uv)
    d_i = np.zeros_like(uv)
    d_ii = np.zeros_like(uv)
    d_iii = np.zeros_like(uv)
    for i in range(0,nt):
         u_part = FVM.partU(uv[:,i])
         C_[:,i],C_i[:,i],C_ii[:,i],C_iii[:,i],d_i[:,i],d_ii[:,i],d_iii[:,i] = scheme(u_part,NN)
         
    dx = x[1]-x[0]
    C_ = np.transpose(C_)
    C_i = np.transpose(C_i)*(dx**2)
    C_ii = np.transpose(C_ii)*(dx**3)
    C_iii = np.transpose(C_iii)*(dx**4)
    d_i = np.transpose(d_i)/(dx**2)
    d_ii = np.transpose(d_ii)/(dx**3)
    d_iii = np.transpose(d_iii)/(dx**4)

    indFirst = 41#ignore 1st few timesteps for scaling plots due to disconintuity
    if(contours):
        indsTP = (np.linspace(0,nt-1,nt)%150==0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        num_cont = 40
        cobarlim = 0.004
        first = ax1.contourf(xg[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                #first = ax1.contourf(xtp[indsTP,:],tg[indsTP,:],C_i[indsTP,:]*np.abs(d_i[indsTP,:]),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
                #first = ax1.contourf(xp+i*L,tg,C_i*np.abs(d_i),np.linspace(np.min((C_i*np.abs(d_i))[indFirst:,:]),np.max((C_i*np.abs(d_i))[indFirst:,:]),100))
        ax1.set_title('(A)')
        ax1.set_xlim(x[0],x[-1])
        ax1.set_xlabel('$x-ct$')
        ax1.set_ylabel('$t$')
    
        second = ax2.contourf(xg[indsTP,:],tg[indsTP,:],C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                #second = ax2.contourf(xp+i*L,tg,C_ii*np.abs(d_ii),np.linspace(np.min((C_ii*np.abs(d_ii))[indFirst:,:]),np.max((C_ii*np.abs(d_ii))[indFirst:,:]),100))
        ax2.set_title('(B)')
        ax2.set_xlim(x[0],x[-1])
        ax2.set_xlabel('$x-ct$')

        third = ax3.contourf(xg[indsTP,:],tg[indsTP,:],C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]),np.linspace(-cobarlim,cobarlim,num_cont))
                #third = ax3.contourf(xp+i*L,tg,C_iii*np.abs(d_iii),np.linspace(np.min((C_iii*np.abs(d_iii))[indFirst:,:]),np.max((C_iii*np.abs(d_iii))[indFirst:,:]),100))
        ax3.set_title('(C)')
        ax3.set_xlim(x[0],x[-1])
        ax3.set_xlabel('$x-ct$')
        
        np.savetxt('XgAll.csv',xg[indsTP,:])
        np.savetxt('TgAll.csv',tg[indsTP,:])
        np.savetxt('VgAll.csv',C_i[indsTP,:]*np.abs(d_i[indsTP,:]))
        np.savetxt('SgAll.csv',C_ii[indsTP,:]*np.abs(d_ii[indsTP,:]))
        np.savetxt('MgAll.csv',C_iii[indsTP,:]*np.abs(d_iii[indsTP,:]))
        #f.subplots_adjust(right=0.8)
        #cbar_ax1 = f.add_axes([.72, 0.15, 0.05, 0.7])
        #cbar_ax2 = f.add_axes([.82, 0.15, 0.05, 0.7])
        #cbar_ax3 = f.add_axes([.92, 0.15, 0.05, 0.7])
        #f.colorbar(first, cax=cbar_ax1)
        #f.colorbar(second, cax=cbar_ax2)
        #f.colorbar(third, cax=cbar_ax3)
        #f.colorbar(first, ax=ax1)
        #f.colorbar(second, ax=ax2)
        #f.colorbar(third, ax=ax3)
        f.tight_layout()
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([.82, 0.15, 0.05, 0.7])
        f.colorbar(third, cax=cbar_ax)
        #f.tight_layout()
    else:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
        ax1.plot(x,C_i[150,:]*np.abs(d_i[150,:]))
        ax1.plot(x,C_i[1500,:]*np.abs(d_i[1500,:]))
        ax1.plot(x,C_i[3750,:]*np.abs(d_i[3750,:]))
        ax1.plot(x,C_i[7500,:]*np.abs(d_i[7500,:]))
        ax1.set_title('(A)')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$t$')
        ax2.plot(x,C_ii[150,:]*np.abs(d_ii[150,:]))
        ax2.plot(x,C_ii[1500,:]*np.abs(d_ii[1500,:]))
        ax2.plot(x,C_ii[3750,:]*np.abs(d_ii[3750,:]))
        ax2.plot(x,C_ii[7500,:]*np.abs(d_ii[7500,:]))
        ax2.set_title('(B)')
        ax2.set_xlabel('$x$')
        ax3.plot(x,C_iii[150,:]*np.abs(d_iii[150,:]))
        ax3.plot(x,C_iii[1500,:]*np.abs(d_iii[1500,:]))
        ax3.plot(x,C_iii[3750,:]*np.abs(d_iii[3750,:]))
        ax3.plot(x,C_iii[7500,:]*np.abs(d_iii[7500,:]))
        ax3.set_title('(C)')
        ax3.set_xlabel('$x$')
        ax3.legend(('$t=2$','$t=20$','$t=50$','$t=100$'))
    
#Below here are official paper visualizations
def threeSolutions(x,t,u_NN,u_WE,u_EX,P):
    '''
    Assume shocks are at middle and end of the x domain at start
    
    Inputs:
        c: shock speed
        x: x coordinates
        y: y coordinates
        u: velocity
        P: periods advected for
        err: plot error if True, otherwise plot solution
    '''
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    
    u_NN = np.transpose(u_NN)
    u_WE = np.transpose(u_WE)
    u_EX = np.transpose(u_EX)
    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg

    for i in range(-2,int(P)+1):
        first = ax1.contourf(xp+i*L,tg,u_EX,np.linspace(-0.2,1.2,57))
    ax1.set_title('Exact')
    ax1.set_xlim(x[0],x[-1])
    ax1.set_xlabel('$x-ct$')
    ax1.set_ylabel('$t$')

    for i in range(-2,int(P)+1):
        second = ax2.contourf(xp+i*L,tg,u_WE,np.linspace(-0.2,1.2,57))
    ax2.set_title('WENO5-JS')
    ax2.set_xlim(x[0],x[-1])
    ax2.set_xlabel('$x-ct$')

    for i in range(-2,int(P)+1):
        third = ax3.contourf(xp+i*L,tg,u_NN,np.linspace(-0.2,1.2,57))
    ax3.set_title('WENO5-NN')
    ax3.set_xlim(x[0],x[-1])
    ax3.set_xlabel('$x-ct$')
    f.tight_layout()
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([.82, 0.15, 0.05, 0.7])
    f.colorbar(third, cax=cbar_ax)

def variousErrors(x,t,u,u_WE):
    #Do L2 error, total variation error, and discontinuity width
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(12, 4))
    #here is the l2 error code:
    L = x[-1] - x[0] + x[1] - x[0]
    dx = x[1] - x[0]
    nx = np.size(x)
    xg, tg = np.meshgrid(t,x)
    xp = xg - tg
    ons = np.ones_like(xp)
    #eex = np.roll(np.greater(ons,xp%L),-1,axis = 0)
    
    eex1 = xp/dx
    eex1[eex1>=1] = 1
    eex1[eex1<=0] = 0
    
    eex2 = (-xp%L-L/2)/dx
    eex2[eex2>=1] = 1
    eex2[eex2<=0] = 0
    
    eex3 = (-xp%L-L/2)/dx
    eex3[eex3>(nx/2-1)] = -(eex3[eex3>(nx/2-1)]-nx/2)
    eex3[eex3>=1] = 1
    eex3[eex3<=0] = 0
    
    er = eex3-u
    ers = np.power(er,2)
    ers0 = np.expand_dims(ers[0,:],axis = 0)
    ers_aug = np.concatenate((ers,ers0), axis = 0)
    err_int = np.trapz(ers_aug, dx = dx, axis = 0)
        
    er_we = eex3-u_WE
    ers_we = np.power(er_we,2)
    ers0_we = np.expand_dims(ers_we[0,:],axis = 0)
    ers_aug_we = np.concatenate((ers_we,ers0_we), axis = 0)
    err_int_we = np.trapz(ers_aug_we, dx = dx, axis = 0)

    
    ax1.plot(t,np.sqrt(err_int),'o')
    ax1.plot(t,np.sqrt(err_int_we),'o')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$E$')
    ax1.set_title('(A)')
    #here is the total variation error code:
    us = np.roll(u, 1, axis = 0)
    tv = np.sum(np.abs(u-us),axis = 0)
    
    uswe = np.roll(u_WE, 1, axis = 0)
    tvwe = np.sum(np.abs(u_WE-uswe),axis = 0)
    #plt.figure()
    ax2.plot(t,tv,'.')
    ax2.plot(t,tvwe,'.')
    #plt.title(title)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$TV$')
    ax2.set_title('(B)')
    #here is the discontinuity width code:
    u = np.transpose(u)
    u_WE = np.transpose(u_WE)

    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    ons = np.ones_like(xp)

    dx = x[1]-x[0]
    '''
    eex = (-xp%L-L/2)/dx
    eex[eex>49] = -(eex[eex>49]-50)
    eex[eex>=1] = 1
    eex[eex<=0] = 0
    '''
    eex = np.greater(xp%L,ons)
    er = np.abs(eex-u)
    er_we = np.abs(eex-u_WE)
    wdth = np.sum(np.greater(er,0.01),axis=1)*dx/2
    wdth_we = np.sum(np.greater(er_we,0.01),axis=1)*dx/2
    
    ax3.plot(t,wdth)
    ax3.plot(t,wdth_we)
    ax3.legend(('WENO-NN','WENO5-JS'))
    ax1.legend(('WENO-NN','WENO5-JS'))
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$w$')
    ax3.set_title('(C)')
    f.tight_layout()

def variousErrors_New(x,t,u,u_WE):
    #Do L2 error, total variation error, and discontinuity width
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(12, 4))
    #here is the l2 error code:
    L = x[-1] - x[0] + x[1] - x[0]
    dx = x[1] - x[0]
    nx = np.size(x)
    xg, tg = np.meshgrid(t,x)
    xp = xg - tg
    ons = np.ones_like(xp)
    #eex = np.roll(np.greater(ons,xp%L),-1,axis = 0)
    
    eex1 = xp/dx
    eex1[eex1>=1] = 1
    eex1[eex1<=0] = 0
    
    eex2 = (-xp%L-L/2)/dx
    eex2[eex2>=1] = 1
    eex2[eex2<=0] = 0
    
    eex3 = (-xp%L-L/2)/dx
    eex3[eex3>(nx/2-1)] = -(eex3[eex3>(nx/2-1)]-nx/2)
    eex3[eex3>=1] = 1
    eex3[eex3<=0] = 0
    
    #np.savetxt('Exact',eex3)
    #np.savetxt('WENO5',u_WE)
    #np.savetxt('NN',u)
    
    '''
    u_int = np.zeros_like(u)
    for i in range(0,np.shape(u)[1]):
        if(i%3==0):
            u_int[:,i] = u[:,i]
        if(i%3==1):
            u_int[:,i] = np.roll((7*np.roll(u[:,i],3)-56*np.roll(u[:,i],2)+280*np.roll(u[:,i],1)+560*np.roll(u[:,i],0)-70*np.roll(u[:,i],-1)+8*np.roll(u[:,i],-2))/729,0)
        if(i%3==2):
            u_int[:,i] = np.roll((8*np.roll(u[:,i],2)-70*np.roll(u[:,i],1)+560*np.roll(u[:,i],0)+280*np.roll(u[:,i],-1)-56*np.roll(u[:,i],-2)+7*np.roll(u[:,i],-3))/729,1)
    '''
    er = eex3-u
    ers = np.power(er,2)
    ers0 = np.expand_dims(ers[0,:],axis = 0)
    ers_aug = np.concatenate((ers,ers0), axis = 0)
    err_int = np.trapz(ers_aug, dx = dx, axis = 0)
    err_avg = (np.roll(err_int,2)+np.roll(err_int,1)+np.roll(err_int,0))/3
    err_avg[0] = err_int[0]
    err_avg[1] = (err_int[0]+err_int[1])/2
    '''
    u_int_WE = np.zeros_like(u)
    for i in range(0,np.shape(u)[1]):
        if(i%3==0):
            u_int_WE[:,i] = u_WE[:,i]
        if(i%3==1):
            u_int_WE[:,i] = (7*np.roll(u_WE[:,i],-3)-56*np.roll(u_WE[:,i],-2)+280*np.roll(u_WE[:,i],-1)+560*np.roll(u_WE[:,i],0)-70*np.roll(u_WE[:,i],-3)+8*np.roll(u_WE[:,i],-3))/729
        if(i%3==2):
            u_int_WE[:,i] = (8*np.roll(u_WE[:,i],-3)-70*np.roll(u_WE[:,i],-2)+560*np.roll(u_WE[:,i],-1)+280*np.roll(u_WE[:,i],0)-56*np.roll(u_WE[:,i],-3)+7*np.roll(u_WE[:,i],-3))/729
    ''' 
    er_we = eex3-u_WE
    ers_we = np.power(er_we,2)
    ers0_we = np.expand_dims(ers_we[0,:],axis = 0)
    ers_aug_we = np.concatenate((ers_we,ers0_we), axis = 0)
    err_int_we = np.trapz(ers_aug_we, dx = dx, axis = 0)
    err_avg_we = (np.roll(err_int_we,2)+np.roll(err_int_we,1)+np.roll(err_int_we,0))/3
    err_avg_we[0] = err_int_we[0]
    err_avg_we[1] = (err_int_we[0]+err_int_we[1])/2
    
    ax1.plot(t,np.sqrt(err_avg),'.')
    ax1.plot(t,np.sqrt(err_avg_we),'.')    
    #ax1.plot(t,np.sqrt(err_int),'.')
    #ax1.plot(t,np.sqrt(err_int_we),'.')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$E$')
    ax1.set_title('(A)')
    ax1.set_ylim((0,0.2))
    #here is the total variation error code:
    us = np.roll(u, 1, axis = 0)
    tv = np.sum(np.abs(u-us),axis = 0)
    
    uswe = np.roll(u_WE, 1, axis = 0)
    tvwe = np.sum(np.abs(u_WE-uswe),axis = 0)
    #plt.figure()
    ax2.plot(t,tv,'.')
    ax2.plot(t,tvwe,'.')
    #plt.title(title)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$TV$')
    ax2.set_title('(B)')
    #here is the discontinuity width code:
    u = np.transpose(u)
    u_WE = np.transpose(u_WE)

    L = x[-1] - x[0] + x[1] - x[0]
    xg, tg = np.meshgrid(x,t)
    xp = xg - tg
    ons = np.ones_like(xp)

    dx = x[1]-x[0]
    '''
    eex = (-xp%L-L/2)/dx
    eex[eex>49] = -(eex[eex>49]-50)
    eex[eex>=1] = 1
    eex[eex<=0] = 0
    '''
    eex = np.greater(xp%L,ons)
    er = np.abs(eex-u)
    er_we = np.abs(eex-u_WE)
    wdth = np.sum(np.greater(er,0.01),axis=1)*dx/2
    wdth_we = np.sum(np.greater(er_we,0.01),axis=1)*dx/2
    
    ax3.plot(t,wdth)
    ax3.plot(t,wdth_we)
    ax3.legend(('WENO-NN','WENO5-JS'))
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$w$')
    ax3.set_title('(C)')
    f.tight_layout()
