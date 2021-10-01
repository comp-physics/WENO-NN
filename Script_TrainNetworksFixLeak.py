# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:25:14 2019

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
from VisualizationFunctions import *

from keras import *
from keras import backend as be
from keras.models import *v
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

Kn = 4
CFL = 2/3
P = 3
nx = 25*Kn
nt = int(25*P*Kn/CFL)+1
L = 2
T = 2*P
x = np.linspace(0,L,nx,endpoint=False)
t = np.linspace(0,T,nt)
dx = x[1]-x[0]
dt = t[1]-t[0]
trainNetworks = True

avgs = (loadInputData("2ndNewAvgs.csv"))
flux = np.transpose(loadOutputData("2ndNewFlux.csv"))
EQ = adv()
FS = LaxFriedrichs(EQ, 1)
#FS = dontSplit(EQ)
IC = step1()
RK = SSPRK3()

#model = WENO51stOrder(0.3)
model = WENO51stOrder(0.14)
adm = optimizers.adam(lr=0.0001)
model.compile(optimizer=adm,loss='mean_squared_error')

L = x[-1] - x[0] + x[1] - x[0]
xg, tg = np.meshgrid(x,t)
xp = xg - tg
ons = np.ones_like(xp)
eex = np.greater(xp%L,ons)
while(trainNetworks):
    t1 = datetime.datetime.now()
    initial_weights = model.get_weights()
    for layr in range(12,20):
        if (layr%2)==0:
            f_in = np.shape(initial_weights[layr])[0]
            f_out = np.shape(initial_weights[layr])[1]
            limt = np.sqrt(6/(f_in+f_out))
            initial_weights[layr] = np.random.rand(f_in,f_out)*2*limt-limt
        else:
            f_in = np.shape(initial_weights[layr])[0]
            initial_weights[layr] = np.zeros(f_in)
    model.set_weights(initial_weights)    
    model.fit(np.transpose(avgs),np.transpose(flux),epochs=10,batch_size=80,verbose = 0)    

    #FVM1 = NNMethod(model)
    FVM1 = NNMethod_noScale(model)
    
    testSim = Simulation(nx, nt, L, T, RK, FS, FVM1, IC)
    uv = testSim.run()
    
    tvm,swm = evalPerf(x,t,P,uv,eex)
    if((tvm<2.01)and(swm<=20)):
        model.save('tvm'+str(tvm)+'swm'+str(swm)+'.h5')
        print('Saved')
    t2 = datetime.datetime.now()
    print(t2-t1)
    #be.clear_session()