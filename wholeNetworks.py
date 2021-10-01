# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:26:48 2019

@author: ben91
"""
import numpy as np
from keras import *
from keras.layers import *


def WENO51stOrder(regC):
    pntsuse = 5
    wub6 = np.array([[ 1, 1, 0, 0, 0, 0],
                     [-2,-4, 1, 1, 0, 0],
                     [ 1, 3,-2, 0, 1, 3],
                     [ 0, 0, 1,-1,-2,-4],
                     [ 0, 0, 0, 0, 1, 1]])
    wub6c = np.array([ 0, 0, 0, 0, 0, 0])
    eps = 1e-6
    wub3 = np.array([[13/12,0,0],
                     [1/4  ,0    ,0],
                     [0    ,13/12,0],
                     [0    ,1/4  ,0],
                     [0    ,0    ,13/12],
                     [0    ,0    ,1/4]])
    wub3c = np.array([eps, eps, eps])
    wba1 = np.array([[0.1],
                     [0],
                     [0]])
    wba1c = np.array([0])
    wba2 = np.array([[0],
                     [0.6],
                     [0]])
    wba2c = np.array([0])
    wba3 = np.array([[0],
                     [0],
                     [0.3]])
    wba3c = np.array([0])
    w_to_c = np.array([[1/3, -7/6, 11/6, 0,0],
                     [0, -1/6, 5/6, 1/3, 0],
                     [0, 0, 1/3, 5/6, -1/6]])
    w_to_cc = np.array([0,0,0,0,0])
    wub1 = np.array([[-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5]])
    wub1c = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
        
    # Make weights for the projection
    u_05 = Input(shape = (5, ))#merge all the average inputs as u_(-2),u_(-1),u_0,u_1,u_2,u_3
        
    b_6 = Dense(6,trainable=False,weights=[wub6,wub6c])(u_05)#6 differences for smoothness indicators
    b_6s = Lambda(lambda x: (x ** 2))(b_6)#6 differences squared
    
    b_3 = Dense(3,trainable=False,weights=[wub3,wub3c])(b_6s)#3 smoothness indicators + epsilon
    b_3i = Lambda(lambda x: 1/(x ** 2))(b_3)#invert and square the smoothness indicators
    
    a_1 = Dense(1,trainable=False,weights=[wba1,wba1c])(b_3i)#1st unscaled nonlinear weight
    a_2 = Dense(1,trainable=False,weights=[wba2,wba2c])(b_3i)#2nd unscaled nonlinear weight
    a_3 = Dense(1,trainable=False,weights=[wba3,wba3c])(b_3i)#3rd unscaled nonlinear weight
    
    a_s = Add()([a_1,a_2,a_3])#sum of nonlinear weights
    a_si = Lambda(lambda x: 1/(x))(a_s)#invert the sum of weights
    
    w1 = Multiply()([a_1,a_si])#scale the 1st smoothness indicator
    w2 = Multiply()([a_2,a_si])#scale the 2nd smoothness indicator
    w3 = Multiply()([a_3,a_si])#scale the 3rd smoothness indicator
    
    w = concatenate([w1,w2,w3])
    
    Cs = Dense(5,trainable=False,weights=[w_to_c,w_to_cc])(w)#Final WENO5 coefficients
    
    x1 = Dense(3,activation='relu')(Cs)
    x2 = Dense(3,activation='relu')(x1)
    x3 = Dense(3,activation='relu')(x2)
    #TODO: Pass arguments to this function that define the regularization and neural network nodes/layers and l1/l2 optimization
    dc = Dense(5,activity_regularizer=regularizers.l2(regC))(x3)#end the DNN, the 5 differences are the outputs
    c_tilde = Subtract()([Cs,dc])#use the differences to modify the coefficients
    
    dc2 = Dense(pntsuse,trainable=False,weights=[wub1,wub1c])(c_tilde)#compute how each coefficient must be changed for consistency
    
    c_all = Add()([c_tilde,dc2])
    
    p2 = dot([u_05,c_all], axes = 1, normalize = False)#compute flux from all 5 coefficients
    
    model = Model(inputs=u_05, outputs=[p2])
    return model

def Const51stOrder(regC):
    pntsuse = 5

    H51 = np.array([[0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0]])
    H51c = np.array([1/30, -13/60, 47/60, 9/20, -1/20])
        
    wub1 = np.array([[-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5],
                     [-1/5,-1/5,-1/5,-1/5,-1/5]])
    wub1c = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
    
    
    # Make weights for the projection
    u_05 = Input(shape = (5, ))#merge all the average inputs as u_(-2),u_(-1),u_0,u_1,u_2,u_3
        
    Cs = Dense(5,trainable=False,weights=[H51,H51c])(u_05)#Final WENO5 coefficients
    reggersA = 0.001
    reggersb = 0.001
    
    x1 = Dense(5,activation='relu')(u_05)
    x2 = Dense(5,activation='relu')(x1)
    x3 = Dense(5,activation='relu')(x2)

    #TODO: Pass arguments to this function that define the regularization and neural network nodes/layers and l1/l2 optimization
    #dc = Dense(5,activity_regularizer=regularizers.l2(regC))(x9)#end the DNN, the 5 differences are the outputs
    dc = Dense(5,trainable=False,activity_regularizer=regularizers.l2(regC))(x3)#end the DNN, the 5 differences are the outputs
    c_tilde = Subtract()([Cs,dc])#use the differences to modify the coefficients
    
    dc2 = Dense(pntsuse,trainable=False,weights=[wub1,wub1c])(c_tilde)#compute how each coefficient must be changed for consistency
    
    c_all = Add()([c_tilde,dc2])
    
    p2 = dot([u_05,c_all], axes = 1, normalize = False)#compute flux from all 5 coefficients
    #p2 = dot([u_05,Cs], axes = 1, normalize = False)#compute flux from all 5 coefficients
    
    model = Model(inputs=u_05, outputs=[p2])
    return model