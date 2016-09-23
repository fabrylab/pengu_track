# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:36:54 2016

@author: Alex
"""

from __future__ import print_function, division
import numpy as np
import scipy.stats as ss


class Model(object):
    def __init__(self, state_dim, controle_dim, meas_dim,
                 evo_dim=None,
                 state_function = None,
                 controle_function = None,
                 meas_function = None,
                 evo_function = None):
        if evo_dim is None:
            evo_dim = state_dim
            
        self.State_dim = int(state_dim)
        self.Controle_dim = int(controle_dim)
        self.Meas_dim = int(meas_dim)
        self.Evolution_dim = int(evo_dim)
        
        if state_function is None:
            state_function = lambda x: np.dot(np.identity(state_dim), x)
        self.State_Function = state_function
        
        if controle_function is None:
            controle_function = lambda x: np.dot(np.identity(max(state_dim, controle_dim))[:state_dim,:controle_dim], x)
        self.Controle_Function = controle_function
        
        if meas_function is None:
            meas_function = lambda x: np.dot(np.identity(max(state_dim, meas_dim))[:meas_dim,:state_dim], x)
        self.Measurement_Function = meas_function
        
        if evo_function is None:
            evo_function = lambda x: np.dot(np.identity(max(state_dim, evo_dim))[:evo_dim,:state_dim], x)
        self.Evolution_Function = evo_function
        
    def predict(self, state_vector, controle_vector):
        return self.State_Function(state_vector) + self.Controle_Function(controle_vector)
    
    def measure(self, state_vector):
        return self.Measurement_Function(state_vector)
        
    def evolute(self, state_vector, random_vector):
        return state_vector + self.Evolution_Function(random_vector)

class RandomWalk(Model):
    def __init__(self, state_dim):
        super(RandomWalk, self).__init__(state_dim, 0, state_dim)
        
class Balistic(Model):
    def __init__(self, dim, damping=None, mass=1., timeconst=1.):
        
        if damping is None:
            damping = np.zeros(state_dim)
        else:
            damping = np.array(damping)
            if damping.shape != (long(dim),):
                damping = np.ones(dim) * np.mean(damping)
        self.Damping = damping
        
        self.Mass = mass
        
        n = dim
        Matrix = np.zeros((3*n,3*n))
        Matrix[::3, ::3] = np.diag(np.ones(n))
        Matrix[1::3, 1::3] = np.diag(np.exp(-1*self.Damping))
        Matrix[2::3, 2::3] = np.diag(np.ones(n))
        Matrix[::3, 1::3] = np.diag(np.ones(n)*timeconst)
        Matrix[1::3, 2::3] = np.diag(np.ones(n)*timeconst/ self.Mass)
        
        Meas = np.zeros((n,n*3))
        Meas[:,::3] =  np.diag(np.ones(n))
        
        state_function = lambda x: np.dot(Matrix,x)
        meas_function = lambda x: np.dot(Meas,x)
        
        super(Balistic, self).__init__(dim*3, 0, dim,
                                       state_function=state_function,
                                       meas_function=meas_function)

class AR(Model):
    def __init__(self, dim, order=1, coefficients=None):
        if coefficients is None:
            coefficients = np.ones(order)
        else:
            coefficients = np.array(coefficients)
            if coefficients.shape != (long(order),):
                coefficients = np.ones(order) * np.mean(coefficients)
        self.Coefficients = np.array(coefficients)
        
        Matrix = np.zeros((order*dim,order*dim))
        for i,c in enumerate(self.Coefficients):
            Matrix[::order,i::order] += np.diag(np.ones(dim)*c)
        
        for i,c in enumerate(self.Coefficients[1:]):
            Matrix[i+1::order,i::order] += np.diag(np.ones(dim))
        
        Meas = np.zeros((dim,order*dim))
        Meas[:,::order] = np.diag(np.ones(dim))
        
        self.State_Matrix = Matrix
        state_function = lambda x: np.dot(Matrix,x)
        self.Measurement_Matrix = Matrix
        meas_function = lambda x: np.dot(Meas,x)
        
        super(AR,self).__init__(dim*order, 0, dim,
                                state_function=state_function,
                                meas_function=meas_function)
 
class MA(Model):
    def __init__(self, dim, order=1, coefficients=None):
        if coefficients is None:
            coefficients = np.ones(order)
        else:
            coefficients = np.array(coefficients)
            if coefficients.shape != (long(order),):
                coefficients = np.ones(order) * np.mean(coefficients)
        self.Coefficients = np.array(coefficients)
        
        Matrix = np.zeros(((order+1)*dim,(order+1)*dim))
        
        Matrix[::order+1,::order+1] += np.diag(np.ones(dim)*self.Coefficients[0])
        Matrix[::order+1,order::order+1] += np.diag(np.ones(dim)*self.Coefficients[-1]*-1*self.Coefficients[0])
        Matrix[1::order+1,::order+1] += np.diag(np.ones(dim))
        
        if self.Coefficients.shape[0]==1:
            Matrix[::order+1,1::order+1] = np.diag(np.zeros(dim))
        for i,c in enumerate(self.Coefficients[1:]):
            Matrix[::order+1,i+1::order+1] = np.diag(np.ones(dim)*c*(1-self.Coefficients[0]))
            Matrix[i+2::order,i+1::order+1] = np.diag(np.ones(dim))
        for i,c in enumerate(self.Coefficients):
            Matrix[1::order+1,i+1::order+1] = -1*np.diag(np.ones(dim)*c)
            
        Meas = np.zeros((dim,(order+1)*dim))
        Meas[:,::order+1] = np.diag(np.ones(dim))
        
        self.State_Matrix = Matrix
        state_function = lambda x: np.dot(Matrix,x)
        self.Measurement_Matrix = Matrix
        meas_function = lambda x: np.dot(Meas,x)
        
        super(MA,self).__init__(dim*(order+1), 0, dim,
                                state_function=state_function,
                                meas_function=meas_function)
        