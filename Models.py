# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:36:54 2016

@author: Alex
"""

from __future__ import print_function, division
import numpy as np
import scipy.stats as ss


class Model(object):
    def __init__(self, state_dim, control_dim, meas_dim, evo_dim):
            
        self.State_dim = int(state_dim)
        self.Control_dim = int(control_dim)
        self.Meas_dim = int(meas_dim)
        self.Evolution_dim = int(evo_dim)
        
        self.State_Matrix = np.identity(state_dim)
        
        self.Control_Matrix = np.identity(max(state_dim, control_dim))[:state_dim, :control_dim]
        
        self.Measurement_Matrix = np.identity(max(state_dim, meas_dim))[:meas_dim, :state_dim]
        
        self.Evolution_Matrix = np.identity(max(evo_dim, state_dim))[:state_dim, :evo_dim]
        
    def predict(self, state_vector, controle_vector):
        return self.state_function(state_vector) + self.control_function(controle_vector)
    
    def measure(self, state_vector):
        return self.measurement_function(state_vector)
        
    def evolute(self, state_vector, random_vector):
        return state_vector + self.evolution_function(random_vector)

    def state_function(self, state_vector):
        return np.dot(self.State_Matrix, state_vector)

    def control_function(self, state_vector):
        return np.dot(self.Control_Matrix, state_vector)

    def measurement_function(self, state_vector):
        return np.dot(self.Measurement_Matrix, state_vector)

    def evolution_function(self, random_vector):
        return np.dot(self.Evolution_Matrix, random_vector)


class RandomWalk(Model):
    def __init__(self, state_dim):
        super(RandomWalk, self).__init__(state_dim, 0, state_dim, state_dim)


class Balistic(Model):
    def __init__(self, dim, damping=None, mass=1., timeconst=1.):
        super(Balistic, self).__init__(dim * 3, 0, dim, dim)

        if damping is None:
            damping = np.zeros(dim)
        else:
            damping = np.array(damping)
            if damping.shape != (long(dim),):
                damping = np.ones(dim) * np.mean(damping)
        self.Damping = damping

        self.Mass = mass

        n = dim
        matrix = np.zeros((3 * n, 3 * n))
        matrix[::3, ::3] = np.diag(np.ones(n))
        matrix[1::3, 1::3] = np.diag(np.exp(-1 * self.Damping))
        matrix[2::3, 2::3] = np.diag(np.ones(n))
        matrix[::3, 1::3] = np.diag(np.ones(n) * timeconst)
        matrix[1::3, 2::3] = np.diag(np.ones(n) * timeconst / self.Mass)

        meas = np.zeros((n, n * 3))
        meas[:, ::3] = np.diag(np.ones(n))

        evo = np.zeros((n * 3, n))
        evo[::3, :] = np.diag(np.ones(n))

        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo


class VariableSpeed(Model):
    def __init__(self, dim, damping=None, timeconst=1.):
        super(VariableSpeed, self).__init__(dim*2, 0, dim, dim)

        if damping is None:
            damping = np.zeros(dim)
        else:
            damping = np.array(damping)
            if damping.shape != (long(dim),):
                damping = np.ones(dim) * np.mean(damping)
        self.Damping = damping

        n = dim
        matrix = np.zeros((2*n, 2*n))
        matrix[::2, ::2] = np.diag(np.ones(n))
        matrix[1::2, 1::2] = np.diag(np.exp(-1*self.Damping))
        matrix[::2, 1::2] = np.diag(np.ones(n)*timeconst)

        meas = np.zeros((n, n*2))
        meas[:, ::2] = np.diag(np.ones(n))

        evo = np.zeros((n*2, n))
        evo[1::2, :] = np.diag(np.ones(n))

        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo


class AR(Model):
    def __init__(self, dim, order=1, coefficients=None):
        order += 1
        super(AR, self).__init__(dim*order, 0, dim, dim)

        if coefficients is None:
            coefficients = np.ones(order)
        else:
            coefficients = np.array(coefficients)
            if coefficients.shape != (long(order),):
                coefficients = np.ones(order) * np.mean(coefficients)
        self.Coefficients = np.array(coefficients)
        
        matrix = np.zeros((order*dim, order*dim))
        for i, c in enumerate(self.Coefficients):
            matrix[::order, i::order] += np.diag(np.ones(dim)*c)
        
        for i, c in enumerate(self.Coefficients[1:]):
            matrix[i+1::order, i::order] += np.diag(np.ones(dim))
        
        meas = np.zeros((dim, order*dim))
        meas[:, ::order] = np.diag(np.ones(dim))

        evo = np.zeros((order*dim, dim))
        evo[::order, :] = np.diag(np.ones(dim))

        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo

 
class MA(Model):
    def __init__(self, dim, order=1, coefficients=None):
        order += 1
        super(MA, self).__init__(dim*(order+1), 0, dim, dim)
        if coefficients is None:
            coefficients = np.ones(order)
        else:
            coefficients = np.array(coefficients)
            if coefficients.shape != (long(order),):
                coefficients = np.ones(order) * np.mean(coefficients)
        self.Coefficients = np.array(coefficients)
        
        matrix = np.zeros(((order+1)*dim, (order+1)*dim))
        
        matrix[::order+1, ::order+1] += np.diag(np.ones(dim)*self.Coefficients[0])
        matrix[::order+1, order::order+1] += np.diag(np.ones(dim)*self.Coefficients[-1]*-1*self.Coefficients[0])
        matrix[1::order+1, ::order+1] += np.diag(np.ones(dim))
        
        if self.Coefficients.shape[0] == 1:
            matrix[::order+1, 1::order+1] += np.diag(np.ones(dim))
        for i, c in enumerate(self.Coefficients[1:]):
            matrix[::order+1, i+1::order+1] += np.diag(np.ones(dim)*c*(1-self.Coefficients[0]))
            matrix[i+2::order+1, i+1::order+1] += np.diag(np.ones(dim))
        for i, c in enumerate(self.Coefficients):
            matrix[1::order+1, i+1::order+1] += -1*np.diag(np.ones(dim)*c)
            
        meas = np.zeros((dim, (order+1)*dim))
        meas[:, ::order+1] = np.diag(np.ones(dim))

        evo = np.zeros(((order+1)*dim, dim))
        evo[::order+1, :] = np.diag(np.ones(dim))
        
        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo

