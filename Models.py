# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:36:54 2016

@author: Alex
"""

from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
import scipy.optimize as opt


class Model(object):
    def __init__(self, *args, **kwargs):
        self.State_dim = int(kwargs.pop('state_dim', 1))
        self.Control_dim = int(kwargs.pop('control_dim', 1))
        self.Meas_dim = int(kwargs.pop('meas_dim', 1))
        self.Evolution_dim = int(kwargs.pop('evo_dim', 1))

        self.Initial_Args = args
        self.Initial_KWArgs = kwargs
        
        self.State_Matrix = np.identity(self.State_dim)
        
        self.Control_Matrix = np.identity(max(self.State_dim, self.Control_dim))[:self.State_dim, : self.Control_dim]
        
        self.Measurement_Matrix = np.identity(max(self.State_dim, self.Meas_dim))[: self.Meas_dim, :self.State_dim]
        
        self.Evolution_Matrix = np.identity(max(self.Evolution_dim, self.State_dim))[:self.State_dim, :self.Evolution_dim]
        
    def predict(self, state_vector, controle_vector):
        return self.state_function(state_vector) + self.control_function(controle_vector)

    def measure(self, state_vector):
        return self.measurement_function(state_vector)
        
    def evolute(self, state_vector, random_vector):
        return state_vector + self.evolution_function(random_vector)

    def state_function(self, state_vector):
        return np.dot(self.State_Matrix, state_vector)

    def control_function(self, controle_vector):
        return np.dot(self.Control_Matrix, controle_vector)

    def measurement_function(self, state_vector):
        return np.dot(self.Measurement_Matrix, state_vector)

    def evolution_function(self, random_vector):
        return np.dot(self.Evolution_Matrix, random_vector)

    def infer_state(self, meas_vector):
        return np.dot(self.pseudo_inverse(self.Measurement_Matrix), meas_vector)

    def pseudo_inverse(self, matrix):
        matrix = np.asarray(matrix)
        return np.dot(matrix.T, np.linalg.inv(np.dot(matrix, matrix.T)))



class RandomWalk(Model):
    def __init__(self, *args, **kwargs):
        dim = kwargs.get('dim', default=1)

        kwargs.update({'state_dim': dim,
                       'control_dim': 0,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(RandomWalk, self).__init__(*args, **kwargs)


class Balistic(Model):
    def __init__(self, *args, **kwargs):
        dim = kwargs.get('dim', default=1)
        kwargs.update({'state_dim': dim*3,
                       'control_dim': 0,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(Balistic, self).__init__(*args, **kwargs)

        self.Damping = np.zeros(dim)
        self.Damping[:] = kwargs.get('damping', default=0)
        self.Mass = float(kwargs.get('mass', default=1))
        self.Timeconst = float(kwargs.get('timeconst', default=1))

        self.Opt_Params = ['damping', 'mass', 'timeconst']

        n = dim
        matrix = np.zeros((3 * n, 3 * n))
        matrix[::3, ::3] = np.diag(np.ones(n))
        matrix[1::3, 1::3] = np.diag(np.exp(-1 * self.Damping))
        matrix[2::3, 2::3] = np.diag(np.ones(n))
        matrix[::3, 1::3] = np.diag(np.ones(n) * self.Timeconst)
        matrix[1::3, 2::3] = np.diag(np.ones(n) * self.Timeconst / self.Mass)

        meas = np.zeros((n, n * 3))
        meas[:, ::3] = np.diag(np.ones(n))

        evo = np.zeros((n * 3, n))
        evo[::3, :] = np.diag(np.ones(n))

        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo


class VariableSpeed(Model):
    def __init__(self, *args, **kwargs):

        dim = kwargs.get('dim', default=1)
        kwargs.update({'state_dim': dim*2,
                       'control_dim': 0,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(VariableSpeed, self).__init__(*args, **kwargs)

        self.Damping = np.zeros(dim)
        self.Damping[:] = kwargs.get('damping', default=0)
        self.Timeconst = float(kwargs.get('timeconst', default=1))

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
    def __init__(self, *args, **kwargs):

        order = kwargs.get('order', default=1)
        order += 1
        dim = kwargs.get('dim', default=1)
        kwargs.update({'state_dim': dim * order,
                       'control_dim': 0,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(AR, self).__init__(*args, **kwargs)

        self.Coefficients = np.zeros(order)
        self.Coefficients[:] = kwargs.get('coefficients', default=0)
        self.Order = order
        
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
    def __init__(self, *args, **kwargs):

        order = kwargs.get('order', default=1)
        order += 1
        dim = kwargs.get('dim', default=1)
        kwargs.update({'state_dim': dim * (order+1),
                       'control_dim': 0,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(MA, self).__init__(*args, **kwargs)

        self.Coefficients = np.zeros(order)
        self.Coefficients[:] = kwargs.get('coefficients', default=0)
        self.Order = order
        
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

