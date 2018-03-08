#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Models.py

# Copyright (c) 2016-2017, Alexander Winterl
#
# This file is part of PenguTrack
#
# PenguTrack is free software: you can redistribute and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the Licensem, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PenguTrack. If not, see <http://www.gnu.org/licenses/>.

"""
Module containing model classes to be used with pengu-track detectors and filters.
"""


from __future__ import print_function, division
import numpy as np
#  import scipy.stats as ss
import scipy.optimize as opt


DIM_NAMES = ['X','Y','Z','U','V','W']

class Model(object):
    """
    This Class describes the abstract function of a physical model in the pengu-track package.
    It is only meant for subclassing.

    Attributes
    ----------
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes the abstract function of a physical model in the pengu-track package.
        It is only meant for subclassing.

        Parameters
        ----------
        state_dim: int, optional
            Number of entries in the state-vector.
        control_dim: int, optional
            Number of entries in the control-vector.
        meas_dim: int, optional
            Number of entries in the measurement-vector.
        evo_dim: int, optional
            Number of entries in the evolution-vector.

        """
        self.State_dim = int(kwargs.pop('state_dim', 1))
        self.Control_dim = int(kwargs.pop('control_dim', 1))
        self.Meas_dim = int(kwargs.pop('meas_dim', 1))
        self.Evolution_dim = int(kwargs.pop('evo_dim', 1))

        self.Opt_Params = []
        self.Opt_Params_Shape = {}
        self.Opt_Params_Borders = {}

        self.Initial_Args = args
        self.Initial_KWArgs = kwargs
        
        self.State_Matrix = np.identity(self.State_dim)
        
        self.Control_Matrix = np.identity(max(self.State_dim, self.Control_dim))[:self.State_dim, : self.Control_dim]
        
        self.Measurement_Matrix = np.identity(max(self.State_dim, self.Meas_dim))[: self.Meas_dim, :self.State_dim]
        
        self.Evolution_Matrix = np.identity(max(self.Evolution_dim,
                                                self.State_dim))[:self.State_dim, :self.Evolution_dim]

        self.Extensions = []

        self.Measured_Variables = []
        self.State_Variables = []

    def predict(self, state_vector, control_vector):
        """
        Function to predict next state from current state and external control.

        Parameters
        ----------
        state_vector: array_like
            Latest state vector.
        control_vector: array_like
            Latest control vector.

        Returns
        -------
        prediction: np.array
            New state vector.
        """
        return self.__state_function__(state_vector) + self.__control_function__(control_vector)

    def measure(self, state_vector):
        """
        Function to predict next measurement from current state.

        Parameters
        ----------
        state_vector: array_like
            Latest state vector.

        Returns
        -------
        measurement: np.array
            Expected measurement vector.
        """
        return self.__measurement_function__(state_vector)
        
    def evolute(self, random_vector,  state_vector=None):
        """
        Function to predict next measurement from current state.

        Parameters
        ----------
        random_vector: array_like
            Vector containing the statistical fluctuations.
        state_vector: array_like, optional
            Latest state vector.

        Returns
        -------
            state: np.array
                Calculated state vector.
        """
        if state_vector is None:
            state_vector = np.zeros(self.State_dim)
        return state_vector + self.__evolution_function__(random_vector)

    def __state_function__(self, state_vector):
        return np.dot(self.State_Matrix, state_vector)

    def __control_function__(self, control_vector):
        return np.dot(self.Control_Matrix, control_vector)

    def __measurement_function__(self, state_vector):
        if np.prod(state_vector.shape)==self.State_dim**2:
            return np.dot(self.Measurement_Matrix, np.dot(state_vector, self.Measurement_Matrix.T))
        return np.dot(self.Measurement_Matrix, state_vector)

    def __evolution_function__(self, random_vector):
        return np.dot(self.Evolution_Matrix, random_vector)

    def infer_state(self, meas_vector):
        """
        Tries to infer state from measurement.

        Parameters
        ----------
        meas_vector: array_like
            Vector containing the measurement.

        Returns
        -------
        state: np.array
            Calculated state vector.

        Raises
        ------
        LinAlgError
            If the state can not be inferred do to singularity in a matrix-inversion.
        """
        return np.dot(self.pseudo_inverse(self.Measurement_Matrix), meas_vector)

    @staticmethod
    def pseudo_inverse(matrix):
        """
        Calculates an alternative inverse for non square (non invertible) matrices.

        Parameters
        ----------
        matrix: array_like
            Non square Matrix to be inverted.

        Returns
        -------
        pseudo-inverse: np.array
            Calculated pseudo-inverse.
        """
        matrix = np.asarray(matrix)
        return np.dot(matrix.T, np.linalg.inv(np.dot(matrix, matrix.T)))

    def add_variable(self, var):
        self.Extensions.append(var)
        self.Meas_dim += 1
        self.Evolution_dim += 1
        self.State_dim += 1
        e_mat = np.zeros((self.State_dim, self.Evolution_dim), dtype=float)
        e_mat[:-1, :-1] = self.Evolution_Matrix
        e_mat[-1, -1] = 1.
        self.Evolution_Matrix = e_mat
        s_mat = np.zeros((self.State_dim, self.State_dim), dtype=float)
        s_mat[:-1, :-1] = self.State_Matrix
        s_mat[-1, -1] = 1.
        self.State_Matrix = s_mat
        m_mat = np.zeros((self.Meas_dim, self.State_dim), dtype=float)
        m_mat[:-1, :-1] = self.Measurement_Matrix
        m_mat[-1, -1] = 1.
        self.Measurement_Matrix = m_mat
        c_mat = np.identity(max(self.State_dim, self.Control_dim))[:self.State_dim, : self.Control_dim]
        c_mat[:-1, :] = self.Control_Matrix
        self.Control_Matrix = c_mat

    def optimize_mult(self, states, params=None):
        if params is None:
            params = self.Opt_Params
        InitArgs = list(self.Initial_Args)
        InitKWArgs = dict(self.Initial_KWArgs)

        s = np.vstack([ss[:-1] for ss in states])
        s_p1 = np.vstack([ss[1:] for ss in states])

        p_opt, p_cov = opt.curve_fit(lambda s,*init_args : self.__opt_func__(s, params, *init_args),
                                     s, s_p1.flatten(),
                                     # p0=[self.Initial_KWArgs[o]*np.ones(self.Opt_Params_Shape[o]) for o in sorted(params)],
                                     p0=self.__flatparams__(self.Initial_KWArgs, params=params),
                                     # bounds=[self.Opt_Params_Borders[o] for o in sorted(params)])
                                     bounds=self.__flatborders__(self.Opt_Params_Borders, params=params))
        self.__init__(*InitArgs, **InitKWArgs)
        return p_opt, p_cov

    def vec_from_meas(self, measurement):
        return np.array([measurement[v] for v in self.Measured_Variables])


    def optimize(self, states, params = None):
        if params is None:
            params = self.Opt_Params
        InitArgs = list(self.Initial_Args)
        InitKWArgs = dict(self.Initial_KWArgs)
        s = states[:-1]
        s_p1 = states[1:]
        p_opt, p_cov = opt.curve_fit(lambda s,*init_args : self.__opt_func__(s, params, *init_args),
                                     s, s_p1.flatten(),
                                     # p0=[self.Initial_KWArgs[o]*np.ones(self.Opt_Params_Shape[o]) for o in sorted(params)],
                                     p0=self.__flatparams__(self.Initial_KWArgs, params=params),
                                     # bounds=[self.Opt_Params_Borders[o] for o in sorted(params)])
                                     bounds=self.__flatborders__(self.Opt_Params_Borders, params=params))
        self.__init__(*InitArgs, **InitKWArgs)
        return p_opt, p_cov

    # def __compare_measurements__(self, measurements):
    #     m = measurements[:-1]
    #     m_p1 = measurements[1:]
    #     return np.mean(np.linalg.norm(m_p1-np.array([self.measure(self.predict(self.infer_state(mm),
    #                                                                            np.zeros((self.Control_dim, 0)))) for mm in m]), axis=1))
    # def __compare_states__(self, states):
    #     s = states[:-1]
    #     s_p1 = s[1:]
    #     return np.mean(np.linalg.norm(s_p1-np.array([self.predict(ss,
    #                                                               np.zeros((self.Control_dim, 0))) for ss in s]), axis=1))

    def __opt_func__(self, states, params, *init_args):
        if params is None:
            params = self.Opt_Params
        init_args = self.__unflatparams__(init_args, params=params)
        kwargs = {}
        kwargs.update(self.Initial_KWArgs)
        for i,o in enumerate(sorted(params)):
            kwargs[o] = init_args[i]
        self.__init__(*self.Initial_Args, **kwargs)
        return np.array([self.predict(ss,np.zeros((self.Control_dim, 1))) for ss in states]).flatten()

    def __flatparams__(self, param_dict, params=None):
        if params is None:
            params = self.Opt_Params
        param_list = [param_dict[o] * np.ones(self.Opt_Params_Shape[o]) for o in sorted(params)]
        return np.hstack([p.flatten() for p in param_list])

    def __unflatparams__(self, param_array, params=None):
        if params is None:
            params = self.Opt_Params
        lens = [0]
        lens.extend([len(np.ones(self.Opt_Params_Shape[o]).flatten()) for o in sorted(params)])
        cum_lens = np.cumsum(lens)
        # return dict([[o, np.array(param_array[c:l]).reshape(self.Opt_Params_Shape[o])]for l, c, o in zip(cum_lens[1:], cum_lens[:-1], sorted(params))])
        return [np.array(param_array[c:l]).reshape(self.Opt_Params_Shape[o])for l, c, o in zip(cum_lens[1:], cum_lens[:-1], sorted(params))]

    def __flatborders__(self, borderarray, params=None):
        if params is None:
            params = self.Opt_Params
        border_0 = [borderarray[o][0] * np.ones(self.Opt_Params_Shape[o]) for o in sorted(params)]
        border_1 = [borderarray[o][1] * np.ones(self.Opt_Params_Shape[o]) for o in sorted(params)]
        return np.hstack([p.flatten() for p in border_0]),np.hstack([p.flatten() for p in border_1])

class RandomWalk(Model):
    """
    This Class describes an easy random walk model.

    Attributes
    ----------
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        """
        dim = kwargs.get('dim', 2)

        kwargs.update({'state_dim': dim,
                       'control_dim': dim,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(RandomWalk, self).__init__(*args, **kwargs)

        self.Opt_Params = []
        self.Opt_Params_Shape = {}
        self.Opt_Params_Borders = {}

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])


class Ballistic(Model):
    """
    This Class describes an simple ballistic model.

    Attributes
    ----------
    Damping: np.array
        Damping constant(s) for ballistic model.
    Mass: float
        Mass of Object.
    Timeconst: float
        Step-width of time-discretization.
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        damping: array_like, optional
            Damping constant(s) of different dimensions for the model.
        mass: float, optional
            Mass of the modelled object.
        timeconst: float, optional
            Step-width of time-discretization.

        """
        dim = kwargs.get('dim', 1)
        kwargs.update({'state_dim': dim*3,
                       'control_dim': dim*3,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(Ballistic, self).__init__(*args, **kwargs)

        self.Damping = np.zeros(dim)
        self.Damping[:] = kwargs.get('damping', 0)
        self.Mass = float(kwargs.get('mass', 1))
        self.Timeconst = float(kwargs.get('timeconst', 1))

        self.Opt_Params = ['damping', 'mass', 'timeconst']
        self.Opt_Params_Shape = {'damping': (dim,), 'mass': (1,), 'timeconst': (1,)}
        self.Opt_Params_Borders = {'damping': (0, np.inf), 'mass': (0, np.inf), 'timeconst': (0, np.inf)}

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

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Velocity%s"%DIM_NAMES[i])
            self.State_Variables.append("Force%s"%DIM_NAMES[i])



class VariableSpeed(Model):
    """
    This Class describes a simple model, assuming slow changing speed-vectors.

    Attributes
    ----------
    Damping: np.array
        Damping constant(s) for ballistic model.
    Timeconst: float
        Step-width of time-discretization.
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings    MultiKal.Model.optimize(state_dict[0])
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes the function of a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        damping: array_like, optional
            Damping constant(s) of different dimensions for the model.
        mass: float, optional
            Mass of the modelled object.
        timeconst: float, optional
            Step-width of time-discretization.

        """

        dim = kwargs.get('dim', 1)
        kwargs.update({'state_dim': dim*2,
                       'control_dim': dim*2,
                       'meas_dim': dim,
                       'evo_dim': dim})

        self.Damping = np.zeros(dim)
        self.Damping[:] = kwargs.get('damping', 0)
        kwargs.update({'damping': self.Damping})
        self.Timeconst = float(kwargs.get('timeconst', 1))
        kwargs.update({'timeconst': self.Timeconst})

        super(VariableSpeed, self).__init__(*args, **kwargs)

        self.Opt_Params = ['damping', 'timeconst']
        self.Opt_Params_Shape = {'damping': (dim,), 'timeconst': (1,)}
        self.Opt_Params_Borders = {'damping': (0, np.inf), 'timeconst': (0, np.inf)}
        n = dim
        matrix = np.zeros((2*n, 2*n))
        matrix[::2, ::2] = np.diag(np.ones(n))
        matrix[1::2, 1::2] = np.diag(np.exp(-1*self.Damping))
        matrix[::2, 1::2] = np.diag(np.ones(n)*self.Timeconst)

        meas = np.zeros((n, n*2))
        meas[:, ::2] = np.diag(np.ones(n))

        evo = np.zeros((n*2, n))
        evo[1::2, :] = np.diag(np.ones(n))

        self.State_Matrix = matrix
        self.Measurement_Matrix = meas
        self.Evolution_Matrix = evo

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Velocity%s"%DIM_NAMES[i])


class BallisticWSpeed(VariableSpeed):
    """
    This Class describes a simple model, assuming slow changing speed-vectors.

    Attributes
    ----------
    Damping: np.array
        Damping constant(s) for ballistic model.
    Timeconst: float
        Step-width of time-discretization.
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings    MultiKal.Model.optimize(state_dict[0])
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes the function of a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        damping: array_like, optional
            Damping constant(s) of different dimensions for the model.
        mass: float, optional
            Mass of the modelled object.
        timeconst: float, optional
            Step-width of time-discretization.

        """
        dim = kwargs.get('dim', 1)
        super(BallisticWSpeed, self).__init__(*args, **kwargs)
        self.Meas_dim = self.State_dim
        self.Measurement_Matrix = np.diag(np.ones(self.State_dim))

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            self.Measured_Variables.append("Velocity%s"%DIM_NAMES[i])
            self.State_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Velocity%s"%DIM_NAMES[i])





class AR(Model):
    """
    This Class describes an auto-regressive model.

    Attributes
    ----------
    Order: int
        Order of the AR-Process. Order = 1 equals an AR1-Process.
    Coefficients: np.array
        Coefficients of the AR-Process. These describe the time-dependent behaviour of the model.
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes the function of a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        order: int optional
            Order of the AR-Process. Order = 1 equals an AR1-Process.
        coefficients: array_like, optional
            Coefficients describing the time evolution of the AR-Process.

        """

        order = kwargs.get('order', 1)
        order += 1
        dim = kwargs.get('dim', 1)
        kwargs.update({'state_dim': dim * order,
                       'control_dim': dim * order,
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(AR, self).__init__(*args, **kwargs)

        self.Coefficients = np.zeros(order)
        self.Coefficients[:] = kwargs.get('coefficients', 0)
        self.Order = order

        self.Opt_Params = ['coefficients']
        self.Opt_Params_Shape = {'coefficients': (order,)}
        self.Opt_Params_Borders = {'coefficients': (-np.inf, np.inf)}
        
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

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            for j in range(order):
                self.State_Variables.append("%s_Order_Parameter%s"%(j, DIM_NAMES[i]))

 
class MA(Model):
    """
    This Class describes an moving-average model.

    Attributes
    ----------
    Order: int
        Order of the MA-Process. Order = 1 equals an MA1-Process.
    Coefficients: np.array
        Coefficients of the AR-Process. These describe the time-dependent behaviour of the model.
    State_dim: int
        Number of entries in the state-vector.
    Control_dim: int
        Number of entries in the control-vector.
    Meas_dim: int
        Number of entries in the measurement-vector.
    Evolution_dim: int
        Number of entries in the evolution-vector.
    Opt_Params: list of strings
        Parameters of model, which can be optimized.
    Opt_Params_Shape: dict
        Keys are Opt_Params, entries are tuples containing shapes of the corresponding Parameters.
    Opt_Params_Borders: dict
        Keys are Opt_Params, entries are tuples containing borders of the corresponding Parameters.
    Initial_Args: list
        The arguments, which were given in the init function.
    Initial_KWArgs: dict
        The keyword-arguments, which were given in the init function.
    State_Matrix: np.array
        The evolution-matrix of the unperturbed system.
    Control_Matrix: np.array
        The evolution-matrix, which shows the influence of external control.
    Measurement_Matrix: np.array
        The matrix, which shows how the state vectors are projected into a measurement vector.
    Evolution_Matrix: np.array
        The matrix, which shows the influence of statistical fluctuations to the state.
    Measured_Variables: list
        List of variables, that are measured within the model.
    State_Variables: list
        List of variables, that are tracked within the model.
    Extensions: list
        List of measured parameters not included in standard model.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes the function of a physical model in the pengu-track package.

        Parameters
        ----------
        dim: int, optional
            Number of dimensions in which the random walk happens.
        order: int optional
            Order of the MA-Process. Order = 1 equals an MA1-Process.
        coefficients: array_like, optional
            Coefficients describing the time evolution of the MA-Process.
        """

        order = kwargs.get('order', 1)
        order += 1
        dim = kwargs.get('dim', 1)
        kwargs.update({'state_dim': dim * (order+1),
                       'control_dim': dim * (order+1),
                       'meas_dim': dim,
                       'evo_dim': dim})

        super(MA, self).__init__(*args, **kwargs)

        self.Coefficients = np.zeros(order)
        self.Coefficients[:] = kwargs.get('coefficients', 0)
        self.Order = order

        self.Opt_Params = ['coefficients']
        self.Opt_Params_Shape = {'coefficients': (order,)}
        self.Opt_Params_Borders = {'coefficients': (-np.inf, np.inf)}
        
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

        self.Measured_Variables = []
        self.State_Variables = []
        for i in range(dim):
            self.Measured_Variables.append("Position%s"%DIM_NAMES[i])
            self.State_Variables.append("Position%s"%DIM_NAMES[i])
            for j in range(order):
                self.State_Variables.append("%s_History_%s"%(j,DIM_NAMES[i]))