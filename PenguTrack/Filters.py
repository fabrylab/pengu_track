#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filters.py

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
Module containing filter classes to be used with pengu-track detectors and models.
"""

from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
import sys
import copy
from PenguTrack.Detectors import Measurement
from .Assignment import *
import scipy.integrate as integrate
#  import scipy.optimize as opt
from .Detectors import array_to_measurement, array_to_pandasDF, pandasDF_to_array, pandasDF_to_measurement, measurements_to_array, measurements_to_pandasDF
import pandas

class Filter(object):
    """
    This Class describes the abstract function of a filter in the pengu-track package.
    It is only meant for subclassing.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.

    """
    def __init__(self, model, no_dist=False, meas_dist=ss.uniform(), state_dist=ss.uniform(), prob_update=True,
                 *args, **kwargs):
        """
        This Class describes the abstract function of a filter in the pengu-track package.
        It is only meant for subclassing.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        meas_dist: scipy.stats.distributions object
            The distribution which describes measurement uncertainty.
        state_dist: scipy.stats.distributions object
            The distribution which describes state vector fluctuations.
        """

        self._meas_cov_ = None
        self._meas_mu_ = None
        self._meas_sig_1 = None
        self._meas_norm_ = None
        self._meas_is_gaussian_ = False
        self._state_cov_ = None
        self._state_mu_ = None
        self._state_sig_1 = None
        self._state_norm_ = None
        self._state_is_gaussian_ = False

        self.Model = model
        self.Measurement_Distribution = meas_dist
        self.State_Distribution = state_dist
        
        self.X = {}
        self.X_error = {}
        self.Predicted_X = {}
        self.Predicted_X_error = {}
        
        self.Measurements = {}
        self.Controls = {}

        self.NoDist = bool(no_dist)
        self.ProbUpdate = bool(prob_update)

    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """


        assert not (u is None and i is None), "One of control vector or time stamp must be specified"

        # Generate i
        if i is None:
            i = max(self.Predicted_X.keys())+1
        # Generate u
        if u is None:
            try:
                u = self.Controls[i-1]
            except KeyError:
                u = np.zeros((self.Model.Control_dim, 1))
        else:
            self.Controls.update({i-1: u})

        # Try to get previous x from believe
        try:
            x = self.X[i-1]
        except KeyError:
            # Try to get previous prediction
            try:
                x = self.Predicted_X[i-1]
            except KeyError:
                # Try recursive prediction from previous timesteps
                if np.any([k < i for k in self.Predicted_X.keys()]) or np.any([k < i for k in self.X.keys()]) :
                    print('Recursive Prediction, i = %s' % i)
                    u_, i_ = self.predict(u=u, i=i-1)
                    x = self.Predicted_X[i-1]
                else:
                    raise KeyError("Nothing to predict from. Need initial value")

        # Make simplest possible Prediction (can be overwritten)
        self.Predicted_X.update({i: self.Model.predict(x, u)})
        return u, i
        
    def update(self, z=None, i=None):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """


        assert not (z is None and i is None), "One of measurement vector or time stamp must be specified"

        # Generate i
        if i is None:
            i = max(self.Measurements.keys())+1
        # Generate z
        if z is None:
            try:
                z = self.Measurements[i]
            except KeyError:
                raise KeyError("No measurement for timepoint %s." % i)
        else:
            if not isinstance(z, Measurement):
                z = np.asarray(z).flatten()
                assert self.Model.Meas_dim == len(z),\
                    "Measurement input shape %s is not equal to model measurement dimension %s"%(
                        len(z),self.Model.Meas_dim)
                z = Measurement(1.0, position=z)
            measurement = z.copy()
            self.Measurements.update({i: measurement})
        # measurement = copy(z)
        # simplest possible update
        # try:
        #     self.X.update({i: np.asarray([z.PositionX, z.PositionY, z.PositionZ])})
        # except(ValueError, AttributeError):
        #     try:
        #         self.X.update({i: np.asarray([z.PositionX, z.PositionY])})
        #     except(ValueError, AttributeError):
        #         self.X.update({i: np.asarray([z.PositionX])})
        #
        z = np.asarray(self.Model.vec_from_meas(measurement))
        # if len(self.Model.Extensions) > 0:
        #     z = np.asarray(np.hstack([np.asarray([measurement[v] for v in self.Model.Measured_Variables]),
        #                     np.asarray([measurement.Data[var] for var in self.Model.Extensions])]))
        # else:
        #     z = np.asarray([measurement[v] for v in self.Model.Measured_Variables])

        self.X.update({i: z})


        return measurement, i

    def filter(self, u=None, z=None, i=None):
        """
        Function to get predictions from the model and update the same timestep i.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like, optional
            Recent control-vector.
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        self.predict(u=u, i=i)
        x = self.update(z=z, i=i)
        return x

    def log_prob(self, keys=None, measurements=None, update=False):
        if keys is None:
            keys = self.X.keys()
        if measurements is None:
            try:
                measurements = dict([[k,self.Measurements[k]] for k in keys])
            except KeyError:
                return None
        prob = 0
        pending_downdates = []
        pending_downpredicts =[]
        for k in keys:
            if k in self.X and k in self.Predicted_X:
                if update:
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k, measurements[k])
            elif k in self.X:
                self.predict(i=k)
                if update:
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k,measurements[k])
            elif k in self.Predicted_X and k in measurements:
                if update:
                    self.update(z=measurements[k],i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k, measurements[k])
            elif k in self.Predicted_X and k in self.Measurements:
                if update:
                    self.update(z=self.Measurements[k],i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k)
            elif k in measurements:
                self.predict(i=k)
                pending_downpredicts.append(k)
                if update:
                    self.update(z=measurements[k], i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k, measurements[k])
            elif k in self.Measurements:
                self.predict(i=k)
                pending_downpredicts.append(k)
                if update:
                    self.update(z=self.Measurements[k], i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k)
            else:
                raise ValueError("Probability for key %s could not be calculated! (missing states)"%k)

        for k in pending_downdates:
            self.downdate(k)
        for k in pending_downpredicts:
            self.unpredict(k)
        return prob

    def _log_prob_(self, key):
        return self._state_log_prob_(key)

    def _state_log_prob_(self, key):
        if self.NoDist:
            # return np.linalg.norm(self.X[key]-self.Predicted_X[key])
            return np.linalg.norm(self.Model.pos_from_state(self.X[key])-self.Model.pos_from_state(self.Predicted_X[key]))
        if self._state_is_gaussian_:
            x = self.X[key]-self.Predicted_X[key] - self._state_mu_[:, None]
            return float(-np.dot(x.T, np.dot(self._state_sig_1**2, x)))
            # return float(self._state_norm_ - 0.5 * np.dot(x.T, np.dot(self._state_sig_1, x)))
            # return float(self._state_norm_ - 0.5 * np.dot(x.T, np.dot(self._state_sig_1, x)))
        return self.State_Distribution.logcdf((self.X[key]-self.Predicted_X[key]).T)
        # return self.State_Distribution.logpdf((self.X[key]-self.Predicted_X[key]).T)

    def _meas_log_prob(self, key, measurement=None):
        if measurement is None:
            measurement = self.Measurements[key]
        if self.NoDist:
            # return np.linalg.norm(self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key]))
            return np.linalg.norm(self.Model.pos_from_measurement(measurement)-self.Model.pos_from_state(self.Predicted_X[key]))
        if self._meas_is_gaussian_:
            x = self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key]) - self._meas_mu_[:,None]
            return float(- np.dot(x.T, np.dot(self._meas_sig_1**2, x)/self.Model.Meas_dim))
            # return float(self._meas_norm_ -0.5*np.dot(x.T, np.dot(self._meas_sig_1, x)))
            # return float(self._meas_norm_ -0.5*np.dot(x.T, np.dot(self._meas_sig_1, x)))
        return self.Measurement_Distribution.logcdf((self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key])).T)
        # return self.Measurement_Distribution.logpdf((self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key])).T)

    def __setattr__(self, key, value):
        if key == "Measurement_Distribution":
            if isinstance(value, ss._multivariate.multivariate_normal_frozen):
                self._meas_cov_ = value.cov
                self._meas_mu_ = value.mean
                self._meas_sig_1 = np.linalg.inv(value.cov)
                self._meas_norm_ = -0.5*np.log(((2*np.pi)**max(value.cov.shape)*np.linalg.det(self._meas_cov_)))
                self._meas_is_gaussian_ = True
        if key == "State_Distribution":
            if isinstance(value, ss._multivariate.multivariate_normal_frozen):
                self._state_cov_ = value.cov
                self._state_mu_ = value.mean
                self._state_sig_1 = np.linalg.inv(value.cov)
                self._state_norm_ = -0.5*np.log((2*np.pi)**max(value.cov.shape)*np.linalg.det(self._state_cov_))
                self._state_is_gaussian_ = True
        super(Filter, self).__setattr__(key, value)

    # def _log_prob_(self, key):
    #     measurement = self.Measurements[key]
    #     try:
    #         position = np.asarray([measurement.PositionX,
    #                                measurement.PositionY,
    #                                measurement.PositionZ])
    #     except (ValueError, AttributeError):
    #         try:
    #             position = np.asarray([measurement.PositionX,
    #                                    measurement.PositionY])
    #         except (ValueError, AttributeError):
    #             position = np.asarray([measurement.PositionX])
    #     return self.Measurement_Distribution.logpdf(position-self.Model.measure(self.X[key]))

    def downfilter(self, t=None):
        """
        Function erases the timestep t from the class dictionaries (Measurement, X, Preditcion)

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """
        # Generate t
        if t is None:
            t = max([max(self.X.keys()),
                     max(self.Predicted_X.keys()),
                     max(self.Measurements.keys()),
                     self.Controls.keys()])
        # Remove all entries for this timestep
        self.Measurements.pop(t, None)
        self.Controls.pop(t, None)
        self.X.pop(t, None)
        self.X_error.pop(t, None)
        self.Predicted_X.pop(t, None)
        self.Predicted_X_error.pop(t, None)

    def downdate(self, t=None):
        """
        Function erases the time-step t from the Measurements and Believes.

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """

        # Generate t
        if t is None:
            t = max(self.X.keys())
        # Remove believe and Measurement entries for this timestep
        self.X.pop(t, None)
        self.X_error.pop(t, None)
        self.Measurements.pop(t, None)

    def unpredict(self, t=None):
        """
        Function erases the time-step t from the Believes, Predictions and Controls.

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """
        # Generate t
        if t is None:
            t = max(self.Predicted_X.keys())
        # Remove believe and Prediction entries for this timestep
        self.Controls.pop(t, None)
        self.X.pop(t, None)
        self.X_error.pop(t, None)
        self.Predicted_X.pop(t, None)
        self.Predicted_X_error.pop(t, None)

    def fit(self, u, z):
        """
        Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde.

        Parameters
        ----------
        u: array_like
            List of control-vectors.
        z: array_like
            List of measurement-vectors.
        """
        u = np.asarray(u)
        z = np.asarray(z)
        assert u.shape[0] == z.shape[0]

        for i in range(z.shape[0]):
            self.predict(u=u[i], i=i+1)
            self.update(z=z[i], i=i+1)
            print(self.log_prob())

        return np.asarray(self.X.values(), dtype=float),\
               np.asarray(self.X_error.values(), dtype=float),\
               np.asarray(self.Predicted_X.values(), dtype=float),\
               np.asarray(self.Predicted_X_error.values(), dtype=float)

    def cost_from_logprob(self, log_prob, **kwargs):
        return cost_from_logprob(log_prob, **kwargs)

class KalmanBaseFilter(Filter):
    """
    This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
    predictions and measurements.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.
    A: np.asarray
        State-Transition-Matrix, describing the evolution of the state without fluctuation.
        Received from the physical Model.
    B: np.asarray
        State-Control-Matrix, describing the influence of external-control on the states.
        Received from the physical Model.
    C: np.asarray
        Measurement-Matrix , describing the projection of the measurement-vector from the state-vector.
        Received from the physical Model.
    G: np.asarray
        Evolution-Matrix, describing the evolution of state vectors by fluctuations from the state-distribution.
        Received from the physical Model.
    Q: np.asarray
        Covariance matrix (time-evolving) for the state-distribution.
    Q_0: np.asarray
        Covariance matrix (initial state) for the state-distribution.
    R: np.asarray
        Covariance matrix (time-evolving) for the measurement-distribution.
    R_0: np.asarray
        Covariance matrix (initial state) for the measurement-distribution.
    P_0: np.asarray
        Covariance matrix (initial state) for the believe-distribution.
    """
    def __init__(self, model, evolution_variance, measurement_variance, **kwargs):
        """
        This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
        predictions and measurements.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        evolution_variance: array_like
            Vector containing the estimated variances of the state-vector-entries.
        measurement_variance: array_like
            Vector containing the estimated variances of the measurement-vector-entries.
        """
        self.Model = model

        evolution_variance = np.asarray(evolution_variance, dtype=float)
        if evolution_variance.shape == (1,):
            evolution_variance = np.diag(np.ones(int(self.Model.Evolution_dim))*evolution_variance)
        # if evolution_variance.shape != (int(self.Model.Evolution_dim),):
        #     evolution_variance = np.ones(self.Model.Evolution_dim) * np.mean(evolution_variance)

        self.Evolution_Variance = evolution_variance
        self.Q = evolution_variance
        self.Q_0 = evolution_variance

        measurement_variance = np.asarray(measurement_variance, dtype=float)
        if measurement_variance.shape == (1,):
            measurement_variance = np.diag(np.ones(int(self.Model.Meas_dim))*measurement_variance)
        # if measurement_variance.shape != (int(self.Model.Meas_dim),):
        #     measurement_variance = np.ones(self.Model.Meas_dim) * np.mean(measurement_variance)

        self.Measurement_Variance = measurement_variance
        self.R = measurement_variance
        self.R_0 = measurement_variance

        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

        self.K = None

        p = np.dot(np.dot(self.C.T, self.R_0), self.C) + \
            np.dot(self.A, np.dot(np.dot(np.dot(self.G, self.Q), self.G.T), self.A.T))#np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.P_0 = p

        kwargs.update(dict(meas_dist=ss.multivariate_normal(cov=self.R),
                           state_dist=ss.multivariate_normal(cov=self.P_0)))
        super(KalmanBaseFilter, self).__init__(model, **kwargs)

        self.X_error.update({0: p})
        self.Predicted_X_error.update({0: p})

    def _handle_state_(self, i, allow_predict=False, return_prediction=False):
        if i in self.X and not return_prediction:
            return self.X[i]
        elif i in self.Predicted_X:
            return self.Predicted_X[i]
        elif allow_predict:
            u, i = self.predict(i=i)
            return self.Predicted_X[i]
        else:
            if return_prediction:
                raise KeyError("No entry in prediction memory for timestamp %s" % i)
            else:
                raise KeyError("No entry in prediction or state memory for timestamp %s" % i)

    def _handle_error_(self, i, allow_predict=False, return_prediction=False):
        if i in self.X_error and not return_prediction:
            return self.X_error[i]
        elif i in self.Predicted_X_error:
            return self.Predicted_X_error[i]
        elif allow_predict:
            u, i = self.predict(i=i)
            return self.Predicted_X_error[i]
        else:
            if return_prediction:
                raise KeyError("No entry in prediction memory for timestamp %s" % i)
            else:
                return self.P_0


class KalmanFilter(KalmanBaseFilter):
    # def __init__(self, model, evolution_variance, measurement_variance, **kwargs):
    #     """
    #     This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
    #     predictions and measurements.
    #
    #     Parameters
    #     ----------
    #     model: PenguTrack.model object
    #         A physical model to gain predictions from data.
    #     evolution_variance: array_like
    #         Vector containing the estimated variances of the state-vector-entries.
    #     measurement_variance: array_like
    #         Vector containing the estimated variances of the measurement-vector-entries.
    #     """
    #     super(KalmanFilter, self).__init__(model, evolution_variance, measurement_variance, **kwargs)
    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """
        u, i = super(KalmanFilter, self).predict(u=u, i=i)

        x = self._handle_state_(i-1)
        p = self._handle_error_(i-1)

        p_ = np.dot(np.dot(self.A, p), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)
        x_ = np.dot(self.A, x) + np.dot(self.B, u)

        self.Predicted_X.update({i: x_})
        self.Predicted_X_error.update({i: p_})

        return u, i

    def update(self, z=None, i=None):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        z, i = super(KalmanFilter, self).update(z=z, i=i)
        measurement = copy(z)
        z = np.asarray(self.Model.vec_from_meas(measurement))

        x = self._handle_state_(i, allow_predict=True, return_prediction=True)
        p = self._handle_error_(i, allow_predict=True, return_prediction=True)

        try:
            k = np.dot(np.dot(p, self.C.transpose()),
                       np.linalg.inv(np.dot(np.dot(self.C, p), self.C.transpose()) + self.R))
            self.K = k
        except np.linalg.LinAlgError:
            e = sys.exc_info()[1]
            print(p)
            raise np.linalg.LinAlgError(e)

        y = z - np.dot(self.C, x)

        x_ = x + np.dot(k, y)
        p_ = p - np.dot(np.dot(k, self.C), p)

        self.X.update({i: x_})
        self.X_error.update({i: p_})

        return measurement, i


class InformationFilter(KalmanBaseFilter):
    """
    This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
    predictions and measurements.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.
    A: np.asarray
        State-Transition-Matrix, describing the evolution of the state without fluctuation.
        Received from the physical Model.
    B: np.asarray
        State-Control-Matrix, describing the influence of external-control on the states.
        Received from the physical Model.
    C: np.asarray
        Measurement-Matrix , describing the projection of the measurement-vector from the state-vector.
        Received from the physical Model.
    G: np.asarray
        Evolution-Matrix, describing the evolution of state vectors by fluctuations from the state-distribution.
        Received from the physical Model.
    Q: np.asarray
        Covariance matrix (time-evolving) for the state-distribution.
    Q_0: np.asarray
        Covariance matrix (initial state) for the state-distribution.
    R: np.asarray
        Covariance matrix (time-evolving) for the measurement-distribution.
    R_0: np.asarray
        Covariance matrix (initial state) for the measurement-distribution.
    P_0: np.asarray
        Covariance matrix (initial state) for the believe-distribution.
    """
    def __init__(self, model, evolution_variance, measurement_variance, **kwargs):
        """
        This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
        predictions and measurements.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        evolution_variance: array_like
            Vector containing the estimated variances of the state-vector-entries.
        measurement_variance: array_like
            Vector containing the estimated variances of the measurement-vector-entries.
        """
        super(InformationFilter, self).__init__(model, evolution_variance, measurement_variance, **kwargs)
        self.Model = model

        evolution_variance = np.asarray(evolution_variance, dtype=float)
        # if evolution_variance.shape != (int(self.Model.Evolution_dim),):
        #     evolution_variance = np.ones(self.Model.Evolution_dim) * np.mean(evolution_variance)

        self.Evolution_Variance = evolution_variance
        self.Q = np.diag(evolution_variance)
        self.Q_0 = np.diag(evolution_variance)
        self.Q_inv = np.linalg.inv(self.Q)

        measurement_variance = np.asarray(measurement_variance, dtype=float)
        # if measurement_variance.shape != (int(self.Model.Meas_dim),):
        #     measurement_variance = np.ones(self.Model.Meas_dim) * np.mean(measurement_variance)

        self.Measurement_Variance = measurement_variance
        self.R = np.diag(measurement_variance)
        self.R_0 = np.diag(measurement_variance)
        self.R_inv = np.linalg.inv(self.R)

        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

        p = np.dot(np.dot(self.C.T, self.R_0), self.C) + np.dot(np.dot(self.G, self.Q), self.G.T)#np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.P_0 = p
        self.O_0 = np.linalg.inv(p)

        kwargs.update(dict(meas_dist=ss.multivariate_normal(cov=self.R),
                           state_dist=ss.multivariate_normal(cov=self.P_0)))

        self.X_error.update({0: p})
        self.Predicted_X_error.update({0: p})

    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """
        u, i = super(super(InformationFilter, self), self).predict(u=u, i=i)

        x_old = self._handle_state_(i-1)
        p_old = self._handle_error_(i-1)

        o_old_inv = p_old
        o_old = np.linalg.inv(p_old)
        z_old = np.dot(o_old, x_old)

        new_o_inv = np.dot(np.dot(self.A, o_old_inv), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)

        z_ = np.dot(o_old, (np.dot(np.dot(self.A, o_old_inv), z_old) + np.dot(self.B, u)))

        self.Predicted_X.update({i: np.dot(new_o_inv, z_)})
        self.Predicted_X_error.update({i: new_o_inv})

        return u, i

    def update(self, z=None, i=None):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        z, i = super(InformationFilter, self).update(z=z, i=i)
        measurement = copy(z)
        z = np.asarray(self.Model.vec_from_meas(measurement))

        x = self._handle_state_(i, allow_predict=True, return_prediction=True)
        p = self._handle_error_(i, allow_predict=True, return_prediction=True)

        o_inv = p
        o = np.linalg.inv(p)
        zz = np.dot(o, x)

        # o_inv = p-np.dot(np.dot(np.dot(np.dot(o_inv, self.C.T), np.linalg.inv(self.R + np.dot(np.dot(self.C, o_inv), self.C.T))), self.C), o)
        o = np.dot(np.dot(self.C.T, self.R_inv), self.C) + o
        zz_ = np.dot(np.dot(self.C.T, self.R_inv), z) + zz

        o_inv = np.linalg.inv(o)

        self.X.update({i: np.dot(o_inv, zz_)})
        self.X_error.update({i: o_inv})

        return measurement, i


class AdaptedKalmanFilter(KalmanBaseFilter):
    def __init__(self, model, evolution_variance, measurement_variance, **kwargs):
        """
        This Class describes a kalman-filter in the pengu-track package. It calculates actual believed values from
        predictions and measurements.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        """
        super(AdaptedKalmanFilter, self).__init__(model, evolution_variance, measurement_variance, **kwargs)
        self.Y = {}

    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """
        u, i = super(AdaptedKalmanFilter, self).predict(u=u, i=i)

        x = self._handle_state_(i-1)
        p = self._handle_error_(i-1)

        p_ = np.dot(np.dot(self.A, p), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)
        x_ = np.dot(self.A, x) + np.dot(self.B, u)

        self.Predicted_X.update({i: x_})
        self.Predicted_X_error.update({i: p_})

        return u, i


    def update(self, z=None, i=None):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        z, i = super(AdaptedKalmanFilter, self).update(z=z, i=i)
        measurement = copy(z)
        z = np.asarray(self.Model.vec_from_meas(measurement))

        x = self._handle_state_(i, allow_predict=True, return_prediction=True)
        p = self._handle_error_(i, allow_predict=True, return_prediction=True)

        y = z - np.dot(self.C, x)
        self.Y.update({i: y})

        sig = y[:, None] * y[None, :]

        alpha = np.trace(sig)/np.trace(np.dot(np.dot(self.C, p), self.C.T) + self.R)
        if alpha > 1:
            alpha = 1
        # elif alpha<0:
        #     alpha = 0
        print(alpha)

        try:
            k = (1./alpha)*np.dot(np.dot(p, self.C.transpose()),
                       np.linalg.inv(np.dot(np.dot(self.C * (1./alpha), p), self.C.transpose()) + self.R))
            self.K = k
        except np.linalg.LinAlgError:
            e = sys.exc_info()[1]
            print(p)
            raise np.linalg.LinAlgError(e)

        x_ = x + np.dot(k, y)
        p_ = (p - np.dot(np.dot(k, self.C), p))*(1./alpha)

        self.X.update({i: x_})
        self.X_error.update({i: p_})

        return measurement, i


class AdvancedKalmanFilter(KalmanFilter):
    """
    This Class describes a advanced, self tuning version of the kalman-filter in the pengu-track package.
    It calculates actual believed values from predictions and measurements.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Lag: int
        The number of state-vectors, which are taken into acount for the self-tuning algorithm.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.
    A: np.asarray
        State-Transition-Matrix, describing the evolution of the state without fluctuation.
        Received from the physical Model.
    B: np.asarray
        State-Control-Matrix, describing the influence of external-control on the states.
        Received from the physical Model.
    C: np.asarray
        Measurement-Matrix , describing the projection of the measurement-vector from the state-vector.
        Received from the physical Model.
    G: np.asarray
        Evolution-Matrix, describing the evolution of state vectors by fluctuations from the state-distribution.
        Received from the physical Model.
    Q: np.asarray
        Covariance matrix (time-evolving) for the state-distribution.
    Q_0: np.asarray
        Covariance matrix (initial state) for the state-distribution.
    R: np.asarray
        Covariance matrix (time-evolving) for the measurement-distribution.
    R_0: np.asarray
        Covariance matrix (initial state) for the measurement-distribution.
    P_0: np.asarray
        Covariance matrix (initial state) for the believe-distribution.
    """
    def __init__(self, *args, **kwargs):
        """
        This Class describes a advanced, self tuning version of the kalman-filter in the pengu-track package.
        It calculates actual believed values from predictions and measurements.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        evolution_variance: array_like
            Vector containing the estimated variances of the state-vector-entries.
        measurement_variance: array_like
            Vector containing the estimated variances of the measurement-vector-entries.
        """
        super(AdvancedKalmanFilter, self).__init__(*args, **kwargs)
        self.Lag = -1 * int(kwargs.pop('lag', -1))
        self._is_obs_ = self.is_observable()

    # def predict(self, *args, **kwargs):
    #     """
    #     Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.
    #
    #     Parameters
    #     ----------
    #     u: array_like, optional
    #         Recent control-vector.
    #     i: int
    #         Recent/corresponding time-stamp
    #
    #     Returns
    #     ----------
    #     u: array_like
    #         Recent control-vector.
    #     i: int
    #         Recent/corresponding time-stamp.
    #     """
    #     if self.Lag == 1 or (-1 * self.Lag > len(self.Predicted_X.keys())):
    #         lag = 0
    #     else:
    #         lag = self.Lag
    #     self.Q = np.dot(np.dot(self.G.T, np.cov(np.asarray(self.Predicted_X.values()[lag:]).T)), self.G)
    #     print("Q at %s"%self.Q)
    #     if np.any(np.isnan(self.Q)):# or np.any(np.linalg.eigvals(self.Q) < np.diag(self.Q_0)):
    #         self.Q = self.Q_0
    #     # print("Q at %s"%self.Q)
    #     return super(AdvancedKalmanFilter, self).predict(*args, **kwargs)

    def update(self, *args, **kwargs):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        z, i = super(AdvancedKalmanFilter, self).update(*args, **kwargs)

        frames = set(self.Measurements.keys()).intersection(self.Predicted_X.keys())

        if self._is_obs_ and len(frames) >= self.Model.State_dim:
            frames = sorted(frames)[-self.Model.State_dim:]
            print("prev", np.diag(self.Q), np.diag(self.R))
            cor_vals = np.asarray([self.Model.vec_from_meas(self.Measurements[f])-self.Model.measure(self.Predicted_X[f])
                                 for f in frames])
            cor = np.mean([cor_vals[-i, None, :] * cor_vals[-i, :, None] for i in range(self.Model.State_dim)], axis=0)[:,:,0]
            # cor = self.COR(cor_vals, max_size=self.Model.State_dim)
            self.R = cor - np.dot(np.dot(self.C, self.X_error[i]), self.C.T)
            if np.any(np.diag(self.R)<0):
                if np.any(np.diag(self.R)>0):
                    v, w = np.linalg.eig(self.R)
                    if np.any(v<0):
                        self.R = np.dot(np.dot(w,np.abs(np.diag(v))), np.linalg.inv(w))
                else:
                    self.R *= -1.
            # self.R = np.abs(self.R)
            self.Q = np.dot(np.dot(np.dot(np.dot(self.G.T, self.K), cor), self.K.T), self.G)
            print("post", np.diag(self.Q), np.diag(self.R))
        # dif = np.asarray([np.dot(self.C, np.asarray(self.X.get(k, None)).T).T
        #                 - np.asarray([self.Measurements[k].PositionX,
        #                               self.Measurements[k].PositionY]) for k in self.Measurements.keys()])
        # self.R = np.cov(dif.T)
        # if np.any(np.isnan(self.R)) or np.any(np.linalg.eigvals(self.R) < np.diag(self.R_0)):
        #     self.R = self.R_0
        # print("R at %s"%self.R)
        return z, i

    # def optimize(self):
    #     n = len(self.A)
    #     M_0 =
    #     K0 = np.dot(np.dot(M_0, self.C), np.linalg.inv(np.dot(self.C, np.dot(M_0, self.C.T)) - self.R))
    #     K = np.dot(np.dot(p, self.C.transpose()),
    #                np.linalg.inv(np.dot(np.dot(self.C, p), self.C.transpose()) + self.R))
    #     AIKC = np.dot(self.A, np.diag(np.ones(n)) - np.dot(K, self.C))
    #     aikc = {0: np.diag(np.ones(n))}
    #     for i in range(2, n):
    #         aikc.update({i: np.dot(AIKC, aikc[i-1])})
    #     A = np.vstack([np.dot(self.C, np.dot(aikc[i], self.A)) for i in range(n)])
    #     A_cross = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
    #
    #     frames = set(self.X.keys()).intersection(self.Predicted_X.keys())
    #     deltas = np.asarray([self.X[f]-self.Predicted_X[f] for f in frames])
    #     C = np.vstack([c for c in self.COR(deltas, max_size=n)])
    #
    #     MH_T = np.dot(A_cross, C)
    #     # B = np.dot(np.vstack([np.dot(self.C, A[i]) for i in range(n)]), self.A)
    #     # B_cross = np.dot(np.linalg.inv(np.dot(B.T, B)), B.T)


    def is_observable(self):
        n = len(self.A)
        A = {0: np.diag(np.ones(n)), 1: self.A}
        for i in range(2, n):
            A.update({i: np.dot(self.A, A[i-1])})
        one = np.hstack([np.dot(self.C, A[i]).T for i in range(n)])
        two = np.hstack([np.dot(A[i], self.G) for i in range(n)])
        return (np.linalg.matrix_rank(one) == n) & (np.linalg.matrix_rank(two) == n)

    def is_optimal(self):
        frames = set(self.X.keys()).intersection(self.Predicted_X.keys())
        deltas = np.asarray([self.X[f]-self.Predicted_X[f] for f in frames])
        P = self.ACOR(deltas)
        N = len(frames)
        return np.all(np.sum([np.abs(np.diag(PP[:,:,0])) > (1.96/N**2) for PP in P[1:]], axis=0) < (0.05*N))

    def COR(self, vals, max_size=-1):
        vals = np.asarray(vals)
        if max_size < 0:
            return np.asarray([np.sum(vals[f:, None, :]*vals[f:, :, None], axis=0) for f in range(len(vals))])
        else:
            return np.asarray([np.sum(vals[f:, None, :]*vals[f:, :, None], axis=0) for f in range(max_size)])

    def ACOR(self, vals):
        c = self.COR(vals)
        cc = (np.diag(c[0,:,:,0])[None,:]*np.diag(c[0,:,:,0])[:,None])**0.5
        return c/cc



class ParticleFilter(Filter):
    """
    This Class describes a particle-filter in the pengu-track package.
    It calculates actual believed values from predictions and measurements.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    N: int
        The number of particles.
    Particles: dict
        The current particle state-vectors.
    Weights: dict
        Weights for every particle. Calculated from probability.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.
    """
    def __init__(self, model, n=100, meas_dist=ss.uniform(), state_dist=ss.uniform()):
        """
        This Class describes a particle-filter in the pengu-track package.
        It calculates actual believed values from predictions and measurements.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        n: int, optional
            The number of particles.
        meas_dist: scipy.stats.distributions object
            The distribution which describes measurement uncertainty.
        state_dist: scipy.stats.distributions object
            The distribution which describes state vector fluctuations.
        """
        super(ParticleFilter, self).__init__(model, state_dist=state_dist, meas_dist=meas_dist)
        self.N = n
        self.Particles = {}
        self.Weights = {}


    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding model. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """
        u, i = super(ParticleFilter, self).predict(u=u, i=i)

        for j in self.Particles.keys():
            mean = self.Model.predict(self.Particles[j], u)
            self.Particles.update({j: self.Model.evolute(self.State_Distribution.rvs()) + mean})

        self.Predicted_X.update({i: np.mean(self.Particles.values(), axis=0)})
        self.Predicted_X_error.update({i: np.std(self.Particles.values(), axis=0)})
        return u, i

    def update(self, z=None, i=None):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.

        Parameters
        ----------
        z: PenguTrack.Measurement, optional
            Recent measurement.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        z: PenguTrack.Measurement
            Recent measurement.
        i: int
            Recent/corresponding time-stamp.
        """
        z, i = super(ParticleFilter, self).update(z=z, i=i)
        measurement = copy.copy(z)
        z = np.asarray([z.PositionX, z.PositionY])
        for j in range(self.N):
            self.Weights.update({j: self.Measurement_Distribution.logpdf(z-self.Model.measure(self.Particles[j]))})
        weights = self.Weights.values()
        w_min = np.amin(weights)
        w_max = np.amax(weights)
        if w_max > w_min:
            weights = (weights - w_min)
            weights = np.cumsum(np.exp(weights))
            weights = weights/weights[-1]
            print("Standard")
        else:
            weights = np.cumsum(weights)
            weights = weights/weights[-1]
            print("Workaround")

        idx = np.sum(np.asarray(np.tile(weights, self.N).reshape((self.N, -1)) <
                     np.tile(np.random.rand(self.N), self.N).reshape((self.N, -1)).T, dtype=int), axis=1)
        print(len(set(idx)))
        values = self.Particles.values()
        for k, j in enumerate(idx):
            try:
                self.Particles.update({k: values[j]})
            except IndexError:
                self.Particles.update({k: values[j-1]})

        self.X.update({i: np.mean(self.Particles.values(), axis=0)})
        self.X_error.update({i: np.std(self.Particles.values(), axis=0)})

        return measurement, i




class MultiFilter(Filter):
    """
    This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
    sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
    is possible to assign possibilities to combinations of measurement and prediction.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Filter_Class: PenguTrack.Filter object
        A Type of Filter from which all subfilters should be built.
    Filters: dict
        Dictionary containing all sub-filters as PenguTrack.Filter objects.
    Active_Filters: dict
        Dictionary containing all sub-filters, which are currently updated.
    FilterThreshold: int
        Number of time steps, before a filter is set inactive.
    LogProbabilityThreshold: float
        Threshold, under which log-probabilities are concerned negligible.
    filter_args: list
        Filter-Type specific arguments from the Multi-Filter initialisation can be stored here.
    filter_kwargs: dict
        Filter-Type specific keyword-arguments from the Multi-Filter initialisation can be stored here.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.

    """
    def __init__(self, _filter, model, *args, **kwargs):
        """
        This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
        sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
        is possible to assign possibilities to combinations of measurement and prediction.

        Sub-filter specific arguments are handles by *args and **kwargs.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        meas_dist: scipy.stats.distributions object
            The distribution which describes measurement uncertainty.
        state_dist: scipy.stats.distributions object
            The distribution which describes state vector fluctuations.
        """
        super(MultiFilter, self).__init__(model)
        self.Filter_Class = _filter
        self.Filters = {}
        self.ActiveFilters = {}
        self.FilterThreshold = 3
        # TODO: rename to CostThreshold
        self.LogProbabilityThreshold = -np.inf#0.99#-200#-18.
        self.MeasurementProbabilityThreshold = 0.05#0.99#-200#-18.
        self.AssignmentProbabilityThreshold = 0.02#0.99#-200#-18.
        self.filter_args = args
        self.filter_kwargs = kwargs
        self.CriticalIndex = None
        self.Probability_Gain = {}
        self.Probability_Gain_Dicts = {}
        self.Probability_Assignment_Dicts = {}
        self.ProbUpdate = kwargs.get("prob_update", False)

    def predict(self, u=None, i=None):
        """
        Function to get predictions from the corresponding sub-filter models. Handles time-stamps and control-vectors.

        Parameters
        ----------
        u: array_like, optional
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp

        Returns
        ----------
        u: array_like
            Recent control-vector.
        i: int
            Recent/corresponding time-stamp.
        """

        assert not (u is None and i is None), "One of control vector or time stamp must be specified"

        # for j in list(self.ActiveFilters.keys()):
        #     _filter = self.ActiveFilters[j]
        #     if np.amax(list(_filter.Predicted_X.keys()))-np.amax(list(_filter.X.keys())) >= self.FilterThreshold:
        #         self.ActiveFilters.pop(j)
        #         print("Stopped track %s in frame %s with no updates for %s frames"%(j, i, self.FilterThreshold))

        predicted_x = {}
        predicted_x_error = {}
        for j in self.ActiveFilters.keys():
            self.ActiveFilters[j].predict(u=u, i=i)
            if i in self.ActiveFilters[j].Predicted_X.keys():
                predicted_x.update({j: self.ActiveFilters[j].Predicted_X[i]})
                predicted_x_error.update({j: self.ActiveFilters[j].Predicted_X_error[i]})

        self.Predicted_X.update({i: predicted_x})
        self.Predicted_X_error.update({i: predicted_x_error})
        return u, i

    def initial_update(self, z, i):
        """
        Function to get updates to the corresponding model in the first time step.
        Handles time-stamps and measurement-vectors.
        This function also handles the assignment of all incoming measurements to the active sub-filters.

        Parameters
        ----------
        z: list of PenguTrack.Measurement objects
            Recent measurements.
        i: int
            Recent/corresponding time-stamp.

        Returns
        ----------
        z: list of PenguTrack.Measurement objects
            Recent measurements.
        i: int
            Recent/corresponding time-stamp.
        """
        print("Initial Filter Update")

        if len(z)<1:
            raise ValueError("No Measurements found!")
        measurements = list(z)
        z = np.asarray([self.Model.vec_from_meas(m) for m in measurements], dtype=float)

        M = z.shape[0]

        for j in range(M):
            _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
            inferred_state = _filter.Model.infer_state(z[j])
            _filter.Predicted_X.update({i: inferred_state})
            _filter.X.update({i: inferred_state})
            _filter.Measurements.update({i: measurements[j]})

            try:
                J = max(self.Filters.keys()) + 1
            except ValueError:
                J = 0
            self.ActiveFilters.update({J: _filter})
            self.Filters.update({J: _filter})

    def update(self, z=None, i=None, big_jumps=False, verbose=False):
        """
        Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.
        This function also handles the assignment of all incoming measurements to the active sub-filters.

        Parameters
        ----------
        z: list of PenguTrack.Measurement objects
            Recent measurements.
        i: int
            Recent/corresponding time-stamp.

        Returns
        ----------
        z: list of PenguTrack.Measurement objects
            Recent measurements.
        i: int
            Recent/corresponding time-stamp.
        """
        assert not (z is None and i is None), "One of measurement vector or time stamp must be specified"

        # convert from pandas to measurement objects if necessary
        if not isinstance(z, np.ndarray) and isinstance(z, pandas.DataFrame):
            measurements = pandasDF_to_measurement(z)
            # self.Measurements.update({i: measurements})
        elif not isinstance(z, np.ndarray) and isinstance(z[0], Measurement):
            measurements = list(z)
            # self.Measurements.update({i:measurements})

            # meas_logp = np.asarray([m.Log_Probability for m in z])
        # if isinstance(z, tuple) and len(z)==2:
        #     z = z[0]
        #     z = np.asarray([self.Model.vec_from_meas(m) for m in measurements], ndmin=2)

        elif isinstance(z, np.ndarray):
            measurements = array_to_measurement(z, dim=self.Model.Meas_dim)
            # if len(z.shape) == 2:
            #     z = z[:, :, None]
            # meas_logp = np.ones(z.shape[0])
            # measurements = [Measurement(1, pos) for pos in z]
            # self.Measurements.update({i: measurements})
        else:
            raise ValueError("Input Positions are not of type array or pengutrack measurement!")

        # TODO: zjzibki7u
        self.Measurements.update({i: measurements})

        # meas_logp = measurements.Log_Probability
        meas_logp = np.asarray([m.Log_Probability for m in measurements])
        if verbose:
            print("Total number of detections: %d, valid detections: %d"%(len(meas_logp),(~np.isneginf(meas_logp)).sum()))

        # z = self.Model.vec_from_pandas(measurements)
        z = np.asarray([self.Model.vec_from_meas(m) for m in measurements])

        mask = ~np.isneginf(meas_logp)
        if not np.all(~mask):
            # meas_logp[~mask] = np.nanmin(meas_logp[mask])
            # mask &= (meas_logp - np.nanmin(meas_logp) >=
            #                (self.MeasurementProbabilityThreshold * (np.nanmax(meas_logp) - np.nanmin(meas_logp)))).astype(bool)
            z = z[mask]
            # measurements = list(np.asarray(measurements)[mask])
            measurements = np.asarray(measurements)[mask]
        else:
            # TODO: will ich das wirklich machen?
            self.Measurements.pop(i, None)
            return measurements, i

        M = z.shape[0]
        N = len(dict(self.ActiveFilters))

        # First frame or no active tracks: initialize tracker
        if N == 0 and M > 0:
            self.initial_update(measurements, i)
            return measurements, i

        # Dictionary to keep correspondence between track number and matrix entry
        gain_dict = dict(enumerate(self.ActiveFilters.keys()))
        # Start with neginf as default for prob gain (cost) matrix
        # probability_gain = np.ones((max(M, N), M)) * -np.inf
        probability_gain = np.ones((N, M)) * -np.inf

        # iter over tracks and corresponing matrix indices
        for j, k in gain_dict.items():
            # iter over measurements and corresponding indices
            for m, meas in enumerate(measurements):
                # built (cost)matrix entry for every entry
                probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i],
                                                                        measurements={i: meas},
                                                                        update=self.ProbUpdate)


        # If LogProbThreshold at neg inf set it to one under matrix minimum
        if self.LogProbabilityThreshold == -np.inf:
            LogProbabilityThreshold = np.nextafter(np.nanmin(probability_gain), -np.inf)
        else:
            LogProbabilityThreshold = self.LogProbabilityThreshold

        if not np.all(np.isneginf(probability_gain)):
            # set positive inf to max of array
            probability_gain[np.isposinf(probability_gain)] = np.amax(probability_gain[~np.isposinf(probability_gain)])

            # set negative inf to min of array
            probability_gain[np.isneginf(probability_gain)] = np.amin(probability_gain[~np.isneginf(probability_gain)])

            # now set nan results to negative inf
            probability_gain[np.isnan(probability_gain)] = -np.inf

            # matrix now contains no pos inf or nans.
            # neg inf masks the unusable values and threshold is over neg inf, but below lowest valid value

            # TODO: change name of logp to cost
            if self.filter_kwargs.get("no_dist"):
                cost_matrix = -np.exp(-probability_gain)
                threshold = -np.exp(-self.LogProbabilityThreshold)
            else:
                cost_matrix = self.cost_from_logprob(probability_gain)
                threshold = self.cost_from_logprob(probability_gain, value=self.LogProbabilityThreshold)

            rows, cols = self.assign(cost_matrix, threshold=threshold)
        else:
            rows, cols = [], []

        self.Probability_Gain.update({i: np.asarray(probability_gain)})

        track_length = dict(zip(gain_dict.values(), np.zeros(len(gain_dict), dtype=int)))
        track_length.update(dict([[k, len(self.ActiveFilters[k].X)] for k in self.ActiveFilters]))

        track_inactivity_time = dict(zip(gain_dict.values(), np.zeros(len(gain_dict), dtype=int)))
        track_inactivity_time.update(dict([[k, i-max(self.ActiveFilters[k].X)-1] for k in self.ActiveFilters]))

        # assignments = dict([(gain_dict[rows[i]], cols[i]) for i in range(len(rows))
        #                if (probability_gain[rows[i], cols[i]] > threshold) | (track_length[gain_dict[rows[i]]] < 2)])

        assignments = dict([(gain_dict[rows[i]], cols[i]) for i in range(len(rows))
                       if (cost_matrix[rows[i], cols[i]] < threshold)])

        not_updated_tracks = set(self.ActiveFilters.keys()).difference(assignments.keys())

        stopped_tracks = set([k for k in not_updated_tracks if track_inactivity_time[k] > self.FilterThreshold])

        not_updated_tracks = not_updated_tracks.difference(stopped_tracks)

        spawned_tracks = dict(zip(max(self.Filters.keys()) + 1 + np.arange(len(measurements)-len(assignments)),
                                  set(range(M)).difference(assignments.values())))

        print("Assigned %d, not updated %d, stopped %d and spawned %d Tracks."%(len(assignments),len(not_updated_tracks), len(stopped_tracks), len(spawned_tracks)))

        for k in assignments:
            m = assignments[k]
            self.ActiveFilters[k].update(z=measurements[m], i=i)
            if verbose:
                print("Updated track %s with prob %s in frame %s" % (k, probability_gain[reverse_dict(gain_dict)[k], m], i))

        for k in not_updated_tracks:
            if verbose:
                print("No update for track %s with best prob %s in frame %s"%(reverse_dict(gain_dict)[k],
                                                                              np.amax(probability_gain[reverse_dict(gain_dict)[k]]), i))

        for k in stopped_tracks:
            self.ActiveFilters.pop(k)
            if verbose:
                print("Stopped track %s with probability %s in frame %s" % (k, probability_gain[reverse_dict(gain_dict)[k], m], i))

        for k in spawned_tracks:
            m = spawned_tracks[k]
            _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
            _filter.Predicted_X.update({i: self.Model.infer_state(z[m])})
            _filter.X.update({i: self.Model.infer_state(z[m])})
            _filter.Measurements.update({i: measurements[m]})

            self.ActiveFilters.update({k: _filter})
            self.Filters.update({k: _filter})


        self.Probability_Gain_Dicts.update({i: gain_dict})
        self.Probability_Assignment_Dicts.update({i: dict([[k, c] for k,c in assignments.items()])})
        return measurements, i

    def fit(self, u, z):
        """Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde.

        Parameters
        ----------
        u: array_like
            List of control-vectors.
        z: list of PenguTrack.Measurement objects
            Recent measurements.
        """
        u = np.asarray(u)
        z = np.asarray(z)
        assert u.shape[0] == z.shape[0]

        for i in range(z.shape[0]):
            self.predict(u=u[i], i=i+1)
            self.update(z=z[i], i=i+1)

        return self.X, self.X_error, self.Predicted_X, self.Predicted_X_error

    def downfilter(self, t=None):
        """
        Function erases the timestep t from the class dictionaries (Measurement, X, Preditcion)

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """
        for k in self.Filters.keys():
            self.Filters[k].downfilter(t=t)

    def downdate(self, t=None):
        """
        Function erases the time-step t from the Measurements and Believes.

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """
        for k in self.Filters.keys():
            self.Filters[k].downdate(t=t)

    def unpredict(self, t=None):
        """
        Function erases the time-step t from the Believes, Predictions and Controls.

        Parameters
        ----------
        t: int, optional
            Time-steps for which filtering should be erased.
        """
        for k in self.Filters.keys():
            self.Filters[k].downdate(t=t)

    def log_prob(self, **kwargs):
        """
        Function to calculate the probability measure by predictions, measurements and corresponding distributions.

        Parameters
        ----------
        keys: list of int, optional
            Time-steps for which probability should be calculated.

        Returns
        ----------
        prob : float
            Probability of measurements at the given time-keys.
        """
        prob = 0
        for j in self.Filters.keys():
            prob += self.Filters[j].log_prob(**kwargs)
        return prob

    def assign(self, cost_matrix, **kwargs):
        return greedy_assignment(cost_matrix)

    def name(self):
        return str(self.__class__).split("'")[1].split(".")[-1] + "_" + str(self.Filter_Class).split("'")[1].split(".")[-1]

class Tracker(MultiFilter):
    pass


class HungarianTracker(Tracker):

    def assign(self, cost_matrix, **kwargs):
        return hungarian_assignment(cost_matrix)

class NetworkTracker(Tracker):
    """
    This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
    sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
    is possible to assign possibilities to combinations of measurement and prediction.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Filter_Class: PenguTrack.Filter object
        A Type of Filter from which all subfilters should be built.
    Filters: dict
        Dictionary containing all sub-filters as PenguTrack.Filter objects.
    Active_Filters: dict
        Dictionary containing all sub-filters, which are currently updated.
    FilterThreshold: int
        Number of time steps, before a filter is set inactive.
    LogProbabilityThreshold: float
        Threshold, under which log-probabilities are concerned negligible.
    filter_args: list
        Filter-Type specific arguments from the Multi-Filter initialisation can be stored here.
    filter_kwargs: dict
        Filter-Type specific keyword-arguments from the Multi-Filter initialisation can be stored here.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.
    Order: int
        The order of network assignments. The higher, the more costy and precise the assignment will be.

    """
    def __init__(self, *args, **kwargs):

        """
        This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
        sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
        is possible to assign possibilities to combinations of measurement and prediction.
        The network filter uses a network representation of the assignment problem and solves for a number of neighbours
        for each node, called order.

        Sub-filter specific arguments are handles by *args and **kwargs.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        meas_dist: scipy.stats.distributions object
            The distribution which describes measurement uncertainty.
        state_dist: scipy.stats.distributions object
            The distribution which describes state vector fluctuations.
        order: int, optioinal
            Order Parameter, number of next-neighbour layer to be taken into account for assignment
        """
        super(NetworkTracker, self).__init__(*args, **kwargs)
        self.Order = kwargs.get("order", 2)
    def assign(self, cost_matrix, threshold=None, method="linear", **kwargs):
        return network_assignment(cost_matrix, threshold=threshold
                                  , method="linear", order=self.Order)


class HybridSolver(Tracker):
    """
    This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
    sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
    is possible to assign possibilities to combinations of measurement and prediction.

    Attributes
    ----------
    Model: PenguTrack.model object
        A physical model to gain predictions from data.
    Filter_Class: PenguTrack.Filter object
        A Type of Filter from which all subfilters should be built.
    Filters: dict
        Dictionary containing all sub-filters as PenguTrack.Filter objects.
    Active_Filters: dict
        Dictionary containing all sub-filters, which are currently updated.
    FilterThreshold: int
        Number of time steps, before a filter is set inactive.
    LogProbabilityThreshold: float
        Threshold, under which log-probabilities are concerned negligible.
    filter_args: list
        Filter-Type specific arguments from the Multi-Filter initialisation can be stored here.
    filter_kwargs: dict
        Filter-Type specific keyword-arguments from the Multi-Filter initialisation can be stored here.
    Measurement_Distribution: scipy.stats.distributions object
        The distribution which describes measurement uncertainty.
    State_Distribution: scipy.stats.distributions object
        The distribution which describes state vector fluctuations.
    X: dict
        The time series of believes calculated for this filter. The keys equal the time stamp.
    X_error: dict
        The time series of errors on the corresponding believes. The keys equal the time stamp.
    Predicted_X: dict
        The time series of predictions made from the associated data. The keys equal the time stamp.
    Predicted_X_error: dict
        The time series of estimated prediction errors. The keys equal the time stamp.
    Measurements: dict
        The time series of measurements assigned to this filter. The keys equal the time stamp.
    Controls: dict
        The time series of control-vectors assigned to this filter. The keys equal the time stamp.

    """
    def __init__(self, *args, **kwargs):

        """
        This Class describes a filter, which is capable of assigning measurements to tracks, which again are represented by
        sub-filters. The type of these can be specified, as well as a physical model for predictions. With these objects it
        is possible to assign possibilities to combinations of measurement and prediction.

        Sub-filter specific arguments are handles by *args and **kwargs.

        Parameters
        ----------
        model: PenguTrack.model object
            A physical model to gain predictions from data.
        meas_dist: scipy.stats.distributions object
            The distribution which describes measurement uncertainty.
        state_dist: scipy.stats.distributions object
            The distribution which describes state vector fluctuations.
        dominance_threshold: float, optioinal
            Threshold between 0 and 1. definining which part of the assignments will be done by hungarian and greedy solver.
        """
        super(HybridSolver, self).__init__(*args, **kwargs)
        self.DominanceThreshold = kwargs.get("dominance_threshold")


# import multiprocessing
# class ThreadedMultiFilter(MultiFilter):
#     def __init__(self, *args, **kwargs):
#         super(ThreadedMultiFilter, self).__init__(*args, **kwargs)
#         self.Pool = multiprocessing.Pool(multiprocessing.cpu_count())
#         self.gain_dict = None
#         self.probability_gain = None
#
#     def update(self, z=None, i=None):
#         """
#         Function to get updates to the corresponding model. Handles time-stamps and measurement-vectors.
#         This function also handles the assignment of all incoming measurements to the active sub-filters.
#
#         Parameters
#         ----------
#         z: list of PenguTrack.Measurement objects
#             Recent measurements.
#         i: int
#             Recent/corresponding time-stamp.
#
#         Returns
#         ----------
#         z: list of PenguTrack.Measurement objects
#             Recent measurements.
#         i: int
#             Recent/corresponding time-stamp.
#         """
#         measurements = list(z)
#
#         meas_logp = np.asarray([m.Log_Probability for m in z])
#         try:
#             z = np.asarray([np.asarray([m.PositionX, m.PositionY, m.PositionZ]) for m in z], ndmin=2)
#         except (ValueError, AttributeError):
#             try:
#                 z = np.asarray([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
#             except (ValueError, AttributeError):
#                 z = np.asarray([np.asarray([m.PositionX]) for m in z], ndmin=2)
#         # print(np.mean(meas_logp), np.amin(meas_logp), np.amax(meas_logp))
#         print(len(z))
#         z = z[meas_logp - np.amin(meas_logp) >= (self.MeasurementProbabilityThreshold * (np.amax(meas_logp)-np.amin(meas_logp)))]
#         measurements = list(np.asarray(measurements)[meas_logp - np.amin(meas_logp) >= (self.MeasurementProbabilityThreshold *
#                                                                   (np.amax(meas_logp)-np.amin(meas_logp)))])
#         print(len(z))
#         M = z.shape[0]
#         N = len(self.ActiveFilters.keys())
#
#         if N == 0 and M > 0:
#             self.initial_update(measurements, i)
#             return measurements, i
#
#         self.gain_dict = {}
#         self.probability_gain = np.ones((max(M, N), M))/0.#*self.LogProbabilityThreshold
#         gain_dict = self.gain_dict
#         probability_gain = self.probability_gain
#
#         pool_jobs=[]
#         for j, k in enumerate(self.ActiveFilters.keys()):
#             self.gain_dict.update({j: k})
#             # for m, meas in enumerate(measurements):
#             #     probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i], measurements={i: meas})
#             for m in range(len(measurements)):
#                 pool_jobs.append([i, self.ActiveFilters[k], measurements[m], j, m])
#
#         for j, m, value in self.Pool.map(ThreadedUpdate, pool_jobs):
#             self.probability_gain[j,m] = value
#
#         probability_gain[np.isinf(probability_gain)] = np.nan
#         probability_gain[np.isnan(probability_gain)] = np.nanmin(probability_gain)
#         if self.LogProbabilityThreshold == -np.inf:
#             LogProbabilityThreshold = np.nanmin(probability_gain)
#         else:
#             LogProbabilityThreshold = self.LogProbabilityThreshold
#
#         # print(np.mean(probability_gain), np.amin(probability_gain), np.amax(probability_gain))
#         # print(probability_gain)
#         # print(np.amin(probability_gain),np.nanmax(probability_gain), np.mean(probability_gain))
#         # norm = np.linalg.det(np.exp(probability_gain))#np.sum(np.exp(probability_gain), axis=1)
#         # print(norm)
#         # probability_gain = probability_gain-norm #(probability_gain.T - np.log(norm)).T
#         # print(probability_gain)
#         self.Probability_Gain.update({i: np.asarray(probability_gain)})
#         # self.CriticalIndex = gain_dict[np.nanargmax([np.sort(a)[-2]/np.sort(a)[-1] for a in probability_gain[:N]])]
#         x = {}
#         x_err = {}
#         from scipy.optimize import linear_sum_assignment
#         rows, cols = linear_sum_assignment(-1*probability_gain)
#
#         for j, k in enumerate(rows):
#             # if not np.all(np.isnan(probability_gain)+np.isinf(probability_gain)):
#             #     k, m = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
#             # else:
#             #     k, m = np.unravel_index(np.nanargmin(probability_gain), probability_gain.shape)
#             k = rows[j]
#             m = cols[j]
#
#             if N > M and m >= M:
#                 continue
#
#             # print(np.amin(probability_gain))
#             # if (probability_gain[k, m] - np.amin(probability_gain) >=
#             #     (self.LogProbabilityThreshold *(np.amax(probability_gain)-np.amin(probability_gain)))
#             #     and gain_dict.has_key(k)):
#             if probability_gain[k,m] > LogProbabilityThreshold and k in gain_dict:
#             # if probability_gain[k, m] - MIN > LIMIT and gain_dict.has_key(k):
#                 self.ActiveFilters[gain_dict[k]].update(z=measurements[m], i=i)
#                 x.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X[i]})
#                 x_err.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X_error[i]})
#
#             else:
#                 print("DEPRECATED TRACK WITH PROB %s IN FRAME %s" % (probability_gain[k, m], i))
#                 try:
#                     n = len(self.ActiveFilters[gain_dict[k]].X.keys())
#                 except KeyError:
#                     n = np.inf
#
#                 l = max(self.Filters.keys()) + 1
#                 _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
#                 _filter.Predicted_X.update({i: self.Model.infer_state(z[m])})
#                 _filter.X.update({i: self.Model.infer_state(z[m])})
#                 _filter.Measurements.update({i: measurements[m]})
#
#                 self.ActiveFilters.update({l: _filter})
#                 self.Filters.update({l: _filter})
#
#             probability_gain[k, :] = np.nan
#             probability_gain[:, m] = np.nan
#
#         # if len(self.ActiveFilters.keys()) < M:
#         #     raise RuntimeError('Lost Filters on the way. This should never happen')
#         return measurements, i


def ThreadedUpdate(arg):
    i, filter, measurement, j, m = arg
    return j, m, filter.log_prob(keys=[i], measurements={i: measurement})




# class AdvancedMultiFilter(MultiFilter):
#     def __init__(self, *args, **kwargs):
#         super(AdvancedMultiFilter, self).__init__(*args,**kwargs)
#     def _critical_(self, ProbMat):
#         critical_i =[]
#         for i,n in enumerate(ProbMat):
