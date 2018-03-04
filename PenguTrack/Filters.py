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
import scipy.integrate as integrate
#  import scipy.optimize as opt


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
        self.Model = model
        self.Measurement_Distribution = meas_dist
        self.State_Distribution = state_dist
        
        self.X = {}
        self.X_error = {}
        self.Predicted_X = {}
        self.Predicted_X_error = {}
        
        self.Measurements = {}
        self.Controls = {}

        self.NoDist = no_dist
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
                z = np.array(z).flatten()
                assert self.Model.Meas_dim == len(z),\
                    "Measurement input shape %s is not equal to model measurement dimension %s"%(
                        len(z),self.Model.Meas_dim)
                z = Measurement(1.0, position=z)
            self.Measurements.update({i: z})
        measurement = copy.copy(z)
        # simplest possible update
        # try:
        #     self.X.update({i: np.asarray([z.PositionX, z.PositionY, z.PositionZ])})
        # except(ValueError, AttributeError):
        #     try:
        #         self.X.update({i: np.asarray([z.PositionX, z.PositionY])})
        #     except(ValueError, AttributeError):
        #         self.X.update({i: np.asarray([z.PositionX])})
        #
        if len(self.Model.Extensions) > 0:
            z = np.array(np.vstack([np.array([measurement[v] for v in self.Model.Measured_Variables]),
                            np.array([[measurement.Data[var][0]] for var in self.Model.Extensions])]))
        else:
            z = np.array([measurement[v] for v in self.Model.Measured_Variables])

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
    
    # def log_prob(self, keys=None, measurements=None, compare_bel=True):
    #     """
    #     Function to calculate the probability measure by predictions, measurements and corresponding distributions.
    #
    #     Parameters
    #     ----------
    #     keys: list of int, optional
    #         Time-steps for which probability should be calculated.
    #     measurements: dict, optional
    #         List of PenguTrack.Measurement objects for which probability should be calculated.
    #     compare_bel: bool, optional
    #         If True, it will be tried to compare the believed state with the measurement. If False or
    #         there is no believe-value, the prediction will be taken.
    #
    #     Returns
    #     ----------
    #     probs : float
    #         Probability of measurements at the given time-keys.
    #     """
    #     probs = 0
    #     if keys is None:
    #         keys = self.Measurements.keys()
    #
    #     if measurements is None:
    #         for i in keys:
    #             # Generate Value for comparison with measurement
    #             try:
    #                 if compare_bel:
    #                     comparison = self.X[i]
    #                 else:
    #                     raise KeyError
    #             except KeyError:
    #                 try:
    #                     comparison = self.Predicted_X[i]
    #                 except KeyError:
    #                     self.predict(i=i)
    #                     comparison = self.Predicted_X[i]
    #             try:
    #                 position = np.asarray([self.Measurements[i].PositionX,
    #                                        self.Measurements[i].PositionY,
    #                                        self.Measurements[i].PositionZ])
    #             except (ValueError, AttributeError):
    #                 try:
    #                     position = np.asarray([self.Measurements[i].PositionX,
    #                                            self.Measurements[i].PositionY])
    #                 except (ValueError, AttributeError):
    #                     position = np.asarray([self.Measurements[i].PositionX])
    #
    #             # def integrand(*args):
    #             #     x = np.array(args)
    #             #     return self.State_Distribution.pdf(x-comparison)*self.Measurement_Distribution.pdf(self.Model.measure(x)-position)
    #             #
    #             # integral = integrate.nquad(integrand,
    #             #                            np.array([-1*np.ones_like(self.Model.State_dim)*100,
    #             #                                      np.ones(self.Model.State_dim)*100]).T)
    #             # print(integral)
    #
    #             try:
    #                 # probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(position
    #                 #                                                              - self.Model.measure(comparison))))
    #                 probs += self.Measurement_Distribution.logpdf(position - self.Model.measure(comparison))
    #
    #                 # print("----------")
    #                 # print(self.Measurement_Distribution.logpdf(position - self.Model.measure(comparison)))
    #                 # print(np.log(self.Measurement_Distribution.pdf(position - self.Model.measure(comparison))))
    #                 # print("----------")
    #                 probs += self.Measurements[i].Log_Probability
    #             except ValueError:
    #                 print(position.shape, position)
    #                 print(comparison.shape, comparison)
    #                 print(self.Model.measure(comparison).shape, self.Model.measure(comparison))
    #                 raise
    #     else:
    #         for i in keys:
    #             # Generate Value for comparison with measurement
    #             try:
    #                 if compare_bel:
    #                     comparison = self.X[i]
    #                 else:
    #                     raise KeyError
    #             except KeyError:
    #                 try:
    #                     comparison = self.Predicted_X[i]
    #                 except KeyError:
    #                     self.predict(i=i)
    #                     comparison = self.Predicted_X[i]
    #             try:
    #                 position = np.asarray([measurements[i].PositionX,
    #                                        measurements[i].PositionY,
    #                                        measurements[i].PositionZ])
    #             except (ValueError, AttributeError):
    #                 try:
    #                     position = np.asarray([measurements[i].PositionX,
    #                                            measurements[i].PositionY])
    #                 except (ValueError, AttributeError):
    #                     position = np.asarray([measurements[i].PositionX])
    #             # try:
    #                 # probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(position
    #                 #                                                              - self.Model.measure(comparison))))
    #             probs += self.Measurement_Distribution.logpdf(position - self.Model.measure(comparison))
    #             # print("----------")
    #             # print(self.Measurement_Distribution.logpdf(position - self.Model.measure(comparison)))
    #             # print(np.log(self.Measurement_Distribution.pdf(position - self.Model.measure(comparison))))
    #             # print("----------")
    #             probs += measurements[i].Log_Probability
    #             # except RuntimeWarning:
    #             #     probs = -np.inf
    #     return probs



    def log_prob(self, keys=None, measurements=None, update=True):
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
            # if self.X.has_key(k) and self.Predicted_X.has_key(k):
                prob += self._log_prob_(k)
            elif k in self.X:
            # elif self.X.has_key(k):
                self.predict(i=k)
                prob += self._log_prob_(k)
            elif k in self.Predicted_X and k in measurements:
            # elif self.Predicted_X.has_key(k) and measurements.has_key(k):
                if update:
                    self.update(z=measurements[k],i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k, measurements[k])
            elif k in self.Predicted_X and k in self.Measurements:
            # elif self.Predicted_X.has_key(k) and self.Measurements.has_key(k):
                if update:
                    self.update(z=self.Measurements[k],i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k)
            elif k in measurements:
            # elif measurements.has_key(k):
                self.predict(i=k)
                pending_downpredicts.append(k)
                if update:
                    self.update(z=measurements[k], i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k, measurements[k])
            elif k in self.Measurements:
            # elif self.Measurements.has_key(k):
                self.predict(i=k)
                pending_downpredicts.append(k)
                if update:
                    self.update(z=self.Measurements[k], i=k)
                    pending_downdates.append(k)
                    prob += self._log_prob_(k)
                else:
                    prob += self._meas_log_prob(k)
            else:
                raise ValueError("Probability for key %s could not be computed!"%k)

        for k in pending_downdates:
            self.downdate(k)
        for k in pending_downpredicts:
            self.unpredict(k)
        return prob

    def _log_prob_(self, key):
        return self._state_log_prob_(key)

    def _state_log_prob_(self, key):
        if self.NoDist:
            return -np.linalg.norm(self.X[key]-self.Predicted_X[key])
        return self.State_Distribution.logpdf((self.X[key]-self.Predicted_X[key]).T)

    def _meas_log_prob(self, key, measurement=None):
        if measurement is None:
            measurement=self.Measurements[key]
        if self.NoDist:
            return -np.linalg.norm(self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key]))
        return self.Measurement_Distribution.logpdf((self.Model.vec_from_meas(measurement)-self.Model.measure(self.Predicted_X[key])).T)
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

        return np.array(self.X.values(), dtype=float),\
               np.array(self.X_error.values(), dtype=float),\
               np.array(self.Predicted_X.values(), dtype=float),\
               np.array(self.Predicted_X_error.values(), dtype=float)

    def cost_from_logprob(self, log_prob):
        return cost_from_logprob(log_prob)


class KalmanFilter(Filter):
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
    A: np.array
        State-Transition-Matrix, describing the evolution of the state without fluctuation.
        Received from the physical Model.
    B: np.array
        State-Control-Matrix, describing the influence of external-control on the states.
        Received from the physical Model.
    C: np.array
        Measurement-Matrix , describing the projection of the measurement-vector from the state-vector.
        Received from the physical Model.
    G: np.array
        Evolution-Matrix, describing the evolution of state vectors by fluctuations from the state-distribution.
        Received from the physical Model.
    Q: np.array
        Covariance matrix (time-evolving) for the state-distribution.
    Q_0: np.array
        Covariance matrix (initial state) for the state-distribution.
    R: np.array
        Covariance matrix (time-evolving) for the measurement-distribution.
    R_0: np.array
        Covariance matrix (initial state) for the measurement-distribution.
    P_0: np.array
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

        evolution_variance = np.array(evolution_variance, dtype=float)
        if evolution_variance.shape != (int(self.Model.Evolution_dim),):
            evolution_variance = np.ones(self.Model.Evolution_dim) * np.mean(evolution_variance)

        self.Evolution_Variance = evolution_variance
        self.Q = np.diag(evolution_variance)
        self.Q_0 = np.diag(evolution_variance)

        measurement_variance = np.array(measurement_variance, dtype=float)
        if measurement_variance.shape != (int(self.Model.Meas_dim),):
            measurement_variance = np.ones(self.Model.Meas_dim) * np.mean(measurement_variance)

        self.Measurement_Variance = measurement_variance
        self.R = np.diag(measurement_variance)
        self.R_0 = np.diag(measurement_variance)

        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

        p = np.dot(np.dot(self.C.T, self.R_0), self.C) + np.dot(np.dot(self.G, self.Q), self.G.T)#np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.P_0 = p

        kwargs.update(dict(meas_dist=ss.multivariate_normal(cov=self.R),
                           state_dist=ss.multivariate_normal(cov=self.P_0)))
        super(KalmanFilter, self).__init__(model, **kwargs)

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
        u, i = super(KalmanFilter, self).predict(u=u, i=i)

        try:
            x_ = np.dot(self.A, self.X[i-1]) + np.dot(self.B, u)
        except KeyError:
            x_ = np.dot(self.A, self.Predicted_X[i-1]) + np.dot(self.B, u)

        try:
            p_ = np.dot(np.dot(self.A, self.X_error[i-1]), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)
        except KeyError:
            try:
                p_ = np.dot(np.dot(self.A, self.Predicted_X_error[i-1]), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)
            except KeyError:
                p_ = np.dot(np.dot(self.A, self.P_0), self.A.T) + np.dot(np.dot(self.G, self.Q), self.G.T)

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
        measurement = copy.copy(z)
        # try:
        #     z = np.asarray([z.PositionX, z.PositionY, z.PositionZ])
        # except (ValueError, AttributeError):
        #     try:
        #         z = np.asarray([z.PositionX, z.PositionY])
        #     except ValueError:
        #         z = np.asarray([z.PositionX])
        if len(self.Model.Extensions) > 0:
            z = np.array(np.vstack([np.array([measurement[v] for v in self.Model.Measured_Variables]),
                            np.array([[measurement.Data[var][0]] for var in self.Model.Extensions])]))
        else:
            z = np.array([measurement[v] for v in self.Model.Measured_Variables])


        # if len(self.Model.Extensions) > 0:
        #     z = np.vstack((z, [[measurement.Data[var][0]] for var in self.Model.Extensions]))

        try:
            x = self.Predicted_X[i]
        except KeyError:
            u, i = self.predict(i=i)
            x = self.Predicted_X[i]

        try:
            p = self.Predicted_X_error[i]
        except KeyError:
            try:
                u, i = self.predict(i=i)
                p = self.Predicted_X_error[i]
            except KeyError:
                p = self.P_0

        try:
            k = np.dot(np.dot(p, self.C.transpose()),
                       np.linalg.inv(np.dot(np.dot(self.C, p), self.C.transpose()) + self.R))
        except np.linalg.LinAlgError:
            e = sys.exc_info()[1]
            print(p)
            raise np.linalg.LinAlgError(e)

        y = z - np.dot(self.C, x)

        x_ = x + np.dot(k, y)
        p_ = p - np.dot(np.dot(k, self.C), p)

        self.X.update({i: x_})
        self.X_error.update({i: p_})

        try:
            self.Measurement_Distribution = ss.multivariate_normal(cov=self.R)
        except np.linalg.LinAlgError:
            self.Measurement_Distribution = ss.multivariate_normal(cov=self.R_0)

        try:
            self.State_Distribution = ss.multivariate_normal(cov=self.X_error[i])
        except np.linalg.LinAlgError:
            self.State_Distribution = ss.multivariate_normal(cov=self.P_0)

        return measurement, i

    def _log_prob_(self, key):
        current_cov = np.copy(self.State_Distribution.cov)
        try:
            self.State_Distribution.cov = self.X_error[key]
        except KeyError:
            self.State_Distribution.cov = self.P_0
        value = super(KalmanFilter, self)._log_prob_(key)
        self.State_Distribution.cov = current_cov
        return value

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
    A: np.array
        State-Transition-Matrix, describing the evolution of the state without fluctuation.
        Received from the physical Model.
    B: np.array
        State-Control-Matrix, describing the influence of external-control on the states.
        Received from the physical Model.
    C: np.array
        Measurement-Matrix , describing the projection of the measurement-vector from the state-vector.
        Received from the physical Model.
    G: np.array
        Evolution-Matrix, describing the evolution of state vectors by fluctuations from the state-distribution.
        Received from the physical Model.
    Q: np.array
        Covariance matrix (time-evolving) for the state-distribution.
    Q_0: np.array
        Covariance matrix (initial state) for the state-distribution.
    R: np.array
        Covariance matrix (time-evolving) for the measurement-distribution.
    R_0: np.array
        Covariance matrix (initial state) for the measurement-distribution.
    P_0: np.array
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

    def predict(self, *args, **kwargs):
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
        if self.Lag == 1 or (-1 * self.Lag > len(self.Predicted_X.keys())):
            lag = 0
        else:
            lag = self.Lag
        self.Q = np.dot(np.dot(self.G.T, np.cov(np.asarray(self.Predicted_X.values()[lag:]).T)), self.G)
        print("Q at %s"%self.Q)
        if np.any(np.isnan(self.Q)):# or np.any(np.linalg.eigvals(self.Q) < np.diag(self.Q_0)):
            self.Q = self.Q_0
        # print("Q at %s"%self.Q)
        return super(AdvancedKalmanFilter, self).predict(*args, **kwargs)

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
        dif = np.array([np.dot(self.C, np.array(self.X.get(k, None)).T).T
                        - np.asarray([self.Measurements[k].PositionX,
                                      self.Measurements[k].PositionY]) for k in self.Measurements.keys()])
        self.R = np.cov(dif.T)
        if np.any(np.isnan(self.R)) or np.any(np.linalg.eigvals(self.R) < np.diag(self.R_0)):
            self.R = self.R_0
        print("R at %s"%self.R)
        return super(AdvancedKalmanFilter, self).update(*args, **kwargs)


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

        idx = np.sum(np.array(np.tile(weights, self.N).reshape((self.N, -1)) <
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
        self.LogProbabilityThreshold = -np.inf#0.99#-200#-18.
        self.MeasurementProbabilityThreshold = 0.05#0.99#-200#-18.
        self.AssignmentProbabilityThreshold = 0.02#0.99#-200#-18.
        self.filter_args = args
        self.filter_kwargs = kwargs
        self.CriticalIndex = None
        self.Probability_Gain = {}
        self.Probability_Gain_Dicts = {}
        self.Probability_Assignment_Dicts = {}
        self.ProbUpdate = kwargs.get("prob_update", True)

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

        for j in list(self.ActiveFilters.keys()):
            _filter = self.ActiveFilters[j]
            if np.amax(list(_filter.Predicted_X.keys()))-np.amax(list(_filter.X.keys())) >= self.FilterThreshold:
                self.ActiveFilters.pop(j)
                print("Stoped track %s in frame %s with no updates for %s frames"%(j, i, self.FilterThreshold))

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
        try:
            if len(self.Model.Extensions) > 0:
                z = np.array(
                    [np.vstack((np.array([m.PositionX, m.PositionY, m.PositionZ]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                              for m in measurements], ndmin=2)
            else:
                z = np.array([np.asarray([m.PositionX, m.PositionY, m.PositionZ]) for m in z], ndmin=2)
        except (ValueError, AttributeError):
            try:
                if len(self.Model.Extensions) > 0:
                    z = np.array(
                        [np.vstack((np.array([m.PositionX, m.PositionY]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                         for m in measurements], ndmin=2)
                else:
                    z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
            except (ValueError, AttributeError):
                if len(self.Model.Extensions) > 0:
                    z = np.array(
                        [np.vstack((np.array([m.PositionX]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                         for m in measurements], ndmin=2)
                else:
                    z = np.array([np.asarray([m.PositionX]) for m in z], ndmin=2)

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
        measurements = list(z)
        self.Measurements.update({i:z})

        meas_logp = np.array([m.Log_Probability for m in z])
        # try:
        #     z = np.array([np.asarray([m.PositionX, m.PositionY, m.PositionZ]) for m in z], ndmin=2)
        # except (ValueError, AttributeError):
        #     try:
        #         z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
        #     except (ValueError, AttributeError):
        #         z = np.array([np.asarray([m.PositionX]) for m in z], ndmin=2)
        try:
            if len(self.Model.Extensions) > 0:
                z = np.array(
                    [np.vstack((np.array([m.PositionX, m.PositionY, m.PositionZ]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                              for m in measurements], ndmin=2)
            else:
                z = np.array([np.asarray([m.PositionX, m.PositionY, m.PositionZ]) for m in z], ndmin=2)
        except (ValueError, AttributeError):
            try:
                if len(self.Model.Extensions) > 0:
                    z = np.array(
                        [np.vstack((np.array([m.PositionX, m.PositionY]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                         for m in measurements], ndmin=2)
                else:
                    z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
            except (ValueError, AttributeError):
                if len(self.Model.Extensions) > 0:
                    z = np.array(
                        [np.vstack((np.array([m.PositionX]), np.array([[m.Data[var][0]] for var in self.Model.Extensions])[None, :]))
                         for m in measurements], ndmin=2)
                else:
                    z = np.array([np.asarray([m.PositionX]) for m in z], ndmin=2)

        mask = ~np.isneginf(meas_logp)
        if not np.all(~mask):
            meas_logp[~mask] = np.nanmin(meas_logp[mask])
            mask &= (meas_logp - np.nanmin(meas_logp) >=
                           (self.MeasurementProbabilityThreshold * (np.nanmax(meas_logp) - np.nanmin(meas_logp)))).astype(bool)
            z = z[mask]
            measurements = list(np.asarray(measurements)[mask])
        else:
            self.Measurements.pop(i, None)
            return measurements, i

        M = z.shape[0]
        N = len(dict(self.ActiveFilters))

        if N == 0 and M > 0:
            self.initial_update(measurements, i)
            return measurements, i

        gain_dict = []
        probability_gain = np.ones((max(M, N), M)) * -np.inf


        filter_keys = list(self.ActiveFilters.keys())
        for j, k in enumerate(filter_keys):
            gain_dict.append([j, k])
            for m, meas in enumerate(measurements):
                probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i], measurements={i: meas},
                                                                        update=self.ProbUpdate)
        gain_dict = dict(gain_dict)

        probability_gain[np.isinf(probability_gain)] = np.nan
        if np.all(np.isnan(probability_gain)):
            probability_gain[:] = -np.inf
        else:
            probability_gain[np.isnan(probability_gain)] = np.nanmin(probability_gain)
        if self.LogProbabilityThreshold == -np.inf:
            LogProbabilityThreshold = np.nanmin(probability_gain)
        else:
            LogProbabilityThreshold = self.LogProbabilityThreshold


        self.Probability_Gain.update({i: np.array(probability_gain)})

        x = {}
        x_err = {}
        from scipy.optimize import linear_sum_assignment

        cost_matrix=self.cost_from_logprob(probability_gain)

        rows, cols = linear_sum_assignment(cost_matrix)

        if verbose:
            for t in range(N):
                if t not in rows:
                        print("No update for track %s with best prob %s in frame %s"%(gain_dict[t], np.amax(probability_gain[t]), i))

        for j, k in enumerate(rows):
            k = rows[j]
            m = cols[j]

            if N > M and m >= M:
                continue

            if k in gain_dict and (
                            probability_gain[k, m] > LogProbabilityThreshold or
                            (len(self.ActiveFilters[gain_dict[k]].X) < 2 and
                                     min([i-o for o in self.ActiveFilters[gain_dict[k]].X.keys() if i>o]) < 2 and big_jumps)) :
                self.ActiveFilters[gain_dict[k]].update(z=measurements[m], i=i)
                x.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X[i]})
                x_err.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X_error[i]})
                if verbose:
                    print("Updated track %s with prob %s in frame %s" % (gain_dict[k], probability_gain[k, m], i))

            else:
                if verbose:
                    if k in gain_dict:
                        print("DEPRECATED TRACK %s WITH PROB %s IN FRAME %s" % (gain_dict[k], probability_gain[k, m], i))
                    else:
                        print("Started track with prob %s in frame %s" % (probability_gain[k, m], i))
                try:
                    n = len(self.ActiveFilters[gain_dict[k]].X.keys())
                except KeyError:
                    n = np.inf

                l = max(self.Filters.keys()) + 1
                _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
                _filter.Predicted_X.update({i: self.Model.infer_state(z[m])})
                _filter.X.update({i: self.Model.infer_state(z[m])})
                _filter.Measurements.update({i: measurements[m]})

                self.ActiveFilters.update({l: _filter})
                self.Filters.update({l: _filter})
                gain_dict.update({k:l})

            probability_gain[k, :] = np.nan
            probability_gain[:, m] = np.nan


        self.Probability_Gain_Dicts.update({i: gain_dict})
        self.Probability_Assignment_Dicts.update({i: dict([[gain_dict[r], c] for r,c in zip(rows, cols)])})
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
        u = np.array(u)
        z = np.array(z)
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


class HungarianTracker(MultiFilter):

    # def __init__(self, _filter, model, *args, **kwargs):
    #     super(HungarianTracker, self).__init__(_filter, model, *args, **kwargs)
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

        measurements = list(z)
        if len(self.Model.Extensions) > 0:
            z = np.array(
                [np.vstack((np.array([m[v] for v in self.Model.Measured_Variables]),
                            np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                 for m in z], ndmin=2)
        else:
            z = np.array([[m[v] for v in self.Model.Measured_Variables] for m in z], ndmin=2)

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

    def update(self, z=None, i=None, big_jumps=False, verbose=True):
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
        measurements = list(z)
        self.Measurements.update({i: z})

        meas_logp = np.array([m.Log_Probability for m in z])

        if len(self.Model.Extensions) > 0:
            z = np.array(
                [np.vstack((np.array([m[v] for v in self.Model.Measured_Variables]),
                            np.array([[m.Data[var][0]] for var in self.Model.Extensions])))
                 for m in measurements], ndmin=2)
        else:
            z = np.array([[m[v] for v in self.Model.Measured_Variables] for m in measurements], ndmin=2)

        mask = ~np.isneginf(meas_logp)
        if not np.all(~mask):
            meas_logp[~mask] = np.nanmin(meas_logp[mask])
            mask &= (meas_logp - np.nanmin(meas_logp) >=
                     (self.MeasurementProbabilityThreshold * (np.nanmax(meas_logp) - np.nanmin(meas_logp)))).astype(
                bool)
            z = z[mask]
            measurements = list(np.asarray(measurements)[mask])
        else:
            self.Measurements.pop(i, None)
            return measurements, i

        M = z.shape[0]
        N = len(dict(self.ActiveFilters))

        if N == 0 and M > 0:
            self.initial_update(measurements, i)
            return measurements, i

        gain_dict = []
        probability_gain = np.ones((max(M, N), M)) * -np.inf

        filter_keys = list(self.ActiveFilters.keys())
        for j, k in enumerate(filter_keys):
            gain_dict.append([j, k])
            for m, meas in enumerate(measurements):
                probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i], measurements={i: meas},
                                                                        update=self.ProbUpdate)
        gain_dict = dict(gain_dict)

        probability_gain[np.isinf(probability_gain)] = np.nan
        if np.all(np.isnan(probability_gain)):
            probability_gain[:] = -np.inf
        else:
            probability_gain[np.isnan(probability_gain)] = np.nanmin(probability_gain)
        if self.LogProbabilityThreshold == -np.inf:
            LogProbabilityThreshold = np.nanmin(probability_gain)
        else:
            LogProbabilityThreshold = self.LogProbabilityThreshold

        self.Probability_Gain.update({i: np.array(probability_gain)})

        x = {}
        x_err = {}
        from scipy.optimize import linear_sum_assignment

        cost_matrix = self.cost_from_logprob(probability_gain)

        rows, cols = linear_sum_assignment(cost_matrix)

        for t in range(N):
            if t not in rows:
                if verbose:
                    print("No update for track %s with best prob %s in frame %s" % (
                gain_dict[t], np.amax(probability_gain[t]), i))

        for j, k in enumerate(rows):
            k = rows[j]
            m = cols[j]

            if N > M and m >= M:
                continue

            if k in gain_dict and (
                            probability_gain[k, m] > LogProbabilityThreshold or
                        (len(self.ActiveFilters[gain_dict[k]].X) < 2 and
                                 min([i - o for o in self.ActiveFilters[gain_dict[k]].X.keys() if
                                      i > o]) < 2 and big_jumps)):
                self.ActiveFilters[gain_dict[k]].update(z=measurements[m], i=i)
                x.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X[i]})
                x_err.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X_error[i]})
                if verbose:
                    print("Updated track %s with prob %s in frame %s" % (gain_dict[k], probability_gain[k, m], i))

            else:
                if verbose:
                    if k in gain_dict:
                        print("DEPRECATED TRACK %s WITH PROB %s IN FRAME %s" % (gain_dict[k], probability_gain[k, m], i))
                    else:
                        print("Started track with prob %s in frame %s" % (probability_gain[k, m], i))
                try:
                    n = len(self.ActiveFilters[gain_dict[k]].X.keys())
                except KeyError:
                    n = np.inf

                l = max(self.Filters.keys()) + 1
                _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
                _filter.Predicted_X.update({i: self.Model.infer_state(z[m])})
                _filter.X.update({i: self.Model.infer_state(z[m])})
                _filter.Measurements.update({i: measurements[m]})

                self.ActiveFilters.update({l: _filter})
                self.Filters.update({l: _filter})

                gain_dict.update({k:l})

            probability_gain[k, :] = np.nan
            probability_gain[:, m] = np.nan

        self.Probability_Gain_Dicts.update({i: gain_dict})
        self.Probability_Assignment_Dicts.update({i: dict([[gain_dict[r], c] for r,c in zip(rows, cols)])})
        return measurements, i


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
#         meas_logp = np.array([m.Log_Probability for m in z])
#         try:
#             z = np.array([np.asarray([m.PositionX, m.PositionY, m.PositionZ]) for m in z], ndmin=2)
#         except (ValueError, AttributeError):
#             try:
#                 z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
#             except (ValueError, AttributeError):
#                 z = np.array([np.asarray([m.PositionX]) for m in z], ndmin=2)
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
#         self.Probability_Gain.update({i: np.array(probability_gain)})
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


def cost_from_logprob(logprob):
    cost_matrix = np.copy(logprob)
    # optimize range of values after exponential function
    cost_matrix -= cost_matrix.min()
    cost_matrix *= 745. / (cost_matrix.max() - cost_matrix.min())
    # cost_matrix *= 1454
    cost_matrix -= 745
    cost_matrix = -1 * np.exp(cost_matrix)
    return cost_matrix


def greedy_assignment(cost):
    r_out=[]
    c_out=[]
    rows, cols = np.unravel_index(np.argsort(cost,kind="heapsort", axis=None), cost.shape)
    while len(rows)>0:
        r_out.append(rows[0])
        c_out.append(cols[0])
        mask = (rows!=rows[0])&(cols!=cols[0])
        rows = rows[mask]
        cols = cols[mask]
    return r_out, c_out

# class AdvancedMultiFilter(MultiFilter):
#     def __init__(self, *args, **kwargs):
#         super(AdvancedMultiFilter, self).__init__(*args,**kwargs)
#     def _critical_(self, ProbMat):
#         critical_i =[]
#         for i,n in enumerate(ProbMat):
