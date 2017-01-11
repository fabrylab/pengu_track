#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Detectors.py

# Copyright (c) 2016, Red Hulk Productions
#
# This file is part of PenguTrack
#
# PenguTrack is beer software: you can serve it and/or drink
# it under the terms of the Book of Revelation as published by
# the evangelist John, either version 3 of the Book, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. It may cause
# penguins to explode. It may also cause further harm on coworkers,
# advisors or even lab equipment. See the Book of Revelation for
# more details.
#
# You should have received a copy of the Book of Revelation
# along with PenguTrack. If not, see <http://trumpdonald.org/>

"""
Module containing filter classes to be used with pengu-track detectors and models.
"""

from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
import sys
import copy
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
    def __init__(self, model, meas_dist=ss.uniform(), state_dist=ss.uniform()):
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
        # Generate i
        if i is None:
            i = max(self.Predicted_X.keys())+1
        # Generate u
        if u is None:
            try:
                u = self.Controls[i-1]
            except KeyError:
                u = np.zeros(self.Model.Control_dim)
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
                if np.any(self.Predicted_X.keys() < i):
                    print('Recursive Prediction, i = %s' % i)
                    u, i = self.predict(u, i=i-1)
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
            self.Measurements.update({i: z})
        # simplest possible update
        self.X.update({i: np.asarray([z.PositionX, z.PositionY])})
        return z, i

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
    
    def log_prob(self, keys=None, measurements=None, compare_bel=True):
        """
        Function to calculate the probability measure by predictions, measurements and corresponding distributions.

        Parameters
        ----------
        keys: list of int, optional
            Time-steps for which probability should be calculated.
        measurements: dict, optional
            List of PenguTrack.Measurement objects for which probability should be calculated.
        compare_bel: bool, optional
            If True, it will be tried to compare the believed state with the measurement. If False or
            there is no believe-value, the prediction will be taken.

        Returns
        ----------
        probs : float
            Probability of measurements at the given time-keys.
        """
        probs = 0
        if keys is None:
            keys = self.Measurements.keys()

        if measurements is None:
            for i in keys:
                # Generate Value for comparison with measurement
                try:
                    if compare_bel:
                        comparison = self.X[i]
                    else:
                        raise KeyError
                except KeyError:
                    try:
                        comparison = self.Predicted_X[i]
                    except KeyError:
                        self.predict(i=i)
                        comparison = self.Predicted_X[i]
                position = np.asarray([self.Measurements[i].PositionX, self.Measurements[i].PositionY])

                # def integrand(*args):
                #     x = np.array(args)
                #     return self.State_Distribution.pdf(x-comparison)*self.Measurement_Distribution.pdf(self.Model.measure(x)-position)
                #
                # integral = integrate.nquad(integrand,
                #                            np.array([-1*np.ones_like(self.Model.State_dim)*100,
                #                                      np.ones(self.Model.State_dim)*100]).T)
                # print(integral)l
                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(position
                                                                                 - self.Model.measure(comparison))))
        else:
            for i in keys:
                # Generate Value for comparison with measurement
                try:
                    if compare_bel:
                        comparison = self.X[i]
                    else:
                        raise KeyError
                except KeyError:
                    try:
                        comparison = self.Predicted_X[i]
                    except KeyError:
                        self.predict(i=i)
                        comparison = self.Predicted_X[i]

                position = np.asarray([measurements[i].PositionX, measurements[i].PositionY])
                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(position
                                                                                 - self.Model.measure(comparison))))
        return probs

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

        super(KalmanFilter, self).__init__(model, meas_dist=ss.multivariate_normal(cov=self.R),
                                           state_dist=ss.multivariate_normal(cov=self.Q))
        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

        p = np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.P_0 = p
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
        z = np.asarray([z.PositionX, z.PositionY])
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
            self.State_Distribution = ss.multivariate_normal(cov=self.Q)
        except np.linalg.LinAlgError:
            self.State_Distribution = ss.multivariate_normal(cov=self.Q_0)

        return z, i


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
        self.LogProbabilityThreshold = -18.
        self.filter_args = args
        self.filter_kwargs = kwargs
        self.CriticalIndex = None
        self.Probability_Gain = {}

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
        for j in self.ActiveFilters.keys():
            _filter = self.ActiveFilters[j]
            if np.array(_filter.Predicted_X.keys()[-1])-np.array(_filter.X.keys()[-1]) >= self.FilterThreshold:
                self.ActiveFilters.pop(j)

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
        print("Initial Filter Update")

        measurements = list(z)
        z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
        M = z.shape[0]

        for j in range(M):
            _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)

            _filter.Predicted_X.update({i: _filter.Model.infer_state(z[j])})
            _filter.X.update({i: _filter.Model.infer_state(z[j])})
            _filter.Measurements.update({i: measurements[j]})

            try:
                J = max(self.Filters.keys()) + 1
            except ValueError:
                J = 0
            self.ActiveFilters.update({J: _filter})
            self.Filters.update({J: _filter})

    def update(self, z=None, i=None):
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
        measurements = list(z)
        z = np.array([np.asarray([m.PositionX, m.PositionY]) for m in z], ndmin=2)
        M = z.shape[0]
        N = len(self.ActiveFilters.keys())

        if N == 0 and M > 0:
            self.initial_update(measurements, i)
            return measurements, i

        gain_dict = {}
        probability_gain = np.ones((max(M, N), M))*self.LogProbabilityThreshold

        for j, k in enumerate(self.ActiveFilters.keys()):
            gain_dict.update({j: k})
            for m in range(M):
                probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i], measurements={i: measurements[m]})

        # self.Probability_Gain.update({i: np.array(probability_gain)})
        # self.CriticalIndex = gain_dict[np.nanargmax([np.sort(a)[-2]/np.sort(a)[-1] for a in probability_gain[:N]])]
        x = {}
        x_err = {}
        for j in range(M):

            if not np.all(np.isnan(probability_gain)+np.isinf(probability_gain)):
                k, m = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
            else:
                k, m = np.unravel_index(np.nanargmin(probability_gain), probability_gain.shape)

            if probability_gain[k, m] > self.LogProbabilityThreshold:
                self.ActiveFilters[gain_dict[k]].update(z=measurements[m], i=i)
                x.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X[i]})
                x_err.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X_error[i]})

            else:
                print("DEPRECATED TRACK WITH PROB %s IN FRAME %s" % (probability_gain[k, m], i))
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

            probability_gain[k, :] = np.nan
            probability_gain[:, m] = np.nan

        if len(self.ActiveFilters.keys()) < M:
            raise RuntimeError('Lost Filters on the way. This should never happen')
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

    def log_prob(self, keys=None):
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
            prob += self.Filters[j].log_prob(keys=keys)
        return prob

# class AdvancedMultiFilter(MultiFilter):
#     def __init__(self, *args, **kwargs):
#         super(AdvancedMultiFilter, self).__init__(*args,**kwargs)
#     def _critical_(self, ProbMat):
#         critical_i =[]
#         for i,n in enumerate(ProbMat):