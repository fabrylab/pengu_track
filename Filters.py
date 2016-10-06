from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
from timeit import default_timer as timer

class Filter(object):
    def __init__(self, model, meas_Dist=ss.uniform, state_dist=ss.uniform):
        self.Model = model
        self.Measurement_Distribution = meas_Dist
        self.State_Distribution = state_dist
        
        self.X = {}
        self.X_error = {}
        self.Predicted_X = {}
        self.Predicted_X_error = {}
        
        self.Measurements = {}
        self.Controls = {}

    def predict(self, u=None, i=None):
        if i is None:
            i = max(self.X.keys())
        if u is None:
            try:
                u = self.Controls[i-1]
            except KeyError:
                u = np.zeros(self.Model.Control_dim)
        else:
            self.Controls.update({i-1: u})
        try:
            x = self.X[i-1]
        except KeyError:
            print('recursion, i=%s'%i)
            x = self.predict(u, i=i-1)
        self.X.update({i: self.Model.predict(x, u)})
        return u, i
        
    def update(self, z=None, i=None):
        if i is None:
            i = max(self.X.keys())
        if z is None:
            z = self.Measurements[i]
        else:
            self.Measurements.update({i: z})
        # self.X.update({i: z})
        return z, i

    def filter(self, u=None, z=None, i=None):
        self.predict(u=u, i=i)
        x = self.update(z=z, i=i)
        return x
    
    def log_prob(self, keys=None):
        probs = 0
        if keys is None:
            keys = self.Measurements.keys()
        for i in keys:
            try:
                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(self.Measurements[i]
                                                                                 - self.Model.measure(self.X[i]))))
            except KeyError:
                self.predict(i=i)
                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(self.Measurements[i]
                                                                                 - self.Model.measure(self.X[i]))))
        return probs

    def downfilter(self, t=None):
        if t is None:
            t = max(self.x.keys())
        self.Measurements.pop(t, default=None)
        self.Controls.pop(t, default=None)
        self.X.pop(t, default=None)
        self.X_error.pop(t, default=None)
        self.Predicted_X.pop(t, default=None)
        self.Predicted_X_error.pop(t, default=None)

    def downdate(self, t=None):
        if t is None:
            t = max(self.X.keys())
        self.Measurements.pop(t, None)
        self.X.update({t: self.Predicted_X[t]})
        self.X_error.update({t: self.Predicted_X_error[t]})
        # self.Controls.pop(t, None)

    def unpredict(self, t=None):
        if t is None:
            t = max(self.X.keys())
        # self.Controls.pop(t, default=None)
        self.X.pop(t, default=None)
        self.X_error.pop(t, default=None)
        self.Predicted_X.pop(t, default=None)
        self.Predicted_X_error.pop(t, default=None)

    def fit(self, u, z):
        '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
        u = np.array(u)
        z = np.array(z)
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
    def __init__(self, model, x0, evolution_variance, measurement_variance, **kwargs):
        self.Model = model

        evolution_variance = np.array(evolution_variance, dtype=float)
        if evolution_variance.shape != (long(self.Model.Evolution_dim),):
            evolution_variance = np.ones(self.Model.Evolution_dim) * np.mean(evolution_variance)
        self.Q = np.diag(evolution_variance)
        self.Q_0 = np.diag(evolution_variance)

        measurement_variance = np.array(measurement_variance, dtype=float)
        if measurement_variance.shape != (long(self.Model.Meas_dim),):
            measurement_variance = np.ones(self.Model.Meas_dim) * np.mean(measurement_variance)
        self.R = np.diag(measurement_variance)
        self.R_0 = np.diag(measurement_variance)

        super(KalmanFilter, self).__init__(model, meas_Dist=ss.multivariate_normal(cov=self.R),
                                           state_dist=ss.multivariate_normal(cov=self.Q))
        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

        p = np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.X_error.update({0: p})
        self.Predicted_X_error.update({0: p})

        self.X_0 = np.array(x0)
        self.X.update({0: x0})
        self.Predicted_X.update({0: x0})
        self.Measurements.update({0: self.Model.measure(x0)})

    def predict(self, u=None, i=None):
        '''Prediction part of the Kalman Filtering process for one step'''
        u, i = super(KalmanFilter, self).predict(u=u, i=i)

        x_ = np.dot(self.A, self.X[i-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.X_error[i-1]), self.A.transpose()) + np.dot(np.dot(self.G, self.Q), self.G.T)

        self.X.update({i: x_})
        self.X_error.update({i: p_})
        self.Predicted_X.update({i: x_})
        self.Predicted_X_error.update({i: p_})

        return u, i

    def update(self, z=None, i=None):
        '''Updating part of the Kalman Filtering process for one step'''
        z, i = super(KalmanFilter, self).update(z=z, i=i)
        try:
            x = self.Predicted_X[i]
            p = self.Predicted_X_error[i]
        except KeyError:
            x, p = self.predict(i=i)

        k = np.dot(np.dot(p, self.C.transpose()), np.linalg.inv(np.dot(np.dot(self.C, p), self.C.transpose()) + self.R))

        y = z - np.dot(self.C, x)

        x_ = x + np.dot(k, y)
        p_ = p - np.dot(np.dot(k, self.C), p)

        self.X.update({i: x_})
        self.X_error.update({i: p_})

        return z, i


class AdvancedKalmanFilter(KalmanFilter):
    def __init__(self, *args, **kwargs):
        super(AdvancedKalmanFilter, self).__init__(*args, **kwargs)
        self.Lag = -1 * int(kwargs.pop('lag', -1))

    def predict(self, *args, **kwargs):
        if self.Lag == 1 or (-1 * self.Lag > len(self.Predicted_X.keys())):
            lag = 0
        else:
            lag = self.Lag
        self.Q = np.dot(np.dot(self.G.T, np.cov(np.asarray(self.Predicted_X.values()[lag:]).T)), self.G)
        if np.any(np.isnan(self.Q)):
            self.Q = self.Q_0
        print("Q at %s"%self.Q)
        return super(AdvancedKalmanFilter, self).predict(*args, **kwargs)

    def update(self, *args, **kwargs):
        dif = np.array([np.dot(self.C, np.array(self.X.get(k, None)).T).T
                        - self.Measurements[k] for k in self.Measurements.keys()])
        self.R = np.cov(dif.T)
        if np.any(np.isnan(self.R)) or np.any(np.diag(self.R) < np.diag(self.R_0)):
            self.R = self.R_0
        print("R at %s"%self.R)
        return super(AdvancedKalmanFilter, self).update(*args, **kwargs)


class ParticleFilter(Filter):
    def __init__(self, model, x0, n=100, meas_dist=ss.uniform(), state_dist=ss.uniform()):
        super(ParticleFilter, self).__init__(model, state_dist=state_dist, meas_Dist=meas_dist)
        self.N = n
        self.X0 = np.array([x0])

        self.Particles = {}
        self.History = {}
        self.Weights = {}

        for n in range(self.N):
            self.Particles.update({n: self.X0[np.random.randint(0, self.X0.shape[0])]})
            self.Weights.update({n: 1./self.N})

        self.X.update({0: np.mean(self.Particles.values(), axis=0)})
        self.Predicted_X.update({0: np.mean(self.Particles.values(), axis=0)})
        self.X_error.update({0: np.std(self.Particles.values(), axis=0)})
        self.Predicted_X_error.update({0: np.std(self.Particles.values(), axis=0)})

        self.Measurements.update({0: self.Model.measure(np.mean(self.X0, axis=0))})

    def predict(self, u=None, i=None):
        '''Prediction part of the Particle Filtering process for one step'''
        u, i = super(ParticleFilter, self).predict(u=u, i=i)

        for j in self.Particles.keys():
            mean = self.Model.predict(self.Particles[j], u)
            self.Particles.update({j: self.State_Distribution.rvs() + mean})

        self.Predicted_X.update({i: np.mean(self.Particles.values(), axis=0)})
        self.Predicted_X_error.update({i: np.std(self.Particles.values(), axis=0)})
        self.X.update({i: np.mean(self.Particles.values(), axis=0)})
        self.X_error.update({i: np.std(self.Particles.values(), axis=0)})
        return u, i

    def update(self, z=None, i=None):
        '''Updating part of the Kalman Filtering process for one step'''
        z, i = super(ParticleFilter, self).update(z=z, i=i)

        for j in self.Weights.keys():
            self.Weights.update({j: self.Measurement_Distribution.logpdf(z-self.Model.measure(self.Particles[j]))})
        weights = self.Weights.values()
        min = np.amin(weights)
        max = np.amax(weights)
        if max > min:
            weights = (weights - min)#/(max - min)
            weights = np.cumsum(np.exp(weights))
            weights = weights/weights[-1]
            print("Standard")
        else:
            weights = np.cumsum(weights)
            weights = weights/weights[-1]
            print("Workaround")

        # print(w)
        # if w > 0:
        #     for j in self.Weights.keys():
        #         self.Weights[j] += np.amin(self.Weights.values())
        #         self.Weights[j] *= 1./w
        # else:
        #     print("Updating failed. Starting New Filter.")
        #     # self.__init__(self.Model, z,
        #     #               n=self.N, meas_dist=self.Measurement_Distribution, state_dist=self.State_Distribution)
        #     for j in self.Particles.keys():
        #         self.Particles.update({j: z})
        #     self.predict(i=i)
        #     return self.update(z=z, i=i)
        #     # raise ValueError('Sum of weights is smaller than zero. No further Particle computation possible.')

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

        return z, i


class MultiFilter(Filter):
    def __init__(self, _filter, model, x0, *args, **kwargs):
        super(MultiFilter, self).__init__(model)
        self.Filter_Class = _filter
        self.Filters = {}
        self.ActiveFilters = {}
        self.FilterThreshold = 2
        self.LogProbabilityThreshold = -1 * np.inf
        self.filter_args = args
        self.filter_kwargs = kwargs

        predicted_x = {}
        predicted_x_err = {}
        for j in range(np.array(x0).shape[0]):
            _filter = self.Filter_Class(model, x0[j], *args, **kwargs)
            self.ActiveFilters.update({j: _filter})
            self.Filters.update({j: _filter})
            predicted_x.update({j: self.Filters[j].X[0]})
            predicted_x_err.update({j: self.Filters[j].X_error[0]})

        self.Predicted_X.update({0: predicted_x})
        self.Predicted_X_error.update({0: predicted_x_err})
        self.X.update({0: predicted_x})
        self.X_error.update({0: predicted_x_err})

    def predict(self, u=None, i=None):
        if i is None:
            i = max(self.X.keys())
        if u is None:
            try:
                u = self.Controls[i-1]
            except KeyError:
                u = np.zeros(self.Model.Control_dim)
        else:
            self.Controls.update({i-1: u})
        try:
            x = self.X[i-1]
        except KeyError:
            print('recursion, i=%s'%i)
            x = self.predict(u, i=i-1)

        for j in self.ActiveFilters.keys():
            _filter = self.ActiveFilters[j]
            if np.array(_filter.Predicted_X.keys()[-1])-np.array(_filter.X.keys()[-1]) > self.FilterThreshold:
                self.ActiveFilters.pop(j, default=None)
        predicted_x = {}
        predicted_x_error = {}
        for j in self.Filters.keys():
            if j in self.ActiveFilters.keys():
                self.ActiveFilters[j].predict(u=u[j], i=i)
                predicted_x.update({j: self.ActiveFilters[j].Predicted_X[i]})
                predicted_x_error.update({j: self.ActiveFilters[j].Predicted_X_error[i]})
        self.Predicted_X.update({i: predicted_x})
        self.Predicted_X_error.update({i: predicted_x_error})
        self.X.update({i: predicted_x})
        self.X_error.update({i: predicted_x_error})
        return u, i

    def update(self, z=None, i=None):
        z, i = super(MultiFilter, self).update(z=z, i=i)
        z = np.array(z)
        M = z.shape[0]

        if M > len(self.ActiveFilters.keys()):
            for j in range(M - len(self.ActiveFilters.keys())):
                _filter = self.Filter_Class(self.Model, np.ones_like(self.ActiveFilters[0].X[0]) * np.nan,
                                            *self.filter_args, **self.filter_kwargs)
                J = max(self.Filters.keys()) + 1
                self.ActiveFilters.update({J: _filter})
                self.Filters.update({J: _filter})

        N = len(self.ActiveFilters.keys())

        gain_dict = {}
        probability_gain = []

        for k, j in enumerate(self.ActiveFilters.keys()):
            gain_dict.update({k: j})
            prob_gain_inner = []
            for n in range(M):
                try:
                    self.ActiveFilters[j].update(z=z[n], i=i)
                    prob_gain_inner.append(self.ActiveFilters[j].log_prob(keys=[i]))
                except ValueError:
                    prob_gain_inner.append(-1*np.inf)
                self.ActiveFilters[j].downdate(t=i)

            # if np.all(np.array(prob_gain_inner) <= self.LogProbabilityThreshold):
            #     self.ActiveFilters.pop(j, default=None)
            #     _filter = self.Filter_Class(self.Model, np.ones_like(self.ActiveFilters[0].X[0]) * np.nan,
            #                                 *self.filter_args, **self.filter_kwargs)
            #     J = max(self.Filters.keys()) + 1
            #     self.ActiveFilters.update({J: _filter})
            #     self.Filters.update({J: _filter})
            #     gain_dict.update({k: J})

            probability_gain.append(prob_gain_inner)

        probability_gain = np.array(probability_gain)
        x = {}
        x_err = {}
        for j in range(N):
            # print(probability_gain)
            if np.all(probability_gain != probability_gain):
                break
            j, m = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
            # print(np.linalg.norm(self.ActiveFilters[gain_dict[j]].Predicted_X[i][::2]-z[m]))
            probability_gain[j, :] = np.nan
            probability_gain[:, m] = np.nan
            self.ActiveFilters[gain_dict[j]].update(z=z[m], i=i)
            x.update({gain_dict[j]: self.ActiveFilters[gain_dict[j]].X[i]})
            x_err.update({gain_dict[j]: self.ActiveFilters[gain_dict[j]].X_error[i]})

        self.X.update({i: x})
        self.X_error.update({i: x_err})

        return z, i

    def fit(self, u, z):
        '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
        u = np.array(u)
        z = np.array(z)
        assert u.shape[0] == z.shape[0]

        for i in range(z.shape[0]):
            self.predict(u=u[i], i=i+1)
            self.update(z=z[i], i=i+1)
            # print(self.log_prob())

        return self.X, self.X_error, self.Predicted_X, self.Predicted_X_error

    def downfilter(self, t=None):
        for k in self.Filters.keys():
            self.Filters[k].downfilter(t=t)

    def downdate(self, t=None):
        for k in self.Filters.keys():
            self.Filters[k].downdate(t=t)

    def unpredict(self, t=None):
        for k in self.Filters.keys():
            self.Filters[k].downdate(t=t)

    def log_prob(self, keys=None):
        prob = 0
        for j in self.Filters.keys():
            prob += self.Filters[j].log_prob(keys=keys)
        return prob


# class MultiKalman(object):
#     def __init__(self, a, b, c, g, q, r, x_0, p_0, **kwargs):
#         super(MultiKalman, self).__init__()
#         self.A = np.array(a)
#         self.B = np.array(b)
#         self.C = np.array(c)
#         self.G = np.array(g)
#         self.Q = np.array(q)
#         self.Q_0 = np.array(q)
#         self.R = np.array(r)
#         self.R_0 = np.array(r)
#         self.Filters = []
#         self.ActiveFilters = []
#         self.FilterThreshold = 2
#
#         self.P0 = np.array(p_0[0])
#
#         for i in range(np.array(x_0).shape[0]):
#             Filter = Kalman(a, b, c, g, q, r, x_0[i], p_0[i])
#             self.ActiveFilters.append(Filter)
#             self.Filters.append(Filter)
#
#         self.x = range(len(self.Filters))
#         self.p = range(len(self.Filters))
#         self.predicted_x = range(len(self.Filters))
#         self.predicted_p = range(len(self.Filters))
#
#         for i, kal in enumerate(self.Filters):
#             self.x[i] = kal.x
#             self.p[i] = kal.p
#             self.predicted_x[i] = kal.predicted_x
#             self.predicted_p[i] = kal.predicted_p
#
#     def predict(self, u):
#         '''Prediction part of the Kalman Filtering process for one step'''
#         for kal in self.ActiveFilters:
#             if kal.Predict_Count > self.FilterThreshold:
#                 self.ActiveFilters.remove(kal)
#
#         X = []
#         P = []
#         for i, kal in enumerate(self.Filters):
#             if kal in self.ActiveFilters:
#                 try:
#                     x, p = kal.predict(u[i])
#                 except IndexError:
#                     'Default Controlparameter is zero'
#                     x, p = kal.predict(np.zeros_like(u[0]))
#             else:
#                 x = kal.x[max(kal.x.keys())]*np.nan
#                 p = kal.p[max(kal.x.keys())]*np.nan
#             X.append(x)
#             P.append(p)
#         return np.array(X), np.array(P)
#
#     def update(self, z):
#         '''Updating part of the Kalman Filtering process for one step'''
#         z = np.array(z)
#         for i in range(z.shape[0] - len(self.ActiveFilters)):
#             Filter = Kalman(self.A, self.B, self.C, self.G, self.Q, self.R,
#                         np.ones_like(self.Filters[0].x[0])*np.nan, self.P0)
#             self.ActiveFilters.append(Filter)
#             self.Filters.append(Filter)
#
#         probability_gain = []
#         for zz in z:
#             probability_gain_filter = []
#             if np.all(zz == zz):
#                 for i in range(len(self.ActiveFilters)):
#                     pre_probability = self.ActiveFilters[i].log_probability()
#                     self.ActiveFilters[i].update(zz, reset=False)
#                     probability_gain_filter.append(self.ActiveFilters[i].log_probability() - pre_probability)
#                     self.ActiveFilters[i].downdate()
#                 probability_gain.append(probability_gain_filter)
#             else:
#                 probability_gain.append([np.nan for u in range(len(self.ActiveFilters))])
#         probability_gain = np.array(probability_gain)
#
#         used = set()
#         for i in range(len(self.ActiveFilters)):
#             if np.all(probability_gain != probability_gain):
#                 break
#             maxi, maxj = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
#             probability_gain[maxi, :] = np.nan
#             used.add(maxi)
#             probability_gain[:, maxj] = np.nan
#             self.ActiveFilters[maxj].update(z[maxi])
#
#         x = []
#         p = []
#         for kal in self.Filters:
#             x.append(kal.x[max(kal.x.keys())])
#             p.append(kal.p[max(kal.x.keys())])
#         return x, p
#
#     def fit(self, u, z):
#         u = np.array(u)
#         z = np.array(z)
#         assert u.shape[0] == z.shape[0]
#         for i in range(z.shape[0]):
#
#             print(len(self.ActiveFilters), len(self.Filters))
#
#             x, p = self.predict(u[i])
#
#             zz = z[i]
#             if np.random.rand() > 0.8:
#                 print('added some noise!')
#                 zz = np.vstack((zz, [zz[0] + np.random.normal(0, 100, zz[0].shape)]))
#
#             x, p = self.update(zz)
#
#         return [np.array(x.values()) for x in self.x], \
#                [np.array(p.values()) for p in self.p], \
#                [np.array(x.values()) for x in self.predicted_x], \
#                [np.array(p.values()) for p in self.predicted_p]


# class Kalman_Mehra(Kalman):
#     def __init__(self, *args, **kwargs):
#         super(Kalman_Mehra, self).__init__(*args, **kwargs)
#         self.Lag = int(kwargs.pop('lag', -1))
#         self.LearnTime = int(kwargs.pop('learn_time', 1))
#
#     def v(self, i):
#         if i == 0:
#             return self.z[min(self.z.keys())]-np.dot(np.dot(self.C, self.A), self.x[max(self.x.keys())])
#         else:
#             return self.z[i]-np.dot(np.dot(self.C, self.A), self.x[i-1])
#
#     def c_hat(self, k):
#         if k < 0:
#             k = len(self.z.keys())-1
#         n = len(self.z.keys())
#         c = np.sum(np.array([
#                             np.tensordot(self.v(i), self.v(i - k).T, axes=0)
#                             for i in range(k, n)
#                             ]), axis=0)
#         return np.array(c/n)
#
#     @staticmethod
#     def pseudo_inverse(m):
#         m = np.dot(np.linalg.inv(np.dot(m.transpose(), m)), m.transpose())
#         return m
#
#     def A_tilde(self):
#         a = None
#         a_ = None
#         n = len(self.z.keys())
#         k = max(0, n - self.Lag)
#         for i in range(k, n):
#             if a is not None:
#                 a_ = np.dot(np.dot(self.A, (np.identity(self.A.shape[0]))-np.dot(self.k(), self.C)), a_)
#                 a = np.vstack((a, np.dot(self.C, np.dot(a_, self.A))))
#             else:
#                 a = np.identity(self.A.shape[0])
#                 a_ = np.array(a, copy=True)
#                 a = np.dot(self.C, np.dot(a, self.A))
#         return a
#
#     def c_tilde(self):
#         c = None
#         n = self.z.shape[0]
#         k = max(0, n - self.Lag)
#         for i in range(k, n):
#             if c is not None:
#                 c = np.vstack((c, self.c_hat(i)))
#             else:
#                 c = self.c_hat(i)
#         return c
#
#     def r_hat(self):
#         try:
#             cc = self.c_tilde()
#             a_cross = self.pseudo_inverse(self.A_tilde())
#             mh_hat = np.dot(self.k(), self.c_hat(0)) + np.dot(a_cross, cc)
#
#             r = self.c_hat(0) - np.dot(self.C, mh_hat)
#         except np.linalg.LinAlgError:
#             r = self.R
#             print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
#         return r
#
#     def predict(self, u):
#         print(self.c_hat(0))
#         if len(self.z.keys()) > self.LearnTime:
#             self.R = self.c_hat(0)
#         return super(Kalman_Mehra, self).predict(u)
#
#     def update(self, meas, **kwargs):
#         return super(Kalman_Mehra, self).update(meas, **kwargs)
#
#     def fit(self, u, z):
#         z = np.array(z)
#         u = np.array(u)
#         ret = super(Kalman_Mehra, self).fit(u, z)
#         return ret

