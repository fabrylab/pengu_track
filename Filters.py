from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
import scipy.optimize as opt
from timeit import default_timer as timer

class Filter(object):
    def __init__(self, model, meas_dist=ss.uniform(), state_dist=ss.uniform(), *args, **kwargs):
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
                    print('Recursive Prediction, i = %s'%i)
                    u, i = self.predict(u, i=i-1)
                    x = self.Predicted_X[i-1]
                else:
                    raise KeyError("Nothing to predict from. Need initial value")

        # Make simplest possible Prediction (can be overwritten)
        self.Predicted_X.update({i: self.Model.predict(x, u)})
        return u, i
        
    def update(self, z=None, i=None):
        # Generate i
        if i is None:
            i = max(self.Measurements.keys())+1
        # Generate z
        if z is None:
            try:
                z = self.Measurements[i]
            except KeyError:
                raise KeyError("No measurement for timepoint %s."%i)
        else:
            self.Measurements.update({i: z})
        # simplest possible update
        self.X.update({i: z})
        return z, i

    def filter(self, u=None, z=None, i=None):
        self.predict(u=u, i=i)
        x = self.update(z=z, i=i)
        return x
    
    def log_prob(self, keys=None, measurements=None):
        probs = 0
        if keys is None:
            keys = self.Measurements.keys()

        if measurements is None:
            for i in keys:
                # Generate Value for comparison with measurement
                try:
                    comparison = self.X[i]
                except KeyError:
                    try:
                        comparison = self.Predicted_X[i]
                    except KeyError:
                        self.predict(i=i)
                        comparison = self.Predicted_X[i]

                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(self.Measurements[i]
                                                                                 - self.Model.measure(comparison))))
        else:
            for i in keys:
                # Generate Value for comparison with measurement
                try:
                    comparison = self.X[i]
                except KeyError:
                    try:
                        comparison = self.Predicted_X[i]
                    except KeyError:
                        self.predict(i=i)
                        comparison = self.Predicted_X[i]

                probs += np.log(np.linalg.norm(self.Measurement_Distribution.pdf(measurements[i]
                                                                                 - self.Model.measure(comparison))))
        return probs

    def downfilter(self, t=None):
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
        # Generate t
        if t is None:
            t = max(self.X.keys())
        # Remove believe and Measurement entries for this timestep
        self.X.pop(t, None)
        self.X_error.pop(t, None)
        self.Measurements.pop(t, None)

    def unpredict(self, t=None):
        # Generate t
        if t is None:
            t = max(self.Predicted_X.keys())
        # Remove believe and Prediction entries for this timestep
        self.Controls.pop(t, None)
        self.X.pop(t, None)
        self.X_error.pop(t, None)
        self.Predicted_X.pop(t, None)
        self.Predicted_X_error.pop(t, None)

    # def analyze_model(self):

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

    # def fit(self, u, z, params):
    #     '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
    #     It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
    #     u = np.array(u)
    #     z = np.array(z)
    #     assert u.shape[0] == z.shape[0]
    #
    #     def evaluate(*args):
    #         self.__init__(self.Model, *args)
    #         for i in range(z.shape[0]):
    #             self.predict(u=u[i], i=i+1)
    #             self.update(z=z[i], i=i+1)
    #         return -1*self.log_prob(self.Measurements.keys())
    #
    #     opt.minimize(evaluate, )
    #
    #
    #     return np.array(self.X.values(), dtype=float),\
    #            np.array(self.X_error.values(), dtype=float),\
    #            np.array(self.Predicted_X.values(), dtype=float),\
    #            np.array(self.Predicted_X_error.values(), dtype=float)


class KalmanFilter(Filter):
    def __init__(self, model, evolution_variance, measurement_variance, **kwargs):
        self.Model = model

        # if x0 is None:
        #     x0 = np.zeros(self.Model.State_dim)
        # x0 = np.asarray(x0)

        evolution_variance = np.array(evolution_variance, dtype=float)
        if evolution_variance.shape != (long(self.Model.Evolution_dim),):
            evolution_variance = np.ones(self.Model.Evolution_dim) * np.mean(evolution_variance)

        self.Evolution_Variance = evolution_variance
        self.Q = np.diag(evolution_variance)
        self.Q_0 = np.diag(evolution_variance)

        measurement_variance = np.array(measurement_variance, dtype=float)
        if measurement_variance.shape != (long(self.Model.Meas_dim),):
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

        # self.X_0 = np.array(x0)
        # self.X.update({0: x0})
        # self.Predicted_X.update({0: x0})
        # self.Measurements.update({0: self.Model.measure(x0)})

    def predict(self, u=None, i=None):
        '''Prediction part of the Kalman Filtering process for one step'''
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
        '''Updating part of the Kalman Filtering process for one step'''
        z, i = super(KalmanFilter, self).update(z=z, i=i)
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
            k = np.dot(np.dot(p, self.C.transpose()), np.linalg.inv(np.dot(np.dot(self.C, p), self.C.transpose()) + self.R))
        except np.linalg.LinAlgError, e:
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
    def __init__(self, *args, **kwargs):
        super(AdvancedKalmanFilter, self).__init__(*args, **kwargs)
        self.Lag = -1 * int(kwargs.pop('lag', -1))

    def predict(self, *args, **kwargs):
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
        dif = np.array([np.dot(self.C, np.array(self.X.get(k, None)).T).T
                        - self.Measurements[k] for k in self.Measurements.keys()])
        self.R = np.cov(dif.T)
        if np.any(np.isnan(self.R)) or np.any(np.linalg.eigvals(self.R) < np.diag(self.R_0)):
            self.R = self.R_0
        print("R at %s"%self.R)
        return super(AdvancedKalmanFilter, self).update(*args, **kwargs)


class ParticleFilter(Filter):
    def __init__(self, model, x0=None, n=100, meas_dist=ss.uniform(), state_dist=ss.uniform()):
        super(ParticleFilter, self).__init__(model, state_dist=state_dist, meas_dist=meas_dist)
        self.N = n
        if x0 is None:
            x0 = np.zeros(self.Model.State_dim)
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
        # self.X.update({i: np.mean(self.Particles.values(), axis=0)})
        # self.X_error.update({i: np.std(self.Particles.values(), axis=0)})
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
    def __init__(self, _filter, model, *args, **kwargs):
        super(MultiFilter, self).__init__(model)
        self.Filter_Class = _filter
        self.Filters = {}
        self.ActiveFilters = {}
        self.FilterThreshold = 3
        self.LogProbabilityThreshold = -1000. #-1 * np.inf
        self.filter_args = args
        self.filter_kwargs = kwargs

    def predict(self, u=None, i=None):
        # if i is None:
        #     i = max(self.Predicted_X.keys())+1
        # if u is None:
        #     try:
        #         u = self.Controls[i-1]
        #     except KeyError:
        #         u = np.zeros(self.Model.Control_dim)
        # else:
        #     self.Controls.update({i-1: u})
        # # try:
        # #     x = self.X[i-1]
        # # except KeyError:
        # #     print('recursion, i=%s'%i)
        # #     x = self.predict(u, i=i-1)

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
        # self.X.update({i: predicted_x})
        # self.X_error.update({i: predicted_x_error})
        return u, i

    def initial_update(self, z, i):
        print("Initial Filter Update")

        z = np.array(z, ndmin=2)
        M = z.shape[0]

        for j in range(M):
            _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)

            _filter.Predicted_X.update({i: _filter.Model.infer_state(z[j])})
            _filter.X.update({i: _filter.Model.infer_state(z[j])})
            _filter.Measurements.update({i: z[j]})

            try:
                J = max(self.Filters.keys()) + 1
            except ValueError:
                J = 0
            self.ActiveFilters.update({J: _filter})
            self.Filters.update({J: _filter})

    def update(self, z=None, i=None):
        # z, i = super(MultiFilter, self).update(z=z, i=i)
        z = np.array(z, ndmin=2)
        M = z.shape[0]
        N = len(self.ActiveFilters.keys())

        if N == 0 and M > 0:
            self.initial_update(z, i)
            return z, i

        gain_dict = {}
        probability_gain = -1./np.zeros((max(M,N),M))

        for j, k in enumerate(self.ActiveFilters.keys()):
            gain_dict.update({j: k})
            for m in range(M):
                probability_gain[j, m] = self.ActiveFilters[k].log_prob(keys=[i], measurements={i: z[m]})

        # print(probability_gain)
        x = {}
        x_err = {}
        for j in range(M):

            if not np.all(np.isnan(probability_gain)+np.isinf(probability_gain)):
                # k = np.nanargmin(np.nanmean(probability_gain, axis=1))
                # m = np.nanargmax(probability_gain[k])
                k, m = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
            else:
                k, m = np.unravel_index(np.nanargmin(probability_gain), probability_gain.shape)

            if probability_gain[k, m] > self.LogProbabilityThreshold:
                self.ActiveFilters[gain_dict[k]].update(z=z[m], i=i)
                x.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X[i]})
                x_err.update({gain_dict[k]: self.ActiveFilters[gain_dict[k]].X_error[i]})
            else:
                l = max(self.Filters.keys()) + 1
                _filter = self.Filter_Class(self.Model, *self.filter_args, **self.filter_kwargs)
                _filter.Predicted_X.update({i: self.Model.infer_state(z[m])})
                _filter.X.update({i: self.Model.infer_state(z[m])})
                _filter.Measurements.update({i: z[m]})

                self.ActiveFilters.update({l: _filter})
                self.Filters.update({l: _filter})
            probability_gain[k, :] = np.nan
            probability_gain[:, m] = np.nan
        # print(probability_gain)

        if len(self.ActiveFilters.keys()) < M:
            raise RuntimeError('Lost Filters on the way. This should never happen')

        # self.X.update({i: x})
        # self.X_error.update({i: x_err})
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

