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
        return self.X[i], i
        
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
    
    def log_prob(self):
        probs = 0
        for i in self.Measurements.keys():
            try:
                probs += np.log(self.Measurement_Distribution().pdf(self.Measurements[i]-self.X[i]))
            except KeyError:
                self.predict(i=i)
                probs += np.log(self.Measurement_Distribution().pdf(self.Measurements[i]-self.X[i]))
        return probs

    def downfilter(self):
        t = max(self.x.keys())
        self.Measurements.pop(t, default=None)
        self.Controls.pop(t, default=None)
        self.X.pop(t, default=None)
        self.X_error.pop(t, default=None)
        self.Predicted_X.pop(t, default=None)
        self.Predicted_X_error.pop(t, default=None)

    def downdate(self):
        t = max(self.X.keys())
        self.Measurements.pop(t, None)
        self.X.update({t: self.Predicted_X[t]})
        self.X_error.update({t: self.Predicted_X_error[t]})
        # self.Controls.pop(t, None)

    def unpredict(self):
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
            print(self.X.values()[-1])

        return np.array(self.X.values(), dtype=float),\
               np.array(self.X_error.values(), dtype=float),\
               np.array(self.Predicted_X.values(), dtype=float),\
               np.array(self.Predicted_X_error.values(), dtype=float)


class KalmanFilter(Filter):
    def __init__(self, model, evolution_variance, measurement_variance, x0):
        super(KalmanFilter, self).__init__(model, meas_Dist=ss.norm, state_dist=ss.norm)
        self.Model = model
        self.A = self.Model.State_Matrix
        self.B = self.Model.Control_Matrix
        self.C = self.Model.Measurement_Matrix
        self.G = self.Model.Evolution_Matrix

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

        p = np.diag(np.ones(self.Model.State_dim) * max(measurement_variance))
        self.X_error.update({0: p})
        self.Predicted_X_error.update({0: p})

        self.X_0 = np.array(x0)
        self.X.update({0: x0})
        self.Predicted_X.update({0: x0})
        self.Measurements.update({0: x0})

    def predict(self, u=None, i=None):
        '''Prediction part of the Kalman Filtering process for one step'''
        x_, i = super(KalmanFilter, self).predict(u=u, i=i)

        x_ = np.dot(self.A, self.X[i-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.X_error[i-1]), self.A.transpose()) + np.dot(np.dot(self.G, self.Q), self.G.T)

        self.X.update({i: x_})
        self.X_error.update({i: p_})
        self.Predicted_X.update({i: x_})
        self.Predicted_X_error.update({i: p_})

        return x_, p_

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

        return x_, p_

class ParticleFilter(Filter):
    def __init__(self, model, x0, n=100, meas_dist=ss.uniform(), state_dist=ss.uniform()):
        super(ParticleFilter, self).__init__(model, state_dist=state_dist, meas_Dist=meas_dist)
        self.N = n
        self.X0 = np.array(x0)

        self.Particles = {}
        self.History = {}
        self.Weights = {}

        for n in range(self.N):
            self.Particles.update({n: self.X0[np.random.randint(0, self.X0.shape[0])]})
            self.Weights.update({n: 1./self.N})

        self.X.update({0: np.mean(self.Particles.values(), axis=0)})
        self.X_error.update({0: np.std(self.Particles.values(), axis=0)})

    def predict(self, u=None, i=None):
        '''Prediction part of the Particle Filtering process for one step'''
        super(ParticleFilter, self).predict(u=u, i=i)
        if i is None:
            i = max(self.X.keys())

        for j in self.Particles.keys():
            mean = self.Model.predict(self.Particles[j], u)
            self.Particles.update({j: self.State_Distribution.rvs() + mean})
            #self.Particles.update({i: ss.multivariate_normal(mean=np.dot(self.A, self.Particles[i]) + np.dot(self.B, u),
                                                             # cov=self.Q).rvs()})

        self.Predicted_X.update({i: np.mean(self.Particles.values(), axis=0)})
        self.Predicted_X_error.update({i: np.std(self.Particles.values(), axis=0)})
        return self.Particles

    def update(self, z=None, i=None):
        '''Updating part of the Kalman Filtering process for one step'''
        super(ParticleFilter, self).update(z=z, i=i)
        if i is None:
            i = max(self.X.keys())
        try:
            x = self.Predicted_X[i]
            p = self.Predicted_X_error[i]
        except KeyError:
            x, p = self.predict(i=i)

        for j in self.Weights.keys():
            self.Weights.update({j: self.Measurement_Distribution.pdf(z-self.Model.measure(self.Particles[j]))})

        w = np.sum(self.Weights.values())
        if w > 0:
            for j in self.Weights.keys():
                self.Weights[j] *= 1./w
        else:
            raise ValueError('Sum of weights is smaller than zero. No further Particle computation possible.')

        idx = np.sum(np.array(np.tile(np.cumsum(self.Weights.values()), self.N).reshape((self.N, -1)) <
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

        return self.Particles

class MultiKalman(object):
    def __init__(self, a, b, c, g, q, r, x_0, p_0, **kwargs):
        super(MultiKalman, self).__init__()
        self.A = np.array(a)
        self.B = np.array(b)
        self.C = np.array(c)
        self.G = np.array(g)
        self.Q = np.array(q)
        self.Q_0 = np.array(q)
        self.R = np.array(r)
        self.R_0 = np.array(r)
        self.Filters = []
        self.ActiveFilters = []
        self.FilterTreshold = 2

        self.P0 = np.array(p_0[0])

        for i in range(np.array(x_0).shape[0]):
            Filter = Kalman(a, b, c, g, q, r, x_0[i], p_0[i])
            self.ActiveFilters.append(Filter)
            self.Filters.append(Filter)

        self.x = range(len(self.Filters))
        self.p = range(len(self.Filters))
        self.predicted_x = range(len(self.Filters))
        self.predicted_p = range(len(self.Filters))

        for i, kal in enumerate(self.Filters):
            self.x[i] = kal.x
            self.p[i] = kal.p
            self.predicted_x[i] = kal.predicted_x
            self.predicted_p[i] = kal.predicted_p

    def predict(self, u):
        '''Prediction part of the Kalman Filtering process for one step'''
        for kal in self.ActiveFilters:
            if kal.Predict_Count > self.FilterTreshold:
                self.ActiveFilters.remove(kal)

        X = []
        P = []
        for i, kal in enumerate(self.Filters):
            if kal in self.ActiveFilters:
                try:
                    x, p = kal.predict(u[i])
                except IndexError:
                    'Default Controlparameter is zero'
                    x, p = kal.predict(np.zeros_like(u[0]))
            else:
                x = kal.x[max(kal.x.keys())]*np.nan
                p = kal.p[max(kal.x.keys())]*np.nan
            X.append(x)
            P.append(p)
        return np.array(X), np.array(P)

    def update(self, z):
        '''Updating part of the Kalman Filtering process for one step'''
        z = np.array(z)
        for i in range(z.shape[0] - len(self.ActiveFilters)):
            Filter = Kalman(self.A, self.B, self.C, self.G, self.Q, self.R,
                        np.ones_like(self.Filters[0].x[0])*np.nan, self.P0)
            self.ActiveFilters.append(Filter)
            self.Filters.append(Filter)

        probability_gain = []
        for zz in z:
            probability_gain_filter = []
            if np.all(zz == zz):
                for i in range(len(self.ActiveFilters)):
                    pre_probability = self.ActiveFilters[i].log_probability()
                    self.ActiveFilters[i].update(zz, reset=False)
                    probability_gain_filter.append(self.ActiveFilters[i].log_probability() - pre_probability)
                    self.ActiveFilters[i].downdate()
                probability_gain.append(probability_gain_filter)
            else:
                probability_gain.append([np.nan for u in range(len(self.ActiveFilters))])
        probability_gain = np.array(probability_gain)

        used = set()
        for i in range(len(self.ActiveFilters)):
            if np.all(probability_gain != probability_gain):
                break
            maxi, maxj = np.unravel_index(np.nanargmax(probability_gain), probability_gain.shape)
            probability_gain[maxi, :] = np.nan
            used.add(maxi)
            probability_gain[:, maxj] = np.nan
            self.ActiveFilters[maxj].update(z[maxi])

        x = []
        p = []
        for kal in self.Filters:
            x.append(kal.x[max(kal.x.keys())])
            p.append(kal.p[max(kal.x.keys())])
        return x, p

    def fit(self, u, z):
        u = np.array(u)
        z = np.array(z)
        assert u.shape[0] == z.shape[0]
        for i in range(z.shape[0]):

            print(len(self.ActiveFilters), len(self.Filters))

            x, p = self.predict(u[i])

            zz = z[i]
            if np.random.rand() > 0.8:
                print('added some noise!')
                zz = np.vstack((zz, [zz[0] + np.random.normal(0, 100, zz[0].shape)]))

            x, p = self.update(zz)

        return [np.array(x.values()) for x in self.x], \
               [np.array(p.values()) for p in self.p], \
               [np.array(x.values()) for x in self.predicted_x], \
               [np.array(p.values()) for p in self.predicted_p]


class Kalman_Mehra(Kalman):
    def __init__(self, *args, **kwargs):
        super(Kalman_Mehra, self).__init__(*args, **kwargs)
        self.Lag = int(kwargs.pop('lag', -1))
        self.LearnTime = int(kwargs.pop('learn_time', 1))

    def v(self, i):
        if i == 0:
            return self.z[min(self.z.keys())]-np.dot(np.dot(self.C, self.A), self.x[max(self.x.keys())])
        else:
            return self.z[i]-np.dot(np.dot(self.C, self.A), self.x[i-1])

    def c_hat(self, k):
        if k < 0:
            k = len(self.z.keys())-1
        n = len(self.z.keys())
        c = np.sum(np.array([
                            np.tensordot(self.v(i), self.v(i - k).T, axes=0)
                            for i in range(k, n)
                            ]), axis=0)
        return np.array(c/n)

    @staticmethod
    def pseudo_inverse(m):
        m = np.dot(np.linalg.inv(np.dot(m.transpose(), m)), m.transpose())
        return m

    def A_tilde(self):
        a = None
        a_ = None
        n = len(self.z.keys())
        k = max(0, n - self.Lag)
        for i in range(k, n):
            if a is not None:
                a_ = np.dot(np.dot(self.A, (np.identity(self.A.shape[0]))-np.dot(self.k(), self.C)), a_)
                a = np.vstack((a, np.dot(self.C, np.dot(a_, self.A))))
            else:
                a = np.identity(self.A.shape[0])
                a_ = np.array(a, copy=True)
                a = np.dot(self.C, np.dot(a, self.A))
        return a

    def c_tilde(self):
        c = None
        n = self.z.shape[0]
        k = max(0, n - self.Lag)
        for i in range(k, n):
            if c is not None:
                c = np.vstack((c, self.c_hat(i)))
            else:
                c = self.c_hat(i)
        return c

    def r_hat(self):
        try:
            cc = self.c_tilde()
            a_cross = self.pseudo_inverse(self.A_tilde())
            mh_hat = np.dot(self.k(), self.c_hat(0)) + np.dot(a_cross, cc)

            r = self.c_hat(0) - np.dot(self.C, mh_hat)
        except np.linalg.LinAlgError:
            r = self.R
            print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
        return r
    
    def predict(self, u):
        print(self.c_hat(0))
        if len(self.z.keys()) > self.LearnTime:
            self.R = self.c_hat(0)
        return super(Kalman_Mehra, self).predict(u)
    
    def update(self, meas, **kwargs):
        return super(Kalman_Mehra, self).update(meas, **kwargs)

    def fit(self, u, z):
        z = np.array(z)
        u = np.array(u)
        ret = super(Kalman_Mehra, self).fit(u, z)
        return ret

