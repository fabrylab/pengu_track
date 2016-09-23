from __future__ import print_function, division
import numpy as np
import scipy.stats as ss
from timeit import default_timer as timer


class Model(object):
    def __init__(self, state_dim, controle_dim, meas_dim,
                 state_function = None,
                 controle_function = None,
                 meas_function = None):
        self.State = np.zeros(state_dim)
        self.Controle = np.zeros(controle_dim)
        self.Meas = np.zeros(meas_dim)
        
        if state_function is None:
            state_function = lambda x: np.dot(np.identity(state_dim), x)
        self.State_Function = state_function
        
        if controle_function is None:
            controle_function = lambda x: np.dot(np.identity(max(state_dim, controle_dim))[:state_dim,:controle_dim], x)
        self.Controle_Function = controle_function
        
        if meas_function is None:
            meas_function = lambda x: np.dot(np.identity(max(state_dim, meas_dim))[:meas_dim,:state_dim], x)
        self.Measurement_Function = meas_function
        
    def predict(self, state_vector, controle_vector):
        return self.State_Function(state_vector) + self.Controle_Function(controle_vector)
    
    def update(self, state_vector):
        return self.Measurement_Function(state_vector)

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
        
        state_function = lambda x: np.dot(Matrix,x)
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
        
        Matrix[1::dim,::dim] = np.diag(np.ones(dim))
        
        Matrix[::dim,::dim] = np.diag(np.ones(dim)*-1*self.Coefficients[0])
        for i,c in enumerate(self.Coefficients):
            Matrix[::dim,i+1::dim] = np.diag(np.ones(dim)*c)
        
        Matrix[1::dim,::dim] = np.diag(np.ones(dim)*-1)
        Matrix[1::dim,::dim] = np.diag(np.ones(dim)*-1)
        for i,c in enumerate(self.Coefficients[1:]):
            Matrix[1::dim,i+1::dim] += np.diag(np.ones(dim)*c)
            Matrix[::dim,i+1::dim] += np.diag(np.ones(dim)*c*(self.Coefficients[0]+1))
        
        Meas = np.zeros((dim, (order+1)*dim))
        Meas[:,::order+1] = np.diag(np.ones(dim))
        
        state_function = lambda x: np.dot(Matrix,x)
        meas_function = lambda x: np.dot(Meas,x)
        
        super(MA,self).__init__(dim*(order+1), 0, dim,
                                state_function=state_function,
                                meas_function=meas_function)
        

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


class Kalman(object):
    def __init__(self, a, b, c, g, q, r, x_0, p_0, **kwargs):
        '''
        The Kalman-Filter evaluates data concerning a physical model mainly implemented in the matrices A,B and C.
        Assuming all measurement-errors are gaussian the filter predicts new measurement values from preceding values.
        Also it updates its predictions with the data and produces new believed values for the measurement,
        which are also gaussian, but have a smaller distribution.

        :param a: Array-like nxn dimensional matrix, describing the evolution of a new state out of an old state.
        :param b: Array-like nxm dimensional matrix, describing, how a new state is influenced by the control sate u.
        :param c: Array-like nxr dimensional matrix, describing how a state projects on a measurement.
        :param q: Array-like nxn dimensional matrix. Covariance matrix of the prediction.
        :param r: Array-like rxr dimensional matrix. Covariance matrix of the measurement.
        :param x_0: Array-like n dimensional vector. Start value of the state vector.
        :param p_0: Array-like nxn dimensional matrix. Start value of the believed uncertainty.
        '''
        super(Kalman, self).__init__()
        self.A = np.array(a)
        self.B = np.array(b)
        self.C = np.array(c)
        self.G = np.array(g)
        self.Q = np.array(q)
        self.Q_0 = np.array(q)
        self.R = np.array(r)
        self.R_0 = np.array(r)
        self.x_0 = np.array(x_0)
        self.p_0 = np.array(p_0)
        self.x = {0: self.x_0}
        self.p = {0: self.p_0}
        self.predicted_x = {0: self.x_0}
        self.predicted_p = {0: self.p_0}

        self.z = kwargs.pop('measurement', {})
        self.u = kwargs.pop('movement', {})

        self.K = self.k()

        self.Predict_Count = 0

    def downfilter(self):
        t = max(self.x.keys())
        self.z.pop(t, default=None)
        self.u.pop(t, default=None)
        self.x.pop(t, default=None)
        self.p.pop(t, default=None)
        self.predicted_x.pop(t, default=None)
        self.predicted_p.pop(t, default=None)

    def downdate(self):
        t = max(self.x.keys())
        self.z.pop(t, None)
        self.x.update({t: self.predicted_x[t]})
        self.p.update({t: self.predicted_p[t]})
        self.u.pop(t, None)

    def downdict(self):
        t = max(self.x.keys())
        self.u.pop(t, default=None)
        self.x.pop(t, default=None)
        self.p.pop(t, default=None)
        self.predicted_x.pop(t, default=None)
        self.predicted_x.pop(t, default=None)

    def predict(self, u):
        '''Prediction part of the Kalman Filtering process for one step'''
        if len(self.x.keys()) == 0:
            t = 0
        else:
            t = max(self.x.keys())+1

        self.Q = self.Q_0

        if self.u is None:
            self.u = {t: u}
        else:
            self.u.update({t: u})

        x_ = np.dot(self.A, self.x[t-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.p[t-1]), self.A.transpose()) + np.dot(np.dot(self.G, self.Q), self.G.T)

        self.x.update({t: x_})
        self.p.update({t: p_})
        self.predicted_x.update({t: x_})
        self.predicted_p.update({t: p_})

        self.Predict_Count += 1
        return x_, p_

    def update(self, z, reset=True):
        '''Updating part of the Kalman Filtering process for one step'''
        if len(self.x.keys()) == 0:
            t = 0
        else:
            t = max(self.x.keys())

        self.K = self.k()
        self.R = self.R_0
        if self.z is None:
            self.z = {t: z}
        else:
            self.z.update({t: z})
        y = z - np.dot(self.C, self.x[t])

        x_ = self.x[t] + np.dot(self.K, y)
        p_ = self.p[t] - np.dot(np.dot(self.K, self.C), self.p[t])

        self.x.update({t: x_})
        self.p.update({t: p_})

        if reset:
            self.Predict_Count = 0
        return x_, p_

    def filter(self, u, z):
        '''Actual Kalman filtering process for one single step'''
        self.predict(u)
        x, p = self.update(z)
        return x, p

    def k(self):
        t = max(self.x.keys())
        return np.dot(np.dot(self.p[t], self.C.transpose()),
                      np.linalg.inv(np.dot(np.dot(self.C, self.p[t]), self.C.transpose()) + self.R))

    def log_probability(self):
        x = np.array(self.x.values())
        z = np.array(self.z.values())
        if self.z is None:
            return 0
        n = min(x.shape[0], z.shape[0])
        if n == 0:
            return 0

        tot = 0
        for i in range(n):
            xx = z[i]-np.dot(self.C, x[i])
            cov = self.R
            tot += -0.5*(np.log(np.linalg.det(2 * np.pi * cov)) + np.dot(np.dot(xx, np.linalg.inv(cov)), xx.T))
        return tot

    def fit(self, u, z):
        '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''

        u = np.array(u)
        z = np.array(z)
        assert u.shape[0] == z.shape[0]

        for i in range(z.shape[0]):
            x, p = self.predict(u[i])
            x, p = self.update(z[i])

        return np.array(self.x.values(), dtype=float),\
               np.array(self.p.values(), dtype=float),\
               np.array(self.predicted_x.values(), dtype=float),\
               np.array(self.predicted_p.values(), dtype=float)


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


class Particle(object):
    def __init__(self, n, a, b, c, q, r, x0):
        super(Particle, self).__init__()
        self.N = int(n)
        self.A = np.array(a)
        self.B = np.array(b)
        self.C = np.array(c)
        self.Q = np.array(q)
        self.R = np.array(r)
        self.X0 = np.array(x0)
        self.X = {}
        self.X_err = {}
        self.predicted_X = {}
        self.predicted_X_err = {}

        self.Particles = {}
        self.History = {}
        self.Weights = {}

        for n in range(self.N):
            self.Particles.update({n: self.X0[np.random.randint(0, self.X0.shape[0])]})
            self.Weights.update({n: 1./self.N})

        if len(self.X.keys()) == 0:
            t = 0
        else:
            t = max(self.X.keys())+1

        self.X.update({t: np.mean(self.Particles.values(), axis=0)})
        self.X_err.update({t: np.std(self.Particles.values(), axis=0)})
        # self.history_update()

    def history_update(self):
        self.History.update({len(self.History.keys()): [np.mean(self.Particles.values(), axis=0),
                                                        np.std(self.Particles.values(), axis=0),
                                                        np.mean(self.Weights.values(), axis=0),
                                                        np.std(self.Weights.values(), axis=0)]})

    def predict(self, u):
        if len(self.X.keys()) == 0:
            t = 0
        else:
            t = max(self.X.keys())+1
        for i in self.Particles.keys():
            self.Particles.update({i: ss.multivariate_normal(mean=np.dot(self.A, self.Particles[i]) + np.dot(self.B, u),
                                                             cov=self.Q).rvs()})

        self.predicted_X.update({t : np.mean(self.Particles.values(), axis=0)})
        self.predicted_X_err.update({t : np.std(self.Particles.values(), axis=0)})
        return self.Particles

    def update(self, z):
        if len(self.X.keys()) == 0:
            t = 0
        else:
            t = max(self.X.keys())+1

        dist = ss.multivariate_normal(mean=np.zeros_like(z), cov=self.R)
        for i in self.Weights.keys():
            dist.mean = np.dot(self.C, self.Particles[i])
            self.Weights.update({i: dist.pdf(z)})

        w = np.sum(self.Weights.values())
        if w > 0:
            for i in self.Weights.keys():
                self.Weights[i] *= 1./w
        else:
            raise ValueError('Sum of weights is smaller than zero. No further Particle computation possible.')

        idx = np.sum(np.array(np.tile(np.cumsum(self.Weights.values()), self.N).reshape((self.N, -1)) <
                     np.tile(np.random.rand(self.N), self.N).reshape((self.N, -1)).T, dtype=int), axis=1)
        print(len(set(idx)))
        values = self.Particles.values()
        for i, j in enumerate(idx):
            try:
                self.Particles.update({i: values[j]})
            except IndexError:
                self.Particles.update({i: values[j-1]})

        self.X.update({t: np.mean(self.Particles.values(), axis=0)})
        self.X_err.update({t: np.std(self.Particles.values(), axis=0)})

        return self.Particles

    def fit(self, u, z):
        for i in range(len(u)):

            self.predict(u[i])

            try:
                self.update(z[i])
            except ValueError:
                break
            self.History.update({i: self.Particles.values()})

        return np.array(self.X.values()),\
               np.array(self.X_err.values()),\
               np.array(self.predicted_X.values()),\
               np.array(self.predicted_X_err.values())
