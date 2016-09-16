from __future__ import print_function, division
import numpy as np
import numpy.ma as ma
from scipy import linalg as LA



class FilterOpt(object):
    def __init__(self, filter):
        super(FilterOpt).__init__(object)
        self.Filter = filter

    def aberrations(self, variable, values, data):
        err = []
        for v in values:
            self.Filter.variables.update({variable:v})
            predictions, believes = self.Filter.filter(data)
            err.append(np.sqrt(np.mean((predictions-data)**2)))
        return err

    def optimize(self, variables, minVals, maxVals, steps, data):
        if len(variables) == 1:
            err = self.aberrations(variables, np.arange(start=minVals, stop=maxVals, step=steps), data)
            return err
        else:
            var = variables.pop()
            min_val = minVals.pop()
            max_val = maxVals.pop()
            step = steps.pop()
            values = np.arange(start=min_val,stop=max_val,step=step)
            err = []
            for v in values:
                self.filter.update({var:v})
                err.append(self.optimize(variables, minVals, maxVals, steps, data))
            err = np.array(err)
            optimal = np.argmin(np.flatten(err))
            optimal = np.unravel_index(optimal, err.shape)
            opt=[]
            for v in variables:
                raise NotImplementedError()
            return optimal


class Filter(object):
    def __init__(self, constants={}, variables={}):
        super(Filter).__init__(object)
        self.Constants = constants
        self.Variables = variables

    def predict(self, data):
        return data

    def update(self, data):
        return data

    def filter(self, data):
        predictions = self.predict(data)
        data = self.update(predictions)
        return predictions, data

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

        X = []
        P = []
        for kal in self.Filters:
            X.append(kal.x[max(kal.x.keys())])
            P.append(kal.p[max(kal.x.keys())])
        return X, P

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import clickpoints
    import scipy.optimize as opt

    db = clickpoints.DataFile("click0.cdb")
    tracks = db.getTracks()
    points = []
    for t in tracks:
        t = t.points_corrected
        if t.shape[0] > 2:
            points.append(t)

    measurements = points[-5][:]
    # measurements = np.array([range(40), range(40)]).T*10
    # measurements = np.random.normal(measurements, scale=2)
    #measurements[:, 1] = 0
    v = measurements[1]-measurements[0]
    X = np.array([measurements[0][0], v[0], measurements[0][1], v[1]])  # initial state (location and velocity)
    U = np.zeros([measurements.shape[0], 4])  # external motion

    A = np.array([[1., 1., 0., 0.],
                  [0., 1, 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 0., 0., 1]])  # next state function

    B = np.zeros_like(A)  # next state function for control parameter
    C = np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.]])  # measurement function
    G = np.array([[0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 1]])

    def loger(uncty):

        #uncty=20.7

        P = np.diag([uncty, uncty, uncty, uncty])  # initial uncertainty
        Q = np.diag([uncty, uncty])  # Prediction uncertainty
        R = np.diag([10, 10])  # Measurement uncertainty

        lag = 0.933355170212
        A = np.array([[1., 1., 0., 0.],
                      [0., 1, 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1]])  # next state function

        kal = Kalman_Mehra(A, B, C, G, Q, R, X, P, lag=5, learn_time=5)

        X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

        return -1*kal.log_probability()

    #optimal = opt.minimize_scalar(loger, bounds=[5, 39], method='Bounded')

    ucty = 20#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 10
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    kal = Kalman_Mehra(A, B, C, G, Q, R, X, P, lag=5, learn_time=5)

    X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

    Bel = X_Post.T[::2].T
    pred_err = np.sqrt(np.abs(np.array([np.linalg.eigvals(p)[::2] for p in Pred_err])))
    #pred_err = np.sqrt(np.abs(np.array([P.T[0,0],P.T[2,2]])))
    pred = Pred.T[::2].T

    plt.errorbar(pred.T[0], pred.T[1], xerr=pred_err.T[0], yerr=pred_err.T[1], color='r')
    #plt.plot(X_Post.T[0], X_Post.T[1], '-go')
    plt.errorbar(measurements.T[0], measurements.T[1], xerr=10, yerr=10, color='b')

    plt.axis('equal')
    plt.show()


    start_point = np.array([p[0] for p in points])
    second_point = np.array([p[1] for p in points])
    V = second_point - start_point
    X = np.array([start_point.T[0], V.T[0], start_point.T[1], V.T[1]]).T
    P = [P for v in V]
    MultiKal = MultiKalman(A, B, C, G, Q, R, X, P, lag=5, learn_time=5)

    maxLen = max([p.shape[0] for p in points])

    timeline = np.array([[p[0] for p in points]])

    for i in range(maxLen):
        timepoint = []
        for p in points:
            try:
                meas = p[i]
            except IndexError:
                meas = [np.nan, np.nan]
            if np.random.rand < 0.5:
                timepoint.append(meas)
            else:
                meas=[meas]
                meas.extend(timepoint)
                timepoint = meas
        timeline = np.vstack((timeline, [timepoint]))

    # timeline = timeline[:45]

    UU = np.zeros((timeline.shape[0], timeline.shape[1], 4))

    X_Post, P_Post, Pred, Pred_err = MultiKal.fit(UU, timeline)

    for i in range(5):
        plt.errorbar(Pred[i].T[0], Pred[i].T[2], color='r')#, xerr=P_Post.T[0, 0][i], yerr=P_Post.T[2, 2][i], color="b")
        plt.errorbar(np.array(MultiKal.Filters[i].z.values()).T[0], np.array(MultiKal.Filters[i].z.values()).T[1], color='b')#, xerr=Pred_err.T[0,0][i], yerr=Pred_err.T[2,2][i], color="r")
        plt.axis('equal')
    plt.show()
    # plt.savefig("/home/alex/Desktop/number%s.png"%i)
    # plt.clf()
    # for i in range(5):
    #     plt.errorbar(points[i].T[0], points[i].T[1], color='g')#, xerr=P_Post.T[0, 0][i], yerr=P_Post.T[2, 2][i], color="b")
    #     plt.axis('equal')
    #     plt.savefig("/home/alex/Desktop/true%s.png"%i)
    #     plt.clf()
        #plt.errorbar(points[3].T[0], points[3].T[1], color='g')#, xerr=Pred_err.T[0,0][i], yerr=Pred_err.T[2,2][i], color="r")
    #plt.plot(X_Post.T[0].T,X_Post.T[1].T)
    #plt.plot(Pred.T[0].T,Pred.T[1].T)



