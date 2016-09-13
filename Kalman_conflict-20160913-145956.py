from __future__ import print_function, division
import numpy as np
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


# class KalmanMehra(object):
#     def __init__(self, a, b, c, g, q, r, x_0, p_0):
#         '''
#         The Kalman-Filter evaluates data concerning a physical model mainly implemented in the matrices A,B and C.
#         Assuming all measurement-errors are gaussian the filter predicts new measurement values from preceding values.
#         Also it updates its predictions with the data and produces new believed values for the measurement,
#         which are also gaussian, but have a smaller distribution.
#
#         :param a: Array-like nxn dimensional matrix, describing the evolution of a new state out of an old state.
#         :param b: Array-like nxm dimensional matrix, describing, how a new state is influenced by the control sate u.
#         :param c: Array-like nxr dimensional matrix, describing how a state projects on a measurement.
#         :param q: Array-like nxn dimensional matrix. Covariance matrix of the prediction.
#         :param r: Array-like rxr dimensional matrix. Covariance matrix of the measurement.
#         :param x_0: Array-like n dimensional vector. Start value of the state vector.
#         :param p_0: Array-like nxn dimensional matrix. Start value of the believed uncertainty.
#         '''
#         super(Kalman).__init__(object)
#         self.A = np.array(a)
#         self.B = np.array(b)
#         self.C = np.array(c)
#         self.G = np.array(g)
#         self.Q = np.array(q)
#         self.Q_0 = np.array(q)
#         self.R = np.array(r)
#         self.R_0 = np.array(r)
#         self.x_0 = np.array(x_0)
#         self.p_0 = np.array(p_0)
#         self.x = [self.x_0]
#         self.p = [self.p_0]
#
#         self.K = self.k()
#
#     def predict(self, u):
#         '''Prediction part of the Kalman Filtering process for one step'''
#         x_ = np.dot(self.A, self.x[-1]) + np.dot(self.B, u)
#         p_ = np.dot(np.dot(self.A, self.p[-1]), self.A.transpose()) + np.dot(np.dot(self.G, self.Q), self.G.T)
#         self.x.append(x_)
#         self.p.append(p_)
#         return x_, p_
#
#     def update(self, z):
#         '''Updating part of the Kalman Filtering process for one step'''
#         y = z - np.dot(self.C, self.x[-1])
#
#         x_ = self.x[-1] + np.dot(self.K, y)
#         p_ = self.p[-1] - np.dot(np.dot(self.K, self.C), self.p[-1])
#
#         self.x[-1] = x_
#         self.p[-1] = p_
#         return x_, p_
#
#     def filter(self, u, z):
#         '''Actual Kalman filtering process for one single step'''
#         self.predict(u)
#         x, p = self.update(z)
#         return x, p
#
#     def k(self):
#         return np.dot(np.dot(self.p[-1], self.C.transpose()),
#                       np.linalg.inv(np.dot(np.dot(self.C, self.p[-1]), self.C.transpose()) + self.R))
#
#     def k_hat(self, z):
#         a_cross = self.pseudo_inverse(self.A_tilde(len(z)))
#         cc = self.c_tilde(z)
#         return self.k() + np.dot(np.dot(a_cross, cc), np.linalg.inv(self.c_hat(0, z)))
#
#     def k_opt(self, z):
#         r = self.R_0
#         try:
#             k_hat = self.k_hat(z)
#         except np.linalg.LinAlgError:
#             k_hat = self.k()
#             print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
#         try:
#             mh_hat = self.mh_hat(z)
#         except np.linalg.LinAlgError:
#             print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
#             return k_hat
#
#         r = self.r_hat(z)
#         delta_m1_hat = self.p[-1]-self.p[-2]
#
#         i = 0
#         while True:
#             ikh = np.identity(self.C.shape[1])-np.dot(k_hat, self.C)
#             k1k0 = k_hat - self.k()
#
#             j = 0
#             while True:
#                 j += 1
#                 delta_m1_hat_new = np.dot(np.dot(np.dot(np.dot(self.A,
#                                                         ikh),
#                                                  delta_m1_hat),
#                                           ikh.T),
#                                           self.A.T) - np.dot(np.dot(np.dot(np.dot(self.A,
#                                                                                   k1k0),
#                                                                            self.c_hat(0, z)),
#                                                                     k1k0.T),
#                                                             self.A.T)
#                 if np.allclose(delta_m1_hat, delta_m1_hat_new):
#                     break
#                 else:
#                     delta_m1_hat = delta_m1_hat_new
#                     if j > 100:
#                         break
#             m2ht_hat = mh_hat + np.dot(delta_m1_hat, self.C.T)
#             try:
#                 k2_hat = np.dot(m2ht_hat, np.linalg.inv(np.dot(self.C, m2ht_hat) + r))
#             except np.linalg.LinAlgError:
#                 return self.k()
#                 print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
#             i += 1
#             if i > 1000:
#                 break
#             if np.linalg.norm(k2_hat) > np.linalg.norm(k2_hat-k_hat) or\
#                 np.linalg.norm(self.p[-2]) > np.linalg.norm(delta_m1_hat):
#                 print(i)
#                 break
#         return k2_hat
#
#     def v(self, i, z_i):
#         return z_i-np.dot(np.dot(self.C, self.A), self.x[i-1])
#
#     def c_hat(self, k, z):
#         c = np.sum(np.array([
#                             np.tensordot(self.v(i, z[i]), self.v(i - k, z[i - k]).transpose(), axes=0)
#                             for i in range(len(z))
#                             ]), axis=0)
#         return np.array(c/len(z))
#
#     @staticmethod
#     def pseudo_inverse(m):
#         m = np.dot(np.linalg.inv(np.dot(m.transpose(), m)), m.transpose())
#         return m
#
#     def A_tilde(self, n):
#         a = np.identity(self.A.shape[0])
#         a_ = np.array(a)
#         a = np.dot(self.C, np.dot(a, self.A))
#         for i in range(n-1):
#             a_ = np.dot(np.dot(self.A, (np.identity(self.A.shape[0]))-np.dot(self.k(), self.C)), a_)
#             a = np.vstack((a, np.dot(self.C, np.dot(a_, self.A))))
#         return a
#
#     def c_tilde(self, z):
#         c = self.c_hat(0, z)
#         for i in range(len(z)-1):
#             c = np.vstack((c, self.c_hat(i, z)))
#         return c
#
#     def mh_hat(self, z):
#         cc = self.c_tilde(z)
#         a_cross = self.pseudo_inverse(self.A_tilde(len(z)))
#
#         return np.dot(self.k(), self.c_hat(0, z)) + np.dot(a_cross, cc)
#
#     def r_hat(self, z):
#         try:
#             cc = self.c_tilde(z)
#             a_cross = self.pseudo_inverse(self.A_tilde(len(z)))
#             mh_hat = np.dot(self.k(), self.c_hat(0, z)) + np.dot(a_cross, cc)
#
#             r = self.c_hat(0, z) - np.dot(self.C, mh_hat)
#         except np.linalg.LinAlgError:
#             r = self.R
#             print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
#         return r
#
#     def fit(self, u, z, opt=''):
#         '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
#         It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
#         xout = []
#         pout = []
#         pred_out = []
#         pred_out_err = []
#         u = np.array(u)
#         z = np.array(z)
#         assert u.shape[0] == z.shape[0]
#         for i in range(z.shape[0]):
#             if opt == 'Mehra' and i > 5000000:
#                 self.R = self.r_hat(z[:i+1])
#                 self.K = self.k_opt(z[:i+1])
#             else:
#                 self.R =self.R_0
#                 self.K = self.k()
#             print(self.R)
#             print(self.K)
#             if np.any(self.R != self.R):
#                 break
#             x, p = self.predict(u[i])
#             pred_out.append(x)
#             pred_out_err.append(p)
#             x, p = self.update(z[i])
#             xout.append(x)
#             pout.append(p)
#         return np.array(xout), np.array(pout), np.array(pred_out), np.array(pred_out_err)


class Kalman(object):
    def __init__(self, a, b, c, g, q, r, x_0, p_0):
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
        super(Kalman).__init__(object)
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
        self.x = [self.x_0]
        self.p = [self.p_0]

        self.K = self.k()

    def predict(self, u):
        '''Prediction part of the Kalman Filtering process for one step'''
        x_ = np.dot(self.A, self.x[-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.p[-1]), self.A.transpose()) + np.dot(np.dot(self.G, self.Q), self.G.T)
        self.x.append(x_)
        self.p.append(p_)
        return x_, p_

    def update(self, z):
        '''Updating part of the Kalman Filtering process for one step'''
        y = z - np.dot(self.C, self.x[-1])

        x_ = self.x[-1] + np.dot(self.K, y)
        p_ = self.p[-1] - np.dot(np.dot(self.K, self.C), self.p[-1])

        self.x[-1] = x_
        self.p[-1] = p_
        return x_, p_

    def filter(self, u, z):
        '''Actual Kalman filtering process for one single step'''
        self.predict(u)
        x, p = self.update(z)
        return x, p

    def k(self):
        return np.dot(np.dot(self.p[-1], self.C.transpose()),
                      np.linalg.inv(np.dot(np.dot(self.C, self.p[-1]), self.C.transpose()) + self.R))

    def log_probability(self, x, z):
        assert len(x) == len(z)
        tot = 0
        for i in range(len(z)):
            xx = z[i]-np.dot(self.C, x[i])
            cov = self.R
            probability = np.linalg.det(2 * np.pi * cov) ** -0.5 * np.exp(-0.5 * np.dot(np.dot(xx, np.linalg.inv(cov)), xx.T))
            tot += np.log(probability)
        return tot

    def fit(self, u, z):
        '''Function to auto-evaluate all measurements z with control-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
        xout = []
        pout = []
        pred_out = []
        pred_out_err = []
        u = np.array(u)
        z = np.array(z)
        assert u.shape[0] == z.shape[0]
        for i in range(z.shape[0]):
            self.R = self.R_0
            self.Q = self.Q_0
            self.K = self.k()

            x, p = self.predict(u[i])
            pred_out.append(x)
            pred_out_err.append(p)

            qp = np.dot(C, np.array(pred_out).T).T  - z[:i+1]
            Q = np.cov(qp.T)

            if np.all(Q == Q):
                print(Q)
                self.Q = Q
            x, p = self.update(z[i])
            xout.append(x)
            pout.append(p)

        return np.array(xout), np.array(pout), np.array(pred_out), np.array(pred_out_err)


class Kalman_Mehra(Kalman):
    def __init__(self, lag=1, measurement=[], *args):
        super(Kalman_Mehra, self).__init__(*args)
        self.Lag = np.assscalar(lag)
        self.z = np.array(measurement, dtype=float)

    def v(self, i):
        return self.z[i]-np.dot(np.dot(self.C, self.A), self.x[i-1])

    def c_hat(self, k):
        n = self.z.shape[0]
        c = np.sum(np.array([
                            np.tensordot(self.v(i, self.z[i]), self.v(i - k, self.z[i - k]).transpose(), axes=0)
                            for i in range(k, n)
                            ]), axis=0)
        c2 = np.tensordot(self.z[:-k], self.z[k:].T, axes=1)
        return np.array(c/n)

    @staticmethod
    def pseudo_inverse(m):
        m = np.dot(np.linalg.inv(np.dot(m.transpose(), m)), m.transpose())
        return m

    def A_tilde(self):
        a = None
        n = self.z.shape[0]
        for i in range(n - self.Lag, n):
            if a:
                a_ = np.dot(np.dot(self.A, (np.identity(self.A.shape[0]))-np.dot(self.k(), self.C)), a_)
                a = np.vstack((a, a_))
            else:
                a = np.identity(self.A.shape[0])
                a_ = np.array(a, copy=True)
        return np.dot(self.C, np.dot(a, self.A))

    def c_tilde(self):
        c = None
        n = self.z.shape[0]
        for i in range(n - self.Lag, n):
            if c:
                c = np.vstack((c, self.c_hat(i)))
            else:
                c = self.c_hat(i)
        return c

    def mh_hat(self, z):
        cc = self.c_tilde()
        a_cross = self.pseudo_inverse(self.A_tilde())

        return np.dot(self.k(), self.c_hat(0)) + np.dot(a_cross, cc)

    def r_hat(self, z):
        try:
            cc = self.c_tilde(z)
            a_cross = self.pseudo_inverse(self.A_tilde(len(z)))
            mh_hat = np.dot(self.k(), self.c_hat(0, z)) + np.dot(a_cross, cc)

            r = self.c_hat(0, z) - np.dot(self.C, mh_hat)
        except np.linalg.LinAlgError:
            r = self.R
            print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
        return r

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import clickpoints
    import scipy.optimize as opt

    db = clickpoints.DataFile("click0.cdb")
    tracks = db.getTracks()
    points = []
    for t in tracks:
        points.append(t.points_corrected)

    #measurements = points[-5][:]
    measurements = np.array([range(40), range(40)]).T*10
    measurements = np.random.normal(measurements, scale=2)
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
                      [0., lag, 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., lag]])  # next state function

        kal = Kalman(A, B, C, G, Q, R, X, P)
        X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

        return -1*kal.log_probability(Pred, measurements)

    optimal = opt.minimize_scalar(loger, bounds=[5, 39], method='Bounded')

    ucty = 20#optimal['x']
    print(ucty)
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 20
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    kal = Kalman(A, B, C, G, Q, R, X, P)

    X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

    Bel = X_Post.T[::2].T
    pred_err = np.sqrt(np.abs(np.array([np.linalg.eigvals(p)[::2] for p in Pred_err])))
    #pred_err = np.sqrt(np.abs(np.array([P.T[0,0],P.T[2,2]])))
    pred = Pred.T[::2].T

    plt.errorbar(pred.T[0], pred.T[1], xerr=pred_err.T[0], yerr=pred_err.T[1], color='r')
    #plt.plot(X_Post.T[0], X_Post.T[1], '-go')
    plt.plot(measurements.T[0], measurements.T[1], '-bx')
    plt.axis('equal')
    plt.show()
