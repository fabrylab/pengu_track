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

class Kalman(object):
    def __init__(self, a, b, c, q, r, x_0, p_0):
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
        self.Q = np.array(q)
        self.R = np.array(r)
        self.x_0 = np.array(x_0)
        self.p_0 = np.array(p_0)
        self.x = [self.x_0]
        self.p = [self.p_0]

    def predict(self, u):
        '''Prediction part of the Kalman Filtering process for one step'''
        x_ = np.dot(self.A, self.x[-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.p[-1]), self.A.transpose()) + self.Q
        self.x.append(x_)
        self.p.append(p_)
        return x_, p_

    def update(self, z, opt=''):
        '''Updating part of the Kalman Filtering process for one step'''
        y = z - np.dot(self.C, self.x[-1])
        s = np.dot(np.dot(self.C, self.p[-1]), self.C.transpose()) - self.R
        if opt == 'Mehra':
            try:
                k = self.k_opt(z)
            except(np.linalg.LinAlgError):
                k = self.k()
                print("Warning: optimal K could not be provided! (singular matrix)")
        else:
            k = self.k()
        x_ = self.x[-1] + np.dot(k, y)
        p_ = self.p[-1] - np.dot(np.dot(k, self.C), self.p[-1])
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
                      np.linalg.inv(np.dot(np.dot(self.C, self.p[-1]), self.C.transpose()) - self.R))

    def k_hat(self, z):
        a_cross = self.pseudo_inverse(self.A_tilde(len(z)))
        cc = self.c_tilde(z)
        return self.k() + np.dot(np.dot(a_cross, cc), np.linalg.inv(self.c_hat(0, z)))

    def k_opt(self, z):
        k_hat = self.k_hat(z)
        mh_hat = self.mh_hat(z)
        R = self.r_hat(z)
        delta_M1_hat = self.p[-1]-self.p[-2]

        i=0
        while True:
            ikh = np.identity(self.C.shape[1])-np.dot(k_hat, self.C)
            k1k0 = k_hat - self.k()
            delta_M1_hat = np.dot(np.dot(np.dot(np.dot(self.A,
                                                       ikh),
                                                delta_M1_hat),
                                         ikh.T),
                                  self.A.T) - np.dot(np.dot(np.dot(np.dot(self.A,
                                                                          k1k0),
                                                                   self.c_hat(0, z)),
                                                            k1k0.T),
                                                     self.A.T)

            m2ht_hat = mh_hat + np.dot(delta_M1_hat, self.C.T)
            k2_hat = np.dot(m2ht_hat, np.linalg.inv(np.dot(self.C, m2ht_hat) + R))
            i += 1
            if i > 1000:
                break
            if np.linalg.norm(k2_hat) > np.linalg.norm(k2_hat-k_hat) or\
                            np.linalg.norm(self.p[-2]) > np.linalg.norm(delta_M1_hat):
                break

        return k2_hat

    def v(self, i, z_i):
        return z_i-np.dot(np.dot(self.C, self.A), self.x[i-1])

    def c_hat(self, k, z):
        c = 0
        for i in range(len(z)):
            c += np.tensordot(self.v(i, z[i]), self.v(i-k, z[i-k]).transpose(), axes=0)
        return np.array(c/len(z))

    def pseudo_inverse(self, M):
        M = np.dot(np.linalg.inv(np.dot(M.transpose(), M)), M.transpose())
        return M

    def A_tilde(self, n):
        a = np.identity(self.A.shape[0])
        a_ = np.array(a)
        a = np.dot(self.C, np.dot(a, self.A))
        for i in range(n-1):
            a_ = np.dot(np.dot(self.A, (np.identity(self.A.shape[0]))-np.dot(self.k(), self.C)), a_)
            a = np.vstack((a, np.dot(self.C, np.dot(a_, self.A))))
        return a

    def c_tilde(self, z):
        c = self.c_hat(0, z)
        for i in range(len(z)-1):
            c = np.vstack((c, self.c_hat(i, z)))
        return c

    def mh_hat(self, z):
        cc = self.c_tilde(z)
        a_cross = self.pseudo_inverse(self.A_tilde(len(z)))

        return np.dot(self.k(), self.c_hat(0, z)) + np.dot(a_cross, cc)

    def r_hat(self, z):
        try:
            r = self.c_hat(0, z) - np.dot(self.C, self.mh_hat(z))
        except(np.linalg.LinAlgError):
            r = self.R
            print("The starting conditions of the uncertainty matrices are to bad. (Inversion Error)")
        return r

    def fit(self, u, z, opt=''):
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
            if opt == 'Mehra':
                self.R = self.r_hat(z[:i+1])
            x, p = self.predict(u[i])
            pred_out.append(x)
            x, p = self.update(z[i])
            pred_out_err.append(p)
            xout.append(x)
            pout.append(p)
        return np.array(xout), np.array(pout), np.array(pred_out), np.array(pred_out_err)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import clickpoints

    db = clickpoints.DataFile("click0.cdb")
    tracks = db.getTracks()
    points = []
    for t in tracks:
        points.append(t.points_corrected)

    measurements = points[-3][:-5]
    X = np.array([measurements[0][0], 0., measurements[0][1], 0.])  # initial state (location and velocity)
    ucty = (np.max(measurements.flatten()) - np.min(measurements.flatten())) / np.sqrt(measurements.shape[0])*1e3
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    U = np.zeros([measurements.shape[0], 4])  # external motion

    A = np.array([[1., 1., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 0., 0., 1.]])  # next state function

    B = np.zeros_like(A)  # next state function for control parameter
    C = np.array([[1., 0., 0., 0.],
                 [0., 0., 1., 0.]])  # measurement function

    xy_uncty = [ucty]
    vxvy_uncty = [ucty]
    meas_uncty = [10]

    #xy_uncty = 10.**np.arange(0.9, 1.3, 0.02)
    #vxvy_uncty = 10.**np.arange(2.7, 3.1, 0.2)
    #meas_uncty = [15.]#10.**np.arange(-3,3,1)
    err1 = []
    for xy in xy_uncty:
        err2 = []
        for vxvy in vxvy_uncty:
            err3 = []
            for meas in meas_uncty:
                Q = np.diag([xy, vxvy, xy, vxvy])  # Prediction uncertainty

                R = np.diag([meas, meas])  # Measurement uncertainty

                kal = Kalman(A, B, C, Q, R, X, P)

                X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

                err3.append(np.sqrt(np.mean((Pred.T[::2].T-measurements)**2)))
                #err3.append(np.mean(np.linalg.det(Pred_err)))
                #err3.append(np.mean(np.linalg.det(Pred_err))/np.sqrt(np.mean((Pred.T[::2].T-measurements)**2)))
            err2.append(err3)
        err1.append(err2)
    err1 = np.array(err1)
    I = np.argmin(np.array(err1).flatten())
    I = np.unravel_index(I, err1.shape)
    print(I)
    print(xy_uncty[I[0]],vxvy_uncty[I[1]],meas_uncty[I[2]])


    xy_uncty = xy_uncty[I[0]]
    vxvy_uncty = vxvy_uncty[I[1]]
    meas_uncty = meas_uncty[I[2]]

    Q = np.diag([xy_uncty, vxvy_uncty, xy_uncty, vxvy_uncty])  # Prediction uncertainty

    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    kal = Kalman(A, B, C, Q, R, X, P)

    X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements, opt='Mehra')

    Bel = X_Post.T[::2].T
    pred_err = np.sqrt(np.array([np.linalg.eigvals(p) for p in Pred_err.T[::2,::2].T]))
    Pred = Pred.T[::2].T

    plt.errorbar(Pred.T[0], Pred.T[1], xerr=pred_err.T[0], yerr=pred_err.T[1], color='r')
    plt.plot(Pred.T[0], Pred.T[1], '-ro')
    plt.plot(measurements.T[0], measurements.T[1], '-bx')
    plt.axis('equal')
    plt.show()

    R = kal.r_hat(measurements[:2])
    print(R)

