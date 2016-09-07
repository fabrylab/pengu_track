import numpy as np


class FilterOpt(object):
    def __init__(self, filter):
        super(FilterOpt).__init__(object)
        self.Filter = filter

    def aberrations(self, variable, values, data):
        err=[]
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
    def __init__(self, a, b, c, r, q, x_0, p_0):
        super(Kalman).__init__(object)
        self.A = np.array(a)
        self.B = np.array(b)
        self.C = np.array(c)
        self.R = np.array(r)
        self.Q = np.array(q)
        self.x_0 = np.array(x_0)
        self.p_0 = np.array(x_0)
        self.x = [self.x_0]
        self.p = [self.p_0]

    def predict(self, u):
        '''Prediction part of the Kalman Filtering process for one step'''
        x_ = np.dot(self.A, self.x[-1]) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.A, self.p[-1]), self.A.transpose()) + self.R
        self.x.append(x_)
        self.p.append(p_)
        return x_, p_

    def update(self, z):
        '''Updating part of the Kalman Filtering process for one step'''
        y = z - np.dot(self.C, self.x[-1])
        s = np.dot(np.dot(self.C, self.p[-1]), self.C.transpose()) - self.Q
        k = np.dot(np.dot(self.p[-1], self.C.transpose()), np.linalg.inv(s))
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

    def fit(self, u, z):
        '''Function to auto-evaluate all measurements z with controll-vectors u and starting probability p.
        It returns the believed values x, the corresponding probabilities p and the predictions x_tilde'''
        xout = []
        pout = []
        pred_out = []
        pred_out_err = []
        u = np.array(u)
        z = np.array(z)
        assert u.shape[0] == z.shape[0]
        for i in range(z.shape[0]):
            x, p = self.predict(u[i])
            pred_out.append(x)
            pred_out_err.append(p)
            x, p = self.update(z[i])
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

    measurements = points[-5][:-5]
    X = np.array([measurements[0][0], 0., measurements[0][1], 0.])  # initial state (location and velocity)
    ucty = (np.max(measurements.flatten()) - np.min(measurements.flatten())) / np.sqrt(measurements.shape[0])
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    U = np.zeros([measurements.shape[0], 4])  # external motion

    A = np.array([[1., 1., 0., 0.],
                  [0., .9, 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 0., 0., .9]])  # next state function

    B = np.zeros_like(A)  # next state function for control parameter
    C = np.array([[1., 0., 0., 0.],
                [0., 0., 1., 0.]])  # measurement function


    xy_uncty = [20]
    vxvy_uncty = [100]
    meas_uncty = [10]

    xy_uncty = 10.**np.arange(-5, 5, 1)
    vxvy_uncty = 10.**np.arange(-5, 5, 1)
    meas_uncty = [15.]#10.**np.arange(-3,3,1)
    err1=[]
    for xy in xy_uncty:
        err2=[]
        for vxvy in vxvy_uncty:
            err3=[]
            for meas in meas_uncty:
                R = np.diag([xy, vxvy, xy, vxvy])  # Prediction uncertainty
                Q = np.diag([meas, meas])  # Measurement uncertainty

                kal = Kalman(A, B, C, R, Q, X, P)

                X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

                err3.append(np.sqrt(np.mean((Pred.T[::2].T-measurements)**2)))
            err2.append(err3)
        err1.append(err2)
    err1=np.array(err1)
    I=np.argmin(np.array(err1).flatten())
    I=np.unravel_index(I, err1.shape)
    print I
    print xy_uncty[I[0]],vxvy_uncty[I[1]],meas_uncty[I[2]]


    xy_uncty = xy_uncty[I[0]]
    vxvy_uncty = vxvy_uncty[I[1]]
    meas_uncty = meas_uncty[I[2]]

    R = np.diag([xy_uncty, vxvy_uncty, xy_uncty, vxvy_uncty])  # Prediction uncertainty
    Q = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    kal = Kalman(A, B, C, R, Q, X, U)

    X_Post, P_Post, Pred, Pred_err = kal.fit(U, measurements)

    Bel = X_Post.T[::2].T
    Pred_err= np.array([Pred_err.T[0,0],Pred_err.T[2,2]]).T
    Pred = Pred.T[::2].T
    print Pred_err

    plt.errorbar(Pred.T[0], Pred.T[1], xerr=Pred_err.T[0], yerr=Pred_err.T[1])
    plt.plot(measurements.T[0], measurements.T[1], '-bx')
    plt.axis('equal')
    plt.show()
