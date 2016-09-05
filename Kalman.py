import numpy as np


class Kalman:
    def __init__(self, f, b, q, r, h):
        self.F = np.array(f)
        self.B = np.array(b)
        self.Q = np.array(q)
        self.R = np.array(r)
        self.H = np.array(h)

    def predict(self, x, p, u):
        x_ = np.dot(self.F, x) + np.dot(self.B, u)
        p_ = np.dot(np.dot(self.F, p), self.F.transpose()) + self.Q
        return x_, p_

    def update(self, x, p, z):
        y = z - np.dot(self.H, x)
        s = np.dot(np.dot(self.H, p), self.H.transpose()) + self.R
        k = np.dot(np.dot(p, self.H.transpose()), np.linalg.inv(s))
        x_ = x + np.dot(k, y)
        p_ = p - np.dot(np.dot(k, s), k.transpose())
        return x_, p_

    def filter(self, x, p, u, z):
        x, p = self.predict(x, p, u)
        x, p = self.update(x, p, z)
        print x,p
        return x, p


if __name__ == '__main__':
    measurements = [1, 2, 3]
    X = np.array([[0.], [0.]])  # initial state (location and velocity)
    P = np.array([[1000., 0.], [0., 1000.]])  # initial uncertainty
    U = np.array([[0.], [0.]])  # external motion
    F = np.array([[1., 1.], [0, 1.]])  # next state function
    H = np.array([[1., 0.]])  # measurement function
    R = np.array([[1.]])  # measurement uncertainty
    I = np.array([[1., 0.], [0., 1.]])  # identity matrix
    Q = np.array([[0.], [0.]])

    kal = Kalman(F, I, Q, R, H)

    for m in measurements:
        X, P = kal.filter(X, P, U, m)

    print X, '\n', P

