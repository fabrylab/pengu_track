if __name__ == '__main__':
    import numpy as np
    from Filters import ParticleFilter
    from Filters import AdvancedKalmanFilter as KalmanFilter
    from Filters import MultiFilter
    from Models import VariableSpeed
    import matplotlib.pyplot as plt
    import clickpoints
    import scipy.optimize as opt
    import scipy.stats as ss

    db = clickpoints.DataFile("click0.cdb")
    tracks = db.getTracks()
    points = []
    for t in tracks:
        t = t.points_corrected
        if t.shape[0] > 2:
            points.append(t)

    measurements = points[-2][:]
    # D = 2*np.random.randint(0, 2, size=100)-1
    # X = [[0, 0]]
    # for i, d in enumerate(D):
    #     X.append([i, X[-1][1]+np.random.normal(loc=d, scale=0.1)])
    # measurements = np.array(X)
    # measurements = np.array([range(40), range(40)]).T*10
    # measurements = np.random.normal(measurements, scale=2)
    # measurements[:, 1] = 0

    model = VariableSpeed(2)
    v = measurements[1]-measurements[0]
    X = np.array([measurements[0, 0], v[0], measurements[0, 1], v[1]])
    # X = np.dot(model.Measurement_Matrix.T, measurements[0])
    U = np.zeros((measurements.shape[0], model.Control_dim))
    A = model.State_Matrix
    B = model.Control_Matrix
    C = model.Measurement_Matrix
    G = model.Evolution_Matrix
    #

    # v = measurements[1]-measurements[0]
    # X = np.array([measurements[0][0], v[0], measurements[0][1], v[1]])  # initial state (location and velocity)
    # U = np.zeros([measurements.shape[0], 4])  # external motion

    # A = np.array([[1., 1., 0., 0.],
    #               [0., 1., 0., 0.],
    #               [0., 0., 1., 1.],
    #               [0., 0., 0., 0.]])  # next state function
    #
    # B = np.zeros_like(A)  # next state function for control parameter
    # C = np.array([[1., 0., 0., 0.],
    #               [0., 0., 1., 0.]])  # measurement function
    # G = np.array([[0, 0],
    #               [1, 0],
    #               [0, 0],
    #               [0, 1]])

    ucty = 100#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 10
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    # Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    # Q = np.diag([vxvy_uncty, vxvy_uncty, vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    # # Part = Particle(100, A, B, C, Q, R, [X])
    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)
    #
    # Part = ParticleFilter(model, X, n=1000, meas_dist=Meas_Dist, state_dist=State_Dist)
    # # kal = KalmanFilter(model, X, np.array([vxvy_uncty,vxvy_uncty]), np.array([meas_uncty,meas_uncty]))
    # X, X_err, Pred, Pred_err = Part.fit(U, measurements)
    # # X, X_err, Pred, Pred_err = kal.fit(U[1:], measurements[1:])
    # # X_err = np.array([np.diag(x) for x in X_err])
    # # Pred_err = np.array([np.diag(p) for p in Pred_err])
    #
    # # plt.errorbar(X.T[0], X.T[2], xerr=X_err.T[0], yerr=X_err.T[2], c='g')
    # plt.errorbar(Pred.T[0], Pred.T[2], xerr=Pred_err.T[0], yerr=Pred_err.T[2], c='r')
    # plt.errorbar(measurements.T[0], measurements.T[1], xerr=meas_uncty, yerr=meas_uncty, c='b')
    # plt.axis('equal')
    # plt.show()

    start_point = np.array([p[0] for p in points])
    second_point = np.array([p[1] for p in points])
    V = second_point - start_point
    X = np.array([start_point.T[0], V.T[0], start_point.T[1], V.T[1]]).T
    P = [P for v in V]
    # MultiKal = MultiFilter(ParticleFilter, model, X, n=1000, meas_dist=Meas_Dist, state_dist=State_Dist)
    MultiKal = MultiFilter(KalmanFilter, model, X, np.array([vxvy_uncty, vxvy_uncty]),
                           np.array([meas_uncty,meas_uncty]), meas_dist=Meas_Dist, state_dist=State_Dist)
    # MultiKal = MultiKalman(A, B, C, G, Q, R, X, P, lag=5, learn_time=5)

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
                meas = [meas]
                meas.extend(timepoint)
                timepoint = meas
        timeline = np.vstack((timeline, [timepoint]))

    # timeline = timeline[:45]

    UU = np.zeros((timeline.shape[0], timeline.shape[1], 0))

    X_Post, P_Post, Pred, Pred_err = MultiKal.fit(UU, timeline)

    XXX = []
    for k in MultiKal.Filters.keys():
        XX = []
        for i in X_Post.keys():
            try:
                XX.append(X_Post[i][k])
            except KeyError:
                pass
        XXX.append(np.array(XX))
    for XX in XXX:
        plt.plot(XX.T[0], XX.T[2])
    plt.axis('equal')
    fig = plt.figure()
    for p in points:
        plt.plot(p.T[0], p.T[1])
    plt.axis('equal')
    plt.show()
    # for i in range(5):
    #     plt.errorbar(Pred[i].T[0], Pred[i].T[2], color='r')#, xerr=P_Post.T[0, 0][i], yerr=P_Post.T[2, 2][i], color="b")
    #     plt.errorbar(np.array(MultiKal.Filters[i].z.values()).T[0], np.array(MultiKal.Filters[i].z.values()).T[1], color='b')#, xerr=Pred_err.T[0,0][i], yerr=Pred_err.T[2,2][i], color="r")
    #     plt.axis('equal')
    # # plt.show()
    #     plt.savefig("/home/alex/Desktop/number%s.png"%i)
    #     plt.clf()

