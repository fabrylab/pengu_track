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

    model = VariableSpeed(2)
    v = measurements[1]-measurements[0]
    X = np.array([measurements[0, 0], v[0], measurements[0, 1], v[1]])
    # X = np.dot(model.Measurement_Matrix.T, measurements[0])
    U = np.zeros((measurements.shape[0], model.Control_dim))
    A = model.State_Matrix
    B = model.Control_Matrix
    C = model.Measurement_Matrix
    G = model.Evolution_Matrix

    ucty = 30**0.5#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 30**0.5
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)

    start_point = np.array([p[0] for p in points])
    second_point = np.array([p[1] for p in points])
    V = second_point - start_point
    X = np.array([start_point.T[0], V.T[0], start_point.T[1], V.T[1]]).T
    P = [P for v in V]
    # MultiKal = MultiFilter(ParticleFilter, model, X, n=1000, meas_dist=Meas_Dist, state_dist=State_Dist)
    MultiKal = MultiFilter(KalmanFilter, model, X, np.array([vxvy_uncty, vxvy_uncty]),
                           np.array([meas_uncty,meas_uncty]), meas_dist=Meas_Dist, state_dist=State_Dist, lag=5)

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


    UU = np.zeros((timeline.shape[0], timeline.shape[1], 0))

    X_Post, P_Post, Pred, Pred_err = MultiKal.fit(UU, timeline)


if __name__ == '__main__':

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # import matplotlib.animation as manimation
    from matplotlib.patches import Ellipse

    # FFMpegWriter = manimation.writers['ffmpeg']
    # metadata = dict(title='Movie Test', artist='Matplotlib',
    #                 comment='Movie support!')
    # writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=1000000)

    fig = plt.figure()
    fig.set_size_inches(40, 20)

    # plt.axis('equal')

    # with writer.saving(fig, "writer_test2.mp4", X.shape[0]):
    for i in X_Post.keys()[1:-1]:
        print(i)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(500, 3500)
        ax.set_ylim(500, 2000)
        ax.plot(timeline.T[0].T[:i], timeline.T[1].T[:i])#, xerr=meas_uncty, yerr=meas_uncty, c='b')
        for k in X_Post[i].keys():
            new_el = Ellipse(Pred[i+1][k][::2], np.sqrt(Pred_err[i+1][k][0, 0]), np.sqrt(Pred_err[i+1][k][1, 1]))
            old_el = Ellipse(Pred[i][k][::2], np.sqrt(Pred_err[i][k][0, 0]), np.sqrt(Pred_err[i][k][1, 1]))
            ax.add_artist(new_el)
            ax.add_artist(old_el)
            new_el.set_clip_box(ax.bbox)
            old_el.set_clip_box(ax.bbox)
            new_el.set_facecolor('none')
            old_el.set_facecolor('none')
            new_el.set_edgecolor('r')
            old_el.set_edgecolor('k')
        # writer.grab_frame()
        plt.savefig('./movie/frame%03d.png'%i)
        ax.clear()

