if __name__ == '__main__':

    import numpy as np
    from Filters import ParticleFilter
    from Filters import AdvancedKalmanFilter as KalmanFilter
    from Models import VariableSpeed
    import clickpoints
    import scipy.stats as ss
    from matplotlib.patches import Ellipse

    db = clickpoints.DataFile("click0.cdb")
    tracks = db.getTracks()
    points = []
    for t in tracks:
        t = t.points_corrected
        if t.shape[0] > 2:
            points.append(t)

    measurements = points[-1][:]
    model = VariableSpeed(2)#, damping=np.log(1/0.9)
    X = np.dot(model.Measurement_Matrix.T, measurements[0])
    U = np.zeros((measurements.shape[0], model.Control_dim))
    A = model.State_Matrix
    B = model.Control_Matrix
    C = model.Measurement_Matrix
    G = model.Evolution_Matrix

    ucty = 10.26#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 10
    P = np.diag([ucty, ucty, ucty, ucty])  # initial uncertainty
    Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)

    # Part = ParticleFilter(model, [X], n=10000, meas_dist=Meas_Dist, state_dist=State_Dist)
    kal = KalmanFilter(model, X, np.array([vxvy_uncty, vxvy_uncty]), np.array([meas_uncty,meas_uncty]), lag=10)
    # X, X_err, Pred, Pred_err = Part.fit(U, measurements)
    X, X_err, Pred, Pred_err = kal.fit(U[1:], measurements[1:])
    X_err = np.array([np.diag(x) for x in X_err])
    Pred_err = np.array([np.diag(p) for p in Pred_err])

if __name__ == '__main__':

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    from matplotlib.patches import Ellipse

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=1000000)

    fig = plt.figure()
    fig.set_size_inches(20, 9)

    # plt.axis('equal')

    with writer.saving(fig, "writer_test.mp4", X.shape[0]):
        for i, p in enumerate(Pred[1:]):
            print(i)
            ax = fig.add_subplot(111, aspect='equal')
            ax.set_xlim(np.amin(X.T[0])-100, np.amax(X.T[0])+100)
            ax.set_ylim(np.amin(X.T[2])-100, np.amax(X.T[2])+100)
            measured_plot, n, m = ax.errorbar(measurements[:i+1][0], measurements[:i+1][0], xerr=meas_uncty, yerr=meas_uncty, c='b')
            el = Ellipse(p[::2], Pred_err[i+1, 0], Pred_err[i+1, 2])
            el2 = Ellipse(Pred[i, ::2], Pred_err[i, 0], Pred_err[i, 2])
            measured_plot.set_data(measurements[:i+1].T[0], measurements[:i+1].T[1])
            ax.add_artist(el)
            el.set_clip_box(ax.bbox)
            el.set_facecolor('none')
            el.set_edgecolor('k')
            ax.add_artist(el2)
            el2.set_clip_box(ax.bbox)
            el2.set_facecolor('none')
            el2.set_edgecolor('r')
            writer.grab_frame()
            fig.clear()
