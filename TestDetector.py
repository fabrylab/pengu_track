if __name__ == '__main__':
    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()

    import clickpoints
    from Filters import KalmanFilter
    from Filters import MultiFilter
    from Models import VariableSpeed
    from Detectors import ViBeDetector
    import scipy.stats as ss
    import numpy as np
    import skimage.filters as filters
    # import matplotlib.pyplot as plt

    penguin_size = 10

    model = VariableSpeed(1, 1, dim=2)
    ucty = 4*penguin_size/0.5#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = penguin_size/0.5
    X = np.zeros(4).T
    P = np.diag([ucty, ucty, ucty, ucty])
    # Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)
    MultiKal = MultiFilter(KalmanFilter, model, np.array([vxvy_uncty, vxvy_uncty]),
                           np.array([meas_uncty, meas_uncty]), meas_dist=Meas_Dist, state_dist=State_Dist)


    db = clickpoints.DataFile("click2.cdb")

    # db2 = clickpoints.DataFile('./adelie_data/gt.cdb')

    images = db.getImageIterator()#start_frame=30, end_frame=40)

    N = db.getImages().count()
    J = np.random.randint(2, N, 20)
    # init = []
    # for j in J:
    #     img = db.getImages(frame=j)
    #     init.append(img.data)
    init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.uint8) for j in J], axis=0), dtype=np.uint8)

    VB = ViBeDetector(init_image=init, n_min=15, r=15, phi=1, object_size=penguin_size, object_number=2)
    print('Initialized')

    marker_type = db.setMarkerType(name="ViBe_Marker", color="#FF0000", style='{"scale":1.2}')
    db.deleteMarkers(type=marker_type)
    marker_type2 = db.setMarkerType(name="ViBe_Kalman_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=marker_type2)
    marker_type3 = db.setMarkerType(name="ViBe_Kalman_Marker_Predictions", color="#0000FF")
    db.deleteMarkers(type=marker_type3)

    db.deleteTracks()
    images = db.getImageIterator()
    for image in images:
        i = image.get_id()
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
        # blobs = VB.detect(filters.gaussian(image.data, 5, multichannel=True))
        blobs = VB.detect(image.data)

        db.setMask(image=image, data=(VB.SegMap ^ True).astype(np.uint8))
        print("Mask save")
        n = 1
        if blobs.shape[0] > 0:
            db.setMarkers(image=image, x=blobs.T[1]*n, y=blobs.T[0]*n, type=marker_type)
            print("Markers Saved (%s)" % blobs.shape[0])
            MultiKal.update(z=np.array([blobs.T[1]*n, blobs.T[0]*n]).T, i=i)

            for k in MultiKal.Filters.keys():
                x = y = np.nan
                if i in MultiKal.Filters[k].Measurements.keys():
                    x, y = MultiKal.Filters[k].Measurements[i]
                    prob = MultiKal.Filters[k].log_prob(keys=[i])
                elif i in MultiKal.Filters[k].X.keys():
                    x, y = MultiKal.Model.measure(MultiKal.Filters[k].X[i])
                    prob = MultiKal.Filters[k].log_prob(keys=[i])

                if i in MultiKal.Filters[k].Measurements.keys():
                    pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                    prob = MultiKal.Filters[k].log_prob(keys=[i])

                if np.isnan(x) or np.isnan(y):
                    pass
                else:
                    db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s"%k, type=marker_type3)
                    try:
                        db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                        print('Set Track(%s)-Marker at %s, %s'%(k,x,y))
                    except:
                        db.setTrack(marker_type2, id=k)
                        db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                        print('Set new Track %s and Track-Marker at %s, %s'%(k, x, y))

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

    print('done with ViBe')

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()

    # ground_truth = []
    # for t in db2.getTracks():
    #     ground_truth.append(np.array(t.points))
    #
    # tracked = []
    # for t in db.getTracks():
    #     tracked.append(np.array(t.points))
    #
    # corrs = {}
    # corrs2 = np.zeros((len(ground_truth),len(tracked)))
    # for i, gt in enumerate(ground_truth):
    #     for j, t in enumerate(tracked):
    #         corr = np.zeros((2, 2))
    #         corr[0, 0] = np.amax(np.correlate(gt.T[0], t.T[0])/(np.correlate(t.T[0], t.T[0])*np.correlate(gt.T[0], gt.T[0]))**0.5)
    #         corr[0, 1] = np.amax(np.correlate(gt.T[0], t.T[1])/(np.correlate(t.T[1], t.T[1])*np.correlate(gt.T[0], gt.T[0]))**0.5)
    #         corr[1, 0] = np.amax(np.correlate(gt.T[1], t.T[0])/(np.correlate(t.T[0], t.T[0])*np.correlate(gt.T[1], gt.T[1]))**0.5)
    #         corr[1, 1] = np.amax(np.correlate(gt.T[1], t.T[1])/(np.correlate(t.T[1], t.T[1])*np.correlate(gt.T[1], gt.T[1]))**0.5)
    #         corrs.update({(i, j): corr})
    #         corrs2[i,j] = 0.5**0.5*(corr[0,0]**2+corr[1,1]**2)**0.5
    # print('Correlation over all %s , %s'%(np.prod(np.amax(corrs2, axis=1)), np.prod(np.amax(corrs2, axis=0))))