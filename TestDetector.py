if __name__ == '__main__':
    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()

    import clickpoints
    from PenguTrack.Filters import KalmanFilter
    from PenguTrack.Filters import MultiFilter
    from PenguTrack.Models import VariableSpeed
    from PenguTrack.Detectors import ViBeSegmentation
    from PenguTrack.Detectors import MoGSegmentation
    from PenguTrack.Detectors import BlobDetector
    from PenguTrack.Detectors import AreaDetector
    from PenguTrack.Detectors import WatershedDetector
    import scipy.stats as ss
    import numpy as np
    import matplotlib as mpl
    mpl.use('AGG')
    import matplotlib.pyplot as plt
    import skimage.filters as filters
    # import matplotlib.pyplot as plt

    penguin_size = 8
    n_penguins = 2

    model = VariableSpeed(1, 1, dim=2, timeconst=0.5)
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


    db = clickpoints.DataFile("./Databases/click2.cdb")

    # db2 = clickpoints.DataFile('./adelie_data/gt.cdb')

    images = db.getImageIterator()#start_frame=30, end_frame=40)

    N = db.getImages().count()
    J = np.random.randint(0, N, 20)
    # J = np.random.randint(3, 20, 20)

    init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int) for j in J], axis=0), dtype=np.int)
    plt.imshow(init)
    plt.savefig("./init.png")
    VB = ViBeSegmentation(init_image=init, n_min=15, r=15, phi=1)
    # images = db.getImageIterator(start_frame=0, end_frame=20)
    for j in np.arange(20)[::-1]:
        SegMap = VB.detect(db.getImage(frame=j).data)
        print("Done")
    # MG = MoGSegmentation(init_image=init, k=3)
    BD = BlobDetector(penguin_size, n_penguins)
    AD = AreaDetector(penguin_size, n_penguins)
    WD = WatershedDetector(penguin_size, n_penguins)
    print('Initialized')

    from PenguTrack.Detectors import AreaBlobDetector
    ABD = AreaBlobDetector()
    ABD.detect()

    marker_type = db.setMarkerType(name="ViBe_Marker", color="#FF0000", style='{"scale":1.2}')
    db.deleteMarkers(type=marker_type)
    marker_type2 = db.setMarkerType(name="ViBe_Kalman_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=marker_type2)
    marker_type3 = db.setMarkerType(name="ViBe_Kalman_Marker_Predictions", color="#0000FF")
    db.deleteMarkers(type=marker_type3)

    # db.deleteTracks()
    images = db.getImageIterator()

    import skimage.morphology as morph

    print('Starting Iteration')
    for image in images:
        i = image.get_id()
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        # blobs = VB.detect(filters.gaussian(image.data, 5, multichannel=True))
        SegMap = VB.detect(image.data)
        # SegMap = MG.detect(image.data)

        # selem = np.ones((2, 2))
        # SegMap = morph.binary_opening(SegMap, selem=selem)
        # SegMap = morph.binary_opening(SegMap, selem=morph.disk(int(penguin_size*0.9)))
        # SegMap = morph.binary_closing(SegMap, selem=morph.disk(penguin_size))
        blobs = BD.detect(SegMap)
        blobs = np.array(blobs, ndmin=2)
        db.setMask(image=image, data=(SegMap ^ True).astype(np.uint8))
        print("Mask save")
        n = 1
        if blobs != np.array([]):
            for l in range(blobs.shape[0]):
                db.setMarker(image=image, x=blobs[l][1]*n, y=blobs[l][0]*n, type=marker_type)#, text=str(180/np.pi*np.arctan2(axes[l][0], axes[l][1])))
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

    print('done with Tracking')

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