if __name__ == '__main__':
    import clickpoints
    from PenguTrack.Filters import KalmanFilter
    from PenguTrack.Filters import MultiFilter
    from PenguTrack.Models import VariableSpeed
    from PenguTrack.Detectors import ViBeSegmentation
    from PenguTrack.Detectors import BlobDetector

    import scipy.stats as ss
    import numpy as np

    object_size = 18  # Object diameter (smallest)
    object_number = 15  # Number of Objects in First Track
    object_size = 11  # Object diameter (smallest)
    object_number = 2  # Number of Objects in First Track
    object_size = 200  # Object diameter (smallest)
    object_number = 2  # Number of Objects in First Track

    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(1, 1, dim=2, timeconst=0.5)

    uncertainty = 8*object_size
    X = np.zeros(4).T  # Initial Value for Position
    Q = np.diag([uncertainty, uncertainty])  # Prediction uncertainty
    R = np.diag([uncertainty*2, uncertainty*2])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

    # Initialize Filter
    MultiKal = MultiFilter(KalmanFilter, model, np.array([uncertainty, uncertainty]),
                           np.array([uncertainty, uncertainty]), meas_dist=Meas_Dist, state_dist=State_Dist)

    # Open ClickPoints Database
    db = clickpoints.DataFile("./cell_data.cdb")
    db = clickpoints.DataFile("./penguin_data.cdb")
    db = clickpoints.DataFile("./pillar_data.cdb")

    # Init_Background from Image_Median
    N = db.getImages().count()
    init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                               for j in np.random.randint(0, N, 20)], axis=0), dtype=np.int)

    # Init Segmentation Module with Init_Image
    VB = ViBeSegmentation(init_image=init, n_min=12, r=1800, phi=4)
    # Init Detection Module
    BD = BlobDetector(object_size, object_number)
    print('Initialized')

    # Define ClickPoints Marker

    marker_type = db.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
    db.deleteMarkers(type=marker_type)
    marker_type2 = db.setMarkerType(name="Track_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=marker_type2)
    marker_type3 = db.setMarkerType(name="Prediction_Marker", color="#0000FF")
    db.deleteMarkers(type=marker_type3)

    # Delete Old Tracks
    db.deleteTracks()

    # Start Iteration over Images
    print('Starting Iteration')
    images = db.getImageIterator(start_frame=342, end_frame=380)
    for image in images:

        i = image.get_id()
        # Prediction step
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        # Detection step
        SegMap = VB.detect(image.data, do_neighbours=False)
        Positions = BD.detect(SegMap)

        # Setting Mask in ClickPoints
        db.setMask(image=image, data=(~SegMap).astype(np.uint8))
        print("Mask save")
        n = 1

        if Positions != np.array([]):

            # Update Filter with new Detections
            MultiKal.update(z=Positions, i=i)

            # Get Tracks from Filter (a little dirty)
            for k in MultiKal.Filters.keys():
                x = y = np.nan
                if i in MultiKal.Filters[k].Measurements.keys():
                    x, y = MultiKal.Filters[k].Measurements[i].Position[::-1]
                    prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)
                elif i in MultiKal.Filters[k].X.keys():
                    x, y = MultiKal.Model.measure(MultiKal.Filters[k].X[i])
                    prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

                if i in MultiKal.Filters[k].Measurements.keys():
                    pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                    prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

                # Write assigned tracks to ClickPoints DataBase
                if np.isnan(x) or np.isnan(y):
                    pass
                else:
                    db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s"%k, type=marker_type3)
                    try:
                        # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                        if k == MultiKal.CriticalIndex:
                            db.setMarker(image=image, type=marker_type, x=x, y=y,
                                         text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                        db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                     text='Track %s, Prob %.2f' % (k, prob))
                        print('Set Track(%s)-Marker at %s, %s'%(k,x,y))
                    except:
                        db.setTrack(marker_type2, id=k)
                        # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                        if k == MultiKal.CriticalIndex:
                            db.setMarker(image=image, type=marker_type, x=x, y=y,
                                         text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                        db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                     text='Track %s, Prob %.2f' % (k, prob))
                        print('Set new Track %s and Track-Marker at %s, %s'%(k, x, y))

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

    print('done with Tracking')
