if __name__ == '__main__':
    import clickpoints
    from PenguTrack.Filters import KalmanFilter
    from PenguTrack.Filters import MultiFilter
    from PenguTrack.Models import VariableSpeed
    from PenguTrack.Detectors import ViBeSegmentation
    from PenguTrack.Detectors import AreaDetector
    from PenguTrack.Detectors import Measurement as Pengu_Measurement

    import scipy.stats as ss
    import numpy as np
    import matplotlib.pyplot as plt

    object_size = 16  # Object diameter (smallest)
    object_area = 200  # Object area in px
    object_number = 20  # Number of Objects in First Track

    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(1, 1, dim=2, timeconst=1)
    q = 2
    r = 2
    X = np.zeros(4).T  # Initial Value for Position
    Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
    R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

    # Initialize Filter
    MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                           np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

    # Open ClickPoints Database
    db = clickpoints.DataFile("./cell_data.cdb")
    # db = clickpoints.DataFile("./penguin_data.cdb")
    # db = clickpoints.DataFile("./pillar_data.cdb")

    # Init_Background from Image_Median
    N = db.getImages().count()
    init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                               for j in np.random.randint(0, N, 20)], axis=0), dtype=np.int)

    # Init Segmentation Module with Init_Image
    VB = ViBeSegmentation(init_image=init, n=20, n_min=18, r=10, phi=1)
    for i in np.random.randint(0, N, 5):
        VB.detect(db.getImage(frame=i).data)

    # Init Detection Module
    AD = AreaDetector(object_area)
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

    # append Database if necessary
    import peewee


    class Measurement(db.base_model):
        # full definition here - no need to use migrate
        marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name="measurement",
                                        on_delete='CASCADE')  # reference to frame and track via marker!
        log = peewee.FloatField(default=0)
        x = peewee.FloatField()
        y = peewee.FloatField()


    if "measurement" not in db.db.get_tables():
        try:
            db.db.connect()
        except peewee.OperationalError:
            pass
        Measurement.create_table()  # important to respect unique constraint

    db.table_measurement = Measurement  # for consistency


    def setMeasurement(marker=None, log=None, x=None, y=None):
        assert not (marker is None), "Measurement must refer to a marker."
        try:
            item = db.table_measurement.get(marker=marker)
        except peewee.DoesNotExist:
            item = db.table_measurement()

        dictionary = dict(marker=marker, x=x, y=y)
        for key in dictionary:
            if dictionary[key] is not None:
                setattr(item, key, dictionary[key])
        item.save()
        return item


    db.setMeasurement = setMeasurement

    # Start Iteration over Images
    print('Starting Iteration')
    images = db.getImageIterator()
    for image in images:

        i = image.get_id()
        # Prediction step
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        # Detection step
        # SegMap = VB.detect(image.data, do_neighbours=False)
        SegMap = VB.segmentate(image.data, do_neighbours=False)
        Detected_Regions = AD.detect(SegMap, return_regions=True)
        Positions = [Pengu_Measurement(1., prop.centroid) for prop in Detected_Regions]
        for prop in Detected_Regions:
            plt.imshow(SegMap[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[1]])
            plt.figure()
        plt.show()
        break
            # SegMap[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = True
        VB.update(SegMap, image.data, do_neighbours=False)
        print(len(Positions))

        # Setting Mask in ClickPoints
        db.setMask(image=image, data=(~SegMap).astype(np.uint8))
        print("Mask save")
        n = 1


        if np.all(Positions != np.array([])):
            # Update Filter with new Detections
            try:
                MultiKal.update(z=Positions, i=i)
            except IndexError:
                continue
            # Get Tracks from Filters
            for k in MultiKal.Filters.keys():
                x = y = np.nan
                # Case 1: we tracked something in this filter
                if i in MultiKal.Filters[k].Measurements.keys():
                    meas = MultiKal.Filters[k].Measurements[i]
                    x = meas.PositionX
                    y = meas.PositionY
                    # rescale to pixel coordinates
                    x_px = x
                    y_px = y
                    prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

                # Case 3: we want to see the prediction markers
                if i in MultiKal.Filters[k].Predicted_X.keys():
                    pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                    # rescale to pixel coordinates
                    pred_x_px = pred_x
                    pred_y_px = pred_y

                # For debugging detection step we set markers at the log-scale detections
                try:
                    yy, xx = [y_px, x_px]
                    db.setMarker(image=image, x=yy, y=xx, text="Detection %s" % k, type=marker_type)
                except:
                    pass

                # Warp back to image coordinates
                x_img, y_img = [y_px, x_px]
                pred_x_img, pred_y_img = [pred_y_px, pred_x_px]

                # Write assigned tracks to ClickPoints DataBase
                if i in MultiKal.Filters[k].Predicted_X.keys():
                    pred_marker = db.setMarker(image=image, x=pred_x_img, y=pred_y_img, text="Track %s" % (100 + k),
                                               type=marker_type3)
                if np.isnan(x) or np.isnan(y):
                    pass
                else:
                    if db.getTrack(k + 100):
                        track_marker = db.setMarker(image=image, type=marker_type2, track=(100 + k), x=x_img, y=y_img,
                                                    text='Track %s, Prob %.2f' % ((100 + k), prob))
                        print('Set Track(%s)-Marker at %s, %s' % ((100 + k), x_img, y_img))
                    else:
                        db.setTrack(marker_type2, id=100 + k)
                        if k == MultiKal.CriticalIndex:
                            db.setMarker(image=image, type=marker_type, x=x_img, y=y_img,
                                         text='Track %s, Prob %.2f, CRITICAL' % ((100 + k), prob))
                        track_marker = db.setMarker(image=image, type=marker_type2, track=100 + k, x=x_img, y=y_img,
                                                    text='Track %s, Prob %.2f' % ((100 + k), prob))
                        print('Set new Track %s and Track-Marker at %s, %s' % ((100 + k), x_img, y_img))

                    # Save measurement in Database
                    db.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

    print('done with Tracking')