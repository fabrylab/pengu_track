if __name__ == '__main__':
    from PenguTrack.Filters import MultiFilter  # Tracker (assignment and data handling)
    from PenguTrack.Filters import KalmanFilter  # Kalman Filter (storage of data, prediction, representation of tracks)
    from PenguTrack.Models import VariableSpeed  # Physical Model (used for predictions)
    from PenguTrack.Detectors import ThresholdSegmentation  # Segmentation Modul (splits image into fore and background
    from PenguTrack.Detectors import RegionFilter, RegionPropDetector  # Detector (finds and filters objects in segmented image)
    # from PenguTrack.Detectors import Measurement as Pengu_Measurement  #
    from PenguTrack.DataFileExtended import DataFileExtended

    import scipy.stats as ss
    import numpy as np
    # import matplotlib.pyplot as plt

    object_size = 8  # Object diameter (smallest)
    object_area = 40  # Object area in px
    intensity_threshold = 150

    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(1, 1, dim=2, timeconst=2)

    # Set up Kalman filter
    q = 1
    r = 1
    X = np.zeros(4).T  # Initial Value for Position
    Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
    R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

    # Initialize Filter/Tracker
    MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                           np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
    MultiKal.LogProbabilityThreshold = -20.

    # Open ClickPoints Database
    db = DataFileExtended(".\ExampleData\ExampleData\cell_data.cdb")

    # Init Segmentation Module
    TS = ThresholdSegmentation(intensity_threshold)

    # Init Detection Module
    rf = RegionFilter("area", object_area, var=0.8*object_area, lower_limit=0.5*object_area, upper_limit=1.5*object_area)
    AD = RegionPropDetector([rf])
    print('Initialized')

    # Define ClickPoints Marker
    detection_marker_type = db.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
    db.deleteMarkers(type=detection_marker_type)
    track_marker_type = db.setMarkerType(name="Track_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=track_marker_type)
    prediction_marker_type = db.setMarkerType(name="Prediction_Marker", color="#0000FF")
    db.deleteMarkers(type=prediction_marker_type)

    # Delete Old Tracks
    db.deleteTracks(type=track_marker_type)

    # Start Iteration over Images
    print('Starting Iteration')
    images = db.getImageIterator()
    # areas = []
    for image in images:

        i = image.get_id()
        # Prediction step, without applied control(vector of zeros)
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        # Detection step
        SegMap = TS.detect(image.data)
        Positions = AD.detect(SegMap)
        # import matplotlib.pyplot as plt
        # areas.extend([pos.Data['area'][0] for pos in Positions])
        # if len(areas)>1000:
        #     plt.hist(areas, bins=100)
        #     plt.show()
        print("Found %s Objects!"%len(Positions))

        # Write Segmentation Mask to Database
        db.setMask(image=image, data=(~SegMap).astype(np.uint8))
        print("Mask save")

        if len(Positions)>0:
            # Update Filter with new Detections
            MultiKal.update(z=Positions, i=i)

            # Get Tracks from Filters
            for k in MultiKal.Filters.keys():
                x = y = np.nan
                # Case 1: we tracked something in this filter
                if i in MultiKal.Filters[k].Measurements.keys():
                    meas = MultiKal.Filters[k].Measurements[i]
                    x = meas.PositionX
                    y = meas.PositionY
                    prob = MultiKal.Filters[k].log_prob(keys=[i])

                    if db.getTrack(k + 100):
                        print('Setting Track(%s)-Marker at %s, %s' % ((100 + k), x, y))
                    else:
                        db.setTrack(track_marker_type, id=100 + k)
                        print('Setting new Track %s and Track-Marker at %s, %s' % ((100 + k), x, y))
                    track_marker = db.setMarker(image=image, type=track_marker_type, track=100 + k, x=y, y=x,
                                                text='Track %s, Prob %.2f' % ((100 + k), prob))

                    # Save measurement in Database
                    db.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

                # Case 2: we want to see the prediction markers
                if i in MultiKal.Filters[k].Predicted_X.keys():
                    pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                    pred_marker = db.setMarker(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                               type=prediction_marker_type)

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

    print('done with Tracking')