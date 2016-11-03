if __name__ == '__main__':
    import clickpoints
    from Filters import KalmanFilter as KalmanFilter
    from Filters import MultiFilter
    from Models import VariableSpeed
    import scipy.stats as ss
    import numpy as np

    model = VariableSpeed(1, 1, dim=2)
    ucty = 4*30**0.5#10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 30**0.5
    X = np.zeros(4).T
    P = np.diag([ucty, ucty, ucty, ucty])
    # Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)
    MultiKal = MultiFilter(KalmanFilter, model, np.array([vxvy_uncty, vxvy_uncty]),
                           np.array([meas_uncty, meas_uncty]), meas_dist=Meas_Dist, state_dist=State_Dist)
    # MultiKal = MultiFilter(KalmanFilter, model, n=100, meas_dist=Meas_Dist, state_dist=State_Dist)

    db = clickpoints.DataFile('./adelie_data/gt.cdb')

    marker_type = db.setMarkerType(name="ViBe_Marker", color="#FF0000", style='{"scale":1.2}')
    # db.deleteMarkers(type=marker_type)
    marker_type2 = db.setMarkerType(name="ViBe_Kalman_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=marker_type2)
    marker_type3 = db.setMarkerType(name="ViBe_Kalman_Marker_Predictions", color="#0000FF")
    # db.deleteMarkers(type=marker_type3)

    # db.deleteTracks()
    images = db.getImageIterator()
    for image in images:
        i = image.get_id()
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
        blobs = np.asarray([[m.y, m.x] for m in db.getMarkers(image=image)])
        print(blobs)
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
                        db.setMarker(image=image, type=marker_type2, track=k+100, x=x, y=y, text='Track %s, Prob %.2f'%(k+100, prob))
                        print('Set Track(%s)-Marker at %s, %s'%(k,x,y))
                    except:
                        db.setTrack(marker_type2, id=k+100)
                        # db2.setTrack(db2marker_type2, id=k)
                        db.setMarker(image=image, type=marker_type2, track=k+100, x=x, y=y, text='Track %s, Prob %.2f'%(k+100, prob))
                        print('Set new Track %s and Track-Marker at %s, %s'%(k,x,y))

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))