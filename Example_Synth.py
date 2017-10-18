#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example_Synth.py

# Copyright (c) 2016-2017, Alexander Winterl
#
# This file is part of PenguTrack
#
# PenguTrack is free software: you can redistribute and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the Licensem, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PenguTrack. If not, see <http://www.gnu.org/licenses/>.

from PenguTrack.Detectors import Measurement
import numpy as np

class SyntheticDataGenerator(object):
    def __init__(self, object_number, position_variation, speed_variation, model):
        self.N = int(object_number)
        self.R = float(position_variation)
        self.Q = float(speed_variation)
        self.Model = model
        vec = np.random.rand(self.N, self.Model.State_dim, 1)
        vec [:,::2] *= self.R
        vec [1:,::2] *= self.Q
        self.Objects = {0: vec}

        for i in range(100):
            self.step()

    def step(self):
        i = max(self.Objects.keys())
        positions = self.Objects[i]
        self.Objects[i+1] = np.array([self.Model.predict(self.Model.evolute(self.Q*np.random.randn(self.Model.Evolution_dim, 1),
                                                                            pos),
                                                         np.zeros((self.Model.Control_dim, 1))) for pos in positions])
        return [Measurement(1.0, self.Model.measure(pos)) for pos in self.Objects[i+1]]

if __name__ == '__main__':
    object_size = 1  # Object diameter (smallest)
    q = 1.  # Variability of object speed relative to object size
    r = 1.  # Error of object detection relative to object size
    log_prob_threshold = -20.   # Threshold for track stopping

    # Physical Model (used for predictions)
    from PenguTrack.Models import VariableSpeed
    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(dim=2, timeconst=1)

    Generator = SyntheticDataGenerator(10, 10., 1., VariableSpeed(dim=2, timeconst=0.5, damping=1.))

    # Tracker (assignment and data handling)
    from PenguTrack.Filters import MultiFilter
    # Kalman Filter (storage of data, prediction, representation of tracks)
    from PenguTrack.Filters import KalmanFilter
    #Standard Modules for parametrization
    import scipy.stats as ss
    import numpy as np
    # Set up Kalman filter
    X = np.zeros(model.State_dim).T  # Initial Value for Position
    Q = np.diag([q*object_size*np.ones(model.Evolution_dim)])  # Prediction uncertainty
    R = np.diag([r*object_size*np.ones(model.Meas_dim)])  # Measurement uncertainty
    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
    # Initialize Filter/Tracker
    MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                           np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
    MultiKal.LogProbabilityThreshold = log_prob_threshold

    # Extended Clickpoints Database for usage with pengutack
    from PenguTrack.DataFileExtended import DataFileExtended
    # Open ClickPoints Database
    db = DataFileExtended("./synth_data.cdb", "w")
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
    for i in range(30):
        image = db.setImage(filename="%s.png"%i, frame=i)
        # Prediction step, without applied control(vector of zeros)
        MultiKal.predict(i=i)

        # Detection step
        Positions = Generator.step()
        print("Found %s Objects!"%len(Positions))

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
                    pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])[:2]
                    pred_marker = db.setMarker(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                               type=prediction_marker_type)

        print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

    print('done with Tracking')

    state_dict = dict([[k, np.array([MultiKal.Filters[k].X[i] for i in MultiKal.Filters[k].X])] for k in MultiKal.Filters])
    params = ["damping", "timeconst"]
    # params = ["timeconst"]
    # params = ["damping"]

    print("Real Params: ",[Generator.Model.Initial_KWArgs[o] for o in sorted(params)])
    new_mod = Generator.Model#VariableSpeed(dim=2, damping=1., timeconst=2)
    print("Easy Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
    print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))

    new_mod = VariableSpeed(dim=2, damping=1., timeconst=1)
    print("Complex Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
    print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
    p_opt = new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params)
    new_mod = VariableSpeed(dim=2, damping=p_opt[0], timeconst=p_opt[1])
    MultiKal.Model = new_mod
