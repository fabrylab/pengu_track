#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example.py

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

if __name__ == '__main__':
    object_size = 8  # Object diameter (smallest)
    object_area = 40  # Object area in px
    intensity_threshold = 150  # Segmentation Threshold
    q = 1.  # Variability of object speed relative to object size
    r = 1.  # Error of object detection relative to object size
    log_prob_threshold = -20.  # Threshold for track stopping

    # Physical Model (used for predictions)
    from PenguTrack.Models import VariableSpeed

    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(dim=2, timeconst=2)
    model.add_variable("area")

    # Segmentation Modul (splits image into fore and background
    from PenguTrack.Detectors import TresholdSegmentation

    # Init Segmentation Module
    TS = TresholdSegmentation(intensity_threshold)

    # Detector (finds and filters objects in segmented image)
    from PenguTrack.Detectors import RegionFilter, RegionPropDetector

    # Init Detection Module
    rf = RegionFilter("area", object_area, var=0.8 * object_area, lower_limit=0.5 * object_area,
                      upper_limit=1.5 * object_area)
    AD = RegionPropDetector([rf])
    print('Initialized')

    # Tracker (assignment and data handling)
    from PenguTrack.Filters import MultiFilter
    # Kalman Filter (storage of data, prediction, representation of tracks)
    from PenguTrack.Filters import KalmanFilter
    # Standard Modules for parametrization
    import scipy.stats as ss
    import numpy as np

    # Set up Kalman filter
    Q = np.diag([q * object_size * np.ones(model.Evolution_dim)])  # Prediction uncertainty
    R = np.diag([r * object_size * np.ones(model.Meas_dim)])  # Measurement uncertainty
    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
    # Initialize Filter/Tracker
    from PenguTrack.Filters import HungarianTracker

    Tracker = HungarianTracker(KalmanFilter, model, np.diag(Q),
                                np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
    Tracker.LogProbabilityThreshold = log_prob_threshold

    i_x = Tracker.Model.Measured_Variables.index("PositionX")
    i_y = Tracker.Model.Measured_Variables.index("PositionY")

    # Clickpoints DataBase
    import clickpoints

    # Open ClickPoints Database
    db = clickpoints.DataFile("./ExampleData/cell_data.cdb")
    # Define ClickPoints Marker
    track_marker_type = db.setMarkerType(name="Track_Marker", color="#00FF00", mode=db.TYPE_Track)
    # Delete Old Tracks
    db.deleteMarkers(type=track_marker_type)
    db.deleteTracks(type=track_marker_type)

    # Start Iteration over Images
    print('Starting Iteration')
    images = db.getImageIterator()
    for image in images:

        i = image.sort_index
        # Prediction step, without applied control(vector of zeros)
        Tracker.predict(i=i)

        # Detection step
        SegMap = TS.detect(image.data)
        Positions, Mask = AD.detect(SegMap)
        print("Found %s Objects!" % len(Positions))

        # Write Segmentation Mask to Database
        db.setMask(image=image, data=(~SegMap).astype(np.uint8))
        print("Mask save")

        if len(Positions) > 0:
            # Update Filter with new Detections
            Tracker.update(z=Positions, i=i)
            db_tracks = [t.id for t in db.getTracks(type=track_marker_type)]

            for k in Tracker.ActiveFilters.keys():
                state = Tracker.ActiveFilters[k].X[i]
                prob = Tracker.Filters[k].log_prob(keys=[i], update=Tracker.Filters[k].ProbUpdate)

                if k not in db_tracks:
                    db.setTrack(type=track_marker_type, id=k)

                x = Tracker.Model.measure(state)[i_x]
                y = Tracker.Model.measure(state)[i_y]

                db.setMarker(image=image, x=x, y=y, type=track_marker_type, track=k)


        print("Got %s Filters" % len(Tracker.ActiveFilters.keys()))

    print('done with Tracking')