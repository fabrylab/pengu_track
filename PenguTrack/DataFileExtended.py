#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DataFileExtended.py

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


import peewee
import clickpoints
from clickpoints.DataFile import packToDictList
import numpy as np
from PenguTrack.Filters import Filter
from PenguTrack.Detectors import Measurement
from PenguTrack.Models import VariableSpeed

class DataFileExtended(clickpoints.DataFile):

    def __init__(self, *args, **kwargs):
        clickpoints.DataFile.__init__(self, *args, **kwargs)
        # Define ClickPoints Marker
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track)
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF")

        class Measurement(self.base_model):
            # full definition here - no need to use migrate
            marker = peewee.ForeignKeyField(self.table_marker, unique=True, related_name="measurement",
                                            on_delete='CASCADE')  # reference to frame and track via marker!
            log = peewee.FloatField(default=0)
            x = peewee.FloatField()
            y = peewee.FloatField(default=0)
            z = peewee.FloatField(default=0)

        if "measurement" not in self.db.get_tables():
            Measurement.create_table()  # important to respect unique constraint

        self.table_measurement = Measurement  # for consistency

    def setMeasurement(self, marker=None, log=None, x=None, y=None, z=None):
        assert not (marker is None), "Measurement must refer to a marker."
        try:
            item = self.table_measurement.get(marker=marker)
        except peewee.DoesNotExist:
            item = self.table_measurement()

        dictionary = dict(marker=marker, log=log, x=x, y=y, z=z)
        for key in dictionary:
            if dictionary[key] is not None:
                setattr(item, key, dictionary[key])
        item.save()
        return item

    def setMeasurements(self, marker=None, log=None, x=None, y=None, z=None):
        data = packToDictList(self.table_measurement, marker=marker, log=log,x=x, y=y, z=z)
        return self.saveUpsertMany(self.table_measurement, data)

    def deletetOld(self):
        # Define ClickPoints Marker
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.deleteMarkers(type=self.detection_marker_type)
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track)
        self.deleteMarkers(type=self.track_marker_type)
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF")
        self.deleteMarkers(type=self.prediction_marker_type)
        # Delete Old Tracks
        self.deleteTracks(type=self.track_marker_type)

    def write_to_DB(self, Tracker, image, i=None, text=None, verbose=True):
        if text is None:
            set_text = True
        else:
            set_text = False
        if i is None:
            i = image.sort_index
        # Get Tracks from Filters
        markerset = []
        measurement_set = []
        pred_markerset = []
        for k in Tracker.Filters.keys():
            x = y = np.nan
            # Case 1: we tracked something in this filter
            if i in Tracker.Filters[k].Measurements.keys():
                meas = Tracker.Filters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                prob = Tracker.Filters[k].log_prob(keys=[i], update=False)

                if self.getTrack(k + 100):
                    pass
                    if verbose:
                        print('Setting Track(%s)-Marker at %s, %s' % ((100 + k), x, y))
                else:
                    self.setTrack(self.track_marker_type, id=100 + k)
                    if verbose:
                        print('Setting new Track %s and Track-Marker at %s, %s' % ((100 + k), x, y))
                if set_text:
                    text = 'Track %s, Prob %.2f' % ((100 + k), prob)

                markerset.append(dict(image=image, type=self.track_marker_type, track=100 + k, x=y, y=x,
                                              text=text))

                # Save measurement in Database
                measurement_set.append(dict(log=prob, x=x, y=y))
                # self.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

            # Case 2: we want to see the prediction markers
            if i in Tracker.Filters[k].Predicted_X.keys():
                prediction = Tracker.Model.measure(Tracker.Filters[k].Predicted_X[i])
                pred_x = prediction[Tracker.Model.Measured_Variables.index("PositionX")]
                pred_y = prediction[Tracker.Model.Measured_Variables.index("PositionY")]
                pred_markerset.append(dict(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                           type=self.prediction_marker_type))
        markers = self.setMarkers(image=[m["image"] for m in markerset],
                                  type=[m["type"] for m in markerset],
                                  track=[m["track"] for m in markerset],
                                  x=[m["x"] for m in markerset],
                                  y=[m["y"] for m in markerset],
                                  text=[m["text"] for m in markerset])
        pred_markers = self.setMarkers(image=[m["image"] for m in pred_markerset],
                                  type=[m["type"] for m in pred_markerset],
                                  x=[m["x"] for m in pred_markerset],
                                  y=[m["y"] for m in pred_markerset],
                                  text=[m["text"] for m in pred_markerset])
        markers = self.getMarkers(image=image, track=[m["track"] for m in markerset])
        measurements = self.setMeasurements(marker=[m.id for m in markers],
                                            x=[m["x"] for m in measurement_set],
                                            y=[m["y"] for m in measurement_set],
                                            log=[m["log"] for m in measurement_set])
        if verbose:
            print("Got %s Filters" % len(Tracker.ActiveFilters.keys()))


    def write_to_DB_cam(self, Tracker, image, i=None, text=None):
        if text is None:
            set_text = True
        else:
            set_text = False
        if i is None:
            i = image.sort_index
        # Get Tracks from Filters
        for k in Tracker.Filters.keys():
            x = y = np.nan
            # Case 1: we tracked something in this filter
            if i in Tracker.Filters[k].Measurements.keys():
                meas = Tracker.Filters[k].Measurements[i]
                x = meas.PositionX_Cam
                y = meas.PositionY_Cam
                prob = Tracker.Filters[k].log_prob(keys=[i], update=False)

                if self.getTrack(k + 100):
                    print('Setting Track(%s)-Marker at %s, %s' % ((100 + k), x, y))
                else:
                    self.setTrack(self.track_marker_type, id=100 + k)
                    print('Setting new Track %s and Track-Marker at %s, %s' % ((100 + k), x, y))
                if set_text:
                    text = 'Track %s, Prob %.2f' % ((100 + k), prob)
                track_marker = self.setMarker(image=image, type=self.track_marker_type, track=100 + k, x=y, y=x,
                                              text=text)

                # Save measurement in Database
                self.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

            # Case 2: we want to see the prediction markers
            if i in Tracker.Filters[k].Predicted_X.keys():
                prediction = Tracker.Model.measure(Tracker.Filters[k].Predicted_X[i])
                pred_x = prediction[Tracker.Model.Measured_Variables.index("PositionX")]
                pred_y = prediction[Tracker.Model.Measured_Variables.index("PositionY")]
                pred_marker = self.setMarker(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                           type=self.prediction_marker_type)
        print("Got %s Filters" % len(Tracker.ActiveFilters.keys()))

    def PT_tracks_from_db(self, type, get_measurements=True):
        Tracks = {}
        if self.getTracks(type=type)[0].markers[0].measurement is not None and get_measurements:
            for track in self.getTracks(type=type):
                Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z])
                    Tracks[track.id].update(z=meas, i=m.image.sort_index)
        else:
            for track in self.getTracks(type=type):
                Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    x, y = m.correctedXY()
                    meas = Measurement(1., [x,
                                            y,
                                            0])
                    Tracks[track.id].update(z=meas, i=m.image.sort_index)
        return Tracks