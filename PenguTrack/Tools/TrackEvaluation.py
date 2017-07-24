from __future__ import print_function, division
import numpy as np
import clickpoints

from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import VariableSpeed
from PenguTrack.DataFileExtended import DataFileExtended

import matplotlib.pyplot as plt
import seaborn as sn
import my_plot

class Evaluator(object):
    def __init__(self):
        self.System_Tracks = {}
        self.GT_Tracks = {}
        self.gt_db = None
        self.system_db = None

    def add_PT_Tracks_to_GT(self, tracks):
        for track in tracks:
            if len(self.Tracks) == 0:
                i = 0
            else:
                i = max(self.Tracks.keys()) + 1
            self.GT_Tracks.update({i: track})

    def add_PT_Tracks_to_System(self, tracks):
        for track in tracks:
            if len(self.Tracks) == 0:
                i = 0
            else:
                i = max(self.Tracks.keys()) + 1
            self.System_Tracks.update({i: track})

    def add_PT_Tracks_from_Tracker_to_GT(self, tracks):
        for i in tracks:
            self.GT_Tracks.update({i: tracks[i]})

    def add_PT_Tracks_from_Tracker_to_System(self, tracks):
        for i in tracks:
            self.System_Tracks.update({i: tracks[i]})

    def load_GT_tracks_from_clickpoints(self, path, type):
        self.GT_db = DataFileExtended(path)
        if self.GT_db.getTracks(type=type)[0].markers[0].measurement is not None and \
            len(self.GT_db.getTracks(type=type)[0].markers[0].measurement)>0:
            for track in self.GT_db.getTracks(type=type):
                self.GT_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z])
                    self.GT_Tracks[track.id].update(z=meas, i=m.image.sort_index)
        else:
            for track in self.GT_db.getTracks(type=type):
                self.GT_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.x,
                                            m.y,
                                            0])
                    self.GT_Tracks[track.id].update(z=meas, i=m.image.sort_index)


    def load_System_tracks_from_clickpoints(self, path, type):
        self.System_db = DataFileExtended(path)
        if self.System_db.getTracks(type=type)[0].markers[0].measurement is not None and \
            len(self.System_db.getTracks(type=type)[0].markers[0].measurement)>0:
            for track in self.System_db.getTracks(type=type):
                self.System_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z],
                                       frame=m.image.sort_index)
                    self.System_Tracks[track.id].update(z=meas, i=m.image.sort_index)
        else:
            for track in self.System_db.getTracks(type=type):
                self.System_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.x,
                                            m.y,
                                            0], frame=m.image.sort_index)
                    self.System_Tracks[track.id].update(z=meas, i=m.image.sort_index)

    def save_tracks_to_db(self, path, type, function=None):
        if function is None:
            function = lambda x : x
        self.db = DataFileExtended(path)

        with self.db.db.atomic() as transaction:
            self.db.deleteTracks(type=type)
            for track in self.Tracks:
                # self.db.deleteTracks(id=track)
                db_track = self.db.setTrack(type=type, id=track)
                for m in self.Tracks[track].Measurements:
                    meas = self.Tracks[track].Measurements[m]
                    pos = [meas.PositionX, meas.PositionY, meas.PositionZ]
                    pos = function(pos)
                    marker = self.db.setMarker(type=type, frame=m, x=pos[0], y=pos[1], track=track)
                    meas = self.Tracks[track].Measurements[m]
                    self.db.setMeasurement(marker=marker, log=meas.Log_Probability,
                                           x=meas.PositionX, y=meas.PositionY, z=meas.PositionZ)

    def save_GT_tracks_to_db(self, path, type, function=None):
        if function is None:
            function = lambda x : x
        self.GT_db = DataFileExtended(path)

        with self.GT_db.db.atomic() as transaction:
            self.GT_db.deleteTracks(type=type)
            for track in self.Tracks:
                # self.db.deleteTracks(id=track)
                GT_db_track = self.GT_db.setTrack(type=type, id=track)
                for m in self.GT_Tracks[track].Measurements:
                    meas = self.GT_Tracks[track].Measurements[m]
                    pos = [meas.PositionX, meas.PositionY, meas.PositionZ]
                    pos = function(pos)
                    marker = self.GT_db.setMarker(type=type, frame=m, x=pos[0], y=pos[1], track=track)
                    meas = self.GT_Tracks[track].Measurements[m]
                    self.GT_db.setMeasurement(marker=marker, log=meas.Log_Probability,
                                           x=meas.PositionX, y=meas.PositionY, z=meas.PositionZ)

    def save_System_tracks_to_db(self, path, type, function=None):
        if function is None:
            function = lambda x : x
        self.System_db = DataFileExtended(path)

        with self.System_db.db.atomic() as transaction:
            self.System_db.deleteTracks(type=type)
            for track in self.System_Tracks:
                # self.db.deleteTracks(id=track)
                System_db_track = self.System_db.setTrack(type=type, id=track)
                for m in self.System_Tracks[track].Measurements:
                    meas = self.System_Tracks[track].Measurements[m]
                    pos = [meas.PositionX, meas.PositionY, meas.PositionZ]
                    pos = function(pos)
                    marker = self.System_db.setMarker(type=type, frame=m, x=pos[0], y=pos[1], track=track)
                    meas = self.System_Tracks[track].Measurements[m]
                    self.System_db.setMeasurement(marker=marker, log=meas.Log_Probability,
                                           x=meas.PositionX, y=meas.PositionY, z=meas.PositionZ)

class Yin_Evaluator(Evaluator):
    def __init__(self, object_size, *args, **kwargs):
        self.TempThreshold = kwargs.pop("temporal_threshold", 0.1)
        self.SpaceThreshold = kwargs.pop("spacial_threshold", 0.1)
        self.Object_Size = object_size
        self.Matches = {}
        super(Yin_Evaluator, self).__init__(*args, **kwargs)

    def temporal_overlap(self, trackA, trackB):
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        return len(set(trackA.X.keys()).intersection(trackB.X.keys()))/len(set(trackA.X.keys()).union(set(trackB.X.keys())))
        # return min(max(trackA.X.keys()),max(trackB.X.keys()))-max(min(trackA.X.keys()),min(trackB.X.keys()))

    def spatial_overlap(self, trackA, trackB):
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
        pointsA = [trackA.X[f] for f in frames]
        pointsB = [trackB.X[f] for f in frames]
        return dict(zip(frames,[self._intersect_(p1,p2)/self._union_(p1,p2) for p1,p2 in zip(pointsA,pointsB)]))

    def _intersect_(self, p1, p2):
        o = self.Object_Size/2.
        l = max(p1[0]-o, p2[0]-o)
        r = min(p1[0]+o, p2[0]+o)
        t = min(p1[1]+o, p2[1]+o)
        b = max(p1[1]-o, p2[1]-o)
        return (t-b)*(r-l) if ((r-l)>0 and (t-b)>0) else 0

    def _union_(self, p1, p2):
        return 2*self.Object_Size**2-self._intersect_(p1,p2)

    def _handle_Track_(self, track):
        if not isinstance(track, Filter):
            try:
                A = int(track)
            except TypeError:
                raise ValueError("%s is not a track!"%track)
            if self.GT_Tracks.has_key(A):
                 return self.GT_Tracks[A]
            elif self.System_Tracks.has_key(A):
                return  self.System_Tracks[A]
            else:
                raise ValueError("%s is not a track!" % track)




        else:
            return track

    def match(self):
        for gt in self.GT_Tracks:
            self.Matches.update({gt: [st for st in self.System_Tracks if (np.mean(self.spatial_overlap(st,gt).values())>self.SpaceThreshold and
                                                self.temporal_overlap(st,gt)>self.TempThreshold)]})




if __name__ == "__main__":
    evaluation = Yin_Evaluator(10, temporal_threshold=0., spacial_threshold=0.)
    evaluation.load_GT_tracks_from_clickpoints(path="/home/birdflight/Desktop/252Horizon.cdb", type="GT")
    evaluation.load_System_tracks_from_clickpoints(path="/home/birdflight/Desktop/252_Tracked.cdb", type="PT_Track_Marker")
    evaluation.match()
    print(evaluation.Matches)