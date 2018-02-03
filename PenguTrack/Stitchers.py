#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Stitchers.py

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

from __future__ import print_function, division
import clickpoints
import numpy as np
from .Detectors import Measurement
from .Filters import Filter
from .Models import Model, VariableSpeed
from .DataFileExtended import DataFileExtended
from scipy.optimize import linear_sum_assignment
import time
import peewee



class Stitcher(object):
    def __init__(self):
        self.Tracks = {}
        self.db = None
        self.track_dict = {}

    def add_PT_Tracks(self, tracks):
        for track in tracks:
            if len(self.Tracks) == 0:
                i = 0
            else:
                i = max(self.Tracks.keys()) + 1
            self.Tracks.update({i: track})

    def add_PT_Tracks_from_Tracker(self, tracks):
        for i in tracks:
            self.Tracks.update({i: tracks[i]})

    def load_tracks_from_clickpoints(self, path, type):
        db = DataFileExtended(path)
        tracks = db.getTracks(type=type)

        for track in tracks:
            n = len(self.track_dict)
            self.track_dict.update({n: track.id})
            self.Tracks.update({track.id: Filter(Model(state_dim=2, meas_dim=2))})
            self.Tracks[track.id].Model.Measured_Variables=["PositionX","PositionY"]
            X = dict([[m.image.sort_index, [[m.x], [m.y]]] for m in track.track_markers])
            # M = dict([Measurement[m.measurement.PositionX,m.measurement.PositionY] for m in track.track_markers])
            for i in sorted(X):
                try:
                    self.Tracks[track.id].predict(i)
                except:
                    pass
                self.Tracks[track.id].update(i=i, z=Measurement(1., X[i]))
        # time1 = time.clock()
        # self.db = DataFileExtended(path)
        # if self.db.getTracks(type=type)[0].markers[0].measurement is not None:
        #     for track in self.db.getTracks(type=type):
        #         self.Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
        #         for m in track.markers:
        #             meas = Measurement(1., [m.measurement[0].x,
        #                                     m.measurement[0].y,
        #                                     m.measurement[0].z])
        #             self.Tracks[track.id].update(z=meas, i=m.image.sort_index)
        # else:
        #     for track in self.db.getTracks(type=type):
        #         self.Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
        #         for m in track.markers:
        #             meas = Measurement(1., [m.x,
        #                                     m.y,
        #                                     0])
        #             self.Tracks[track.id].update(z=meas, i=m.image.sort_index)
        # time2 = time.clock()
        # self.db.db.close()
        # print("Load Tracks Time:",time2 - time1)
    def pos_delta(self):
        first_pos = np.zeros((len(self.Tracks), len(self.Tracks), self.Tracks[self.Tracks.keys()[0]].Model.State_dim,1))
        last_pos = np.zeros((len(self.Tracks), len(self.Tracks), self.Tracks[self.Tracks.keys()[0]].Model.State_dim,1))
        for i in range(len(self.Tracks)):
            first_pos[:,i] = self.Tracks[self.track_dict[i]].X[min([k for k in self.Tracks[self.track_dict[i]].X])]
            last_pos[i,:] = self.Tracks[self.track_dict[i]].X[max([k for k in self.Tracks[self.track_dict[i]].X])]
        return last_pos-first_pos

    def spatial_diff(self):
        first_pos = np.zeros((len(self.Tracks), len(self.Tracks), self.Tracks[self.Tracks.keys()[0]].Model.State_dim,1))
        last_pos = np.zeros((len(self.Tracks), len(self.Tracks), self.Tracks[self.Tracks.keys()[0]].Model.State_dim,1))
        for i in range(len(self.Tracks)):
            first_pos[:,i] = self.Tracks[self.track_dict[i]].X[min([k for k in self.Tracks[self.track_dict[i]].X])]
            last_pos[i,:] = self.Tracks[self.track_dict[i]].X[max([k for k in self.Tracks[self.track_dict[i]].X])]
        return np.linalg.norm(last_pos-first_pos, axis=(2,3))

    def temporal_diff(self):
        first_time = np.zeros((len(self.Tracks), len(self.Tracks)))
        last_time = np.zeros((len(self.Tracks), len(self.Tracks)))
        for i in range(len(self.Tracks)):
            first_time[:,i] = min([k for k in self.Tracks[self.track_dict[i]].X])
            last_time[i,:] = max([k for k in self.Tracks[self.track_dict[i]].X])
        return first_time-last_time

    def _stitch_(self, cost, threshold=np.inf):
        rows, cols = linear_sum_assignment(cost)
        dr = dict(np.array([rows, cols]).T)
        dc = dict(np.array([cols, rows]).T)
        for j, k in enumerate(rows):
            if not cost[rows[j],cols[j]] >= threshold and not rows[j]==cols[j]:
                a = rows[j]
                b = cols[j]
                while self.track_dict[a] not in self.Tracks:
                    a = dc[a]
                while self.track_dict[b] not in self.Tracks:
                    b = dr[b]
                self.__stitch__(self.track_dict[a], self.track_dict[b])


    def __stitch__(self, i, j):
        idx = [i,j][np.argmin([min(self.Tracks[i].X), min(self.Tracks[j].X)])]
        n_idx = i if j==idx else j
        times = set([ii for ii in self.Tracks[i].X])
        times.update([ii for ii in self.Tracks[j].X])
        for t in times:
            if t not in self.Tracks[idx].Predicted_X and t not in self.Tracks[idx].X:
                self.Tracks[idx].predict(i=t)
            if t in self.Tracks[idx].X and t in self.Tracks[n_idx].X:
                pass
            elif t in self.Tracks[n_idx].X:
                self.Tracks[idx].update(i=t, z=self.Tracks[n_idx].Measurements[t])
                print("stitch!")
        self.Tracks.pop(n_idx)

    def stitch(self):
        pass

    def save_tracks_to_db(self, path, type, function=None):
        if function is None:
            function = lambda x : x
        db = DataFileExtended(path)

        track_set = []
        for track in self.Tracks:
            db_track = db.setTrack(type=type.name)
            for i in self.Tracks[track].X:
                meas = self.Tracks[track].Measurements[i]
                pos = self.Tracks[track].Model.vec_from_meas(meas)
                pos = function(pos)
                track_set.append(dict(track=db_track.id, type=type.name, frame=i, x=pos[1], y=pos[0]))
        try:
            db.setMarkers(track=[m["track"] for m in track_set],
                               frame=[m["frame"] for m in track_set],
                               x=[m["x"] for m in track_set],
                               y=[m["y"] for m in track_set])
        except peewee.OperationalError:
            for track in set([m["track"] for m in track_set]):
                small_set = [m for m in track_set if m["track"]==track]
                db.setMarkers(track=[m["track"] for m in small_set],
                              frame=[m["frame"] for m in small_set],
                              x=[m["x"] for m in small_set],
                              y=[m["y"] for m in small_set])


class DistanceStitcher(Stitcher):
    def __init__(self, max_velocity, max_frames=3):
            super(DistanceStitcher, self).__init__()
            self.MaxV = float(max_velocity)
            self.MaxF = int(max_frames)

    def stitch(self):
        s = self.spatial_diff()
        t = self.temporal_diff()
        cost = s/t
        # velocity[np.abs(velocity)>=self.MaxV] = np.nan
        cost[np.diag(np.ones(len(cost), dtype=bool))] = self.MaxV
        cost[cost<0]=self.MaxV
        cost[np.abs(t)>self.MaxF]=self.MaxV
        self._stitch_(cost, threshold=self.MaxV)

class expDistanceStitcher(Stitcher):
    def __init__(self, k, w, max_distance=np.inf, max_delay=np.inf, dim=(1,)):
        super(expDistanceStitcher, self).__init__()
        k=np.array(k, ndmin=2)
        if len(k)>1:
            self.K = np.array(k, ndmin=2)
        else:
            self.K = k * np.ones(dim).T
        self.W = float(w)
        self.MaxDist=float(max_distance)
        self.MaxDelay=float(max_delay)
        self.Dim = dim

    def stitch(self):
        s = self.pos_delta()
        s_abs = np.linalg.norm(s,axis=(2,3))
        t = self.temporal_diff()[:,:,None]
        cost = np.exp((np.dot(self.K, np.abs(s)) + self.W*np.abs(t)))[0].T[0].T
        max_cost = np.amax(cost)*2
        cost[np.diag(np.ones(len(cost), dtype=bool))] = max_cost
        cost[s_abs>self.MaxDist] = max_cost
        cost[t.T[0].T>self.MaxDelay] = max_cost
        cost[t.T[0].T<0] = max_cost
        print(cost.shape)
        self._stitch_(cost, threshold=max_cost)






class Heublein_Stitcher(Stitcher):
    """
    Class for Stitching Tracks
    """

    def __init__(self, a1, a2, a3, a4, a5, max_cost, limiter):
        """
        Stitcher for connecting shorter tracks to longer ones depending on geometric and time-based differences.

        Parameters
        ----------
        tracks: list
            List of tracks containing array-like object with shape (x,y,z,t)
        tracks: list
            List of clickpoints tracks.
        a1: float
            Parameter for cost function xy-distance.
        a2: float
            Parameter for cost function z-distance.
        a3: float
            Parameter for cost function track1 end_frame track2 start_frame distance.
        a4: float
            Parameter for cost function Frame distance for negative distances.
        a5: float
            Parameter for cost function extra cost for xy,z-distance for negative distances.
        max_cost: int
            Parameter for cost function.
        limiter: int
            Limits the number of loops.
        """
        super(Heublein_Stitcher, self).__init__()
        self.limiter = limiter
        # self.path = path
        # self.db = clickpoints.DataFile(path)
        self.max_cost = max_cost
        # self.track_type = str(track_type)
        # self.end_frame = int(end_frame)
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.a3 = float(a3)
        self.a4 = float(a4)
        self.a5 = float(a5)

        self.end_frame = None

    def __ndargmin__(self,array):
        """
        Works like numpy.argmin, but returns array of indices for multi-dimensional input arrays. Ignores NaN entries.
        """
        return np.unravel_index(np.nanargmin(array), array.shape)

    def __points__(self,track1, track2):
        """
        Returns the cost for stitching 2 tracks.

        Parameters
        ----------
        track1: object
            track object in the right format
        track2: object
            track object in the right format
        """
        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        # end_frame = self.end_frame
        # end_frame = max([max(track.Measurements) for track in self.Tracks.values()])
        default = np.nan
        if track1["end"] == self.end_frame:
            x = default
        elif track1["id"] == track2["id"] or track1["end"] == track1["start"] or track2["end"] <= track1["end"]:
            x = default
        else:
            distxy = np.sqrt(np.sum((track2["start_pos"] - track1["end_pos"]) ** 2.))
            distz = np.abs(track2["z_start"] - track1["z_end"])
            dt = track2["start"] - track1["end"]
            # print(distxy, distz, dt)
            if dt < 0:
                x = a5 * (a1 * distxy + a2 * distz) / (-dt + 1) + a4 * (-dt)
            else:
                x = (a1 * distxy + a2 * distz) / (dt + 1) + a3 * dt
        return x

    def getHeubleinTracks(self):
        """
        Returns the tracks in the right format.
        """
        H_Tracks = []
        for track in self.Tracks:
            track_data = self.Tracks[track]
            t_start = min(track_data.X.keys())
            t_end = max(track_data.X.keys())
            meas_start = self.Tracks[track].Measurements[t_start]
            meas_end = self.Tracks[track].Measurements[t_end]
            H_Tracks.append(
                {
                    "id": track,
                    "start": t_start,
                    "end": t_end,
                    "start_pos": np.asarray([meas_start.PositionX, meas_start.PositionY]),
                    "end_pos": np.asarray([meas_end.PositionX, meas_end.PositionY]),
                    "z_start": meas_start.PositionZ,
                    "z_end": meas_end.PositionZ
                })
        return H_Tracks

    def find_matches(self, Tracks, ids):
        """
        Uses the points function to find the best matches.

                Parameters
        ----------
        Tracks: list
            list of tracks in the right format.
        ids: list
            list of track ids.
        """
        # ids = [x['id'] for x in Tracks]
        time1 = time.clock()
        max_cost = self.max_cost
        distance_matrix = np.zeros((len(Tracks), len(Tracks)))
        for i, track1 in enumerate(Tracks):
            for j, track2 in enumerate(Tracks):
                distance = self.__points__(track1, track2)
                distance_matrix[i, j] = distance
        matches = []  # Find the best Matches
        while True:
            if np.all(np.isnan(distance_matrix)):
                break
            [i, j] = self.__ndargmin__(distance_matrix)
            cost = distance_matrix[i, j]

            if cost < max_cost:
                matches.append([ids[i], ids[j]])
                distance_matrix[i, :] = np.nan
                distance_matrix[:, j] = np.nan

                distance_matrix[:, i] = np.nan
                distance_matrix[j, :] = np.nan
            else:
                break
        time2 = time.clock()
        print("find matches time:", time2 - time1)
        return matches

    def Delete_Tracks_with_length_1(self):
        """
        Delets tracks with a length of 1.
        """
        shorts = [track for track in self.Tracks if len(self.Tracks[track].Measurements)<=2]
        for short in shorts:
            self.Tracks.pop(short)
        print(len(self.Tracks))

    def average_Measurements(self, meas1, meas2):
        """
        returns mean of two measurements
        """
        prob = (meas1.Log_Probability + meas2.Log_Probability)/2.
        posX = (meas1.PositionX + meas2.PositionX)/2.
        posY = (meas1.PositionY + meas2.PositionY)/2.
        posZ = (meas1.PositionZ + meas2.PositionZ)/2.
        return Measurement(prob, [posX, posY, posZ])

    def stitch(self):
        """
        Uses all the functions and runs the main process
        """
        self.end_frame = max([max(track.Measurements) for track in self.Tracks.values()])
        print (self.end_frame)
        start_time = time.clock()
        len_test = 0
        len_test2 = 1
        limit = 1
        limiter = self.limiter
        while limit < limiter and (len_test - len_test2) != 0:
            len_test = len(self.Tracks)
            print(len_test)
            H_Tracks = self.getHeubleinTracks()
            ids = [x['id'] for x in H_Tracks]
            matches = self.find_matches(H_Tracks, ids)
            for group in matches:
                start = H_Tracks[ids.index(group[1])]['start']
                end = H_Tracks[ids.index(group[0])]['end']
                track1 = self.Tracks[group[0]]
                track2 = self.Tracks[group[1]]
                if start == end:
                    # Reposition Marker
                    meas1 = track1.Measurements[end]
                    meas2 = track2.Measurements[start]
                    track1.update(z = self.average_Measurements(meas1, meas2), i = start)
                    # track2.Measurements.pop(end)
                    for meas in track2.Measurements:
                        track1.update(z = track2.Measurements[meas], i = meas)
                    # self.Tracks.pop(group[1])
                elif start < end:
                    # Reposition Marker
                    for frame in track2.Measurements:
                        if track1.Measurements.has_key(frame):
                            meas1 = track1.Measurements[frame]
                            meas2 = track2.Measurements[frame]
                            try:
                                track1.update(z = self.average_Measurements(meas1, meas2), i = frame)
                            except KeyError:
                                if not np.any([k<frame for k in track1.X.keys()]):
                                    pass
                                else:
                                    raise
                            # track2.Measurements.pop(frame)
                        else:
                            try:
                                track1.update(z=track2.Measurements[frame], i=frame)
                            except KeyError:
                                if not np.any([k<frame for k in track1.X.keys()]):
                                    pass
                                else:
                                    raise
                    # self.Tracks.pop(group[1])

                else:
                    for frame in track2.Measurements:
                        track1.update(z=track2.Measurements[frame], i=frame)
                self.Tracks.pop(group[1])

            len_test2 = len(self.Tracks)
            limit += 1

        self.Delete_Tracks_with_length_1()
        end_time = time.clock()
        print("stitch time:", end_time - start_time)
        print ("-----------Done with stitching-----------")


