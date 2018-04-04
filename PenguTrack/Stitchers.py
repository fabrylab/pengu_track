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
from PenguTrack import Filters
from .Models import Model, VariableSpeed
from .DataFileExtended import DataFileExtended,\
    add_PT_Tracks, add_PT_Tracks_from_Tracker, load_tracks_from_clickpoints, load_measurements_from_clickpoints, \
    reverse_dict
# from PenguTrack.Filters import greedy_assignment
# from scipy.optimize import linear_sum_assignment
from .Assignment import *
import time
import peewee
from copy import copy
import inspect


all_filters = tuple([v for v in dict(inspect.getmembers(Filters)).values() if type(v)==type])

class new_Stitcher(object):
    def __init__(self, greedy=False, method="hungarian"):
        self.Track_Dict = {}
        self.Track_array = None
        self.db = None
        self.greedy = bool(greedy)
        self.method = str(method)
        self.name = "Stitcher"

    def load_tracks_from_clickpoints(self, path, type):
        db = clickpoints.DataFile(path)
        tracks = db.getTracks(type=type)
        if len(tracks) < 1:
            raise ValueError("The given track-type is empty!")
        track_dict = dict(enumerate([t.id for t in tracks]))
        array = np.array([db.db.execute_sql("select (SELECT x from marker as m WHERE m.image_id = i.id AND m.track_id=?) as x, (SELECT y from marker as m WHERE m.image_id = i.id AND m.track_id=?) as y from image as i order by sort_index",[track_dict[k],track_dict[k]]).fetchall() for k in sorted(track_dict.keys())], dtype=float)
        print("Tracks loaded!")
        self.db, self.Track_array, self.Track_Dict = db, array, track_dict

        self.stiched_type = self.db.setMarkerType(name=self.name, color="F0F0FF")


    def add_PT_Tracks_from_Tracker(self, tracks):
        frames = set()
        for track in tracks:
            frames.update(track.X.keys())
        array = np.ones((len(tracks), len(frames), 2))
        track_dict = enumerate(tracks.keys())
        for f in sorted(frames):
            for t in track_dict:
                if f in tracks[track_dict[t]].X:
                    array[t, f, :] = tracks[track_dict[t]].X[f][:2,0]
        self.Track_Dict, self.Track_array =  track_dict, array

    def pos_delta(self, kernel=None, ignore_nans=False, array=None):
        if array is None:
            array = self.Track_array
        else:
            if len(array.shape)<2:
                raise ValueError("Wrong shape for array, allowed only %sx%sxN"%(self.Track_array.shape[0],
                                                                              self.Track_array.shape[1]))
            elif len(array.shape)==2:
                array = array[:, :, None]
        if kernel is None:
            # look where tracks are defined
            not_none = np.all(~np.isnan(self.Track_array), axis=-1)
            # get positions where cumsum of not_none is 1 and track is defined (first occurence of track)
            first_vals = array[(np.cumsum(not_none, axis=-1) == 1) & (not_none == True)]
            # same as above but with flipped time axis to get last occurence. Reflip mask array in the end
            last_vals = array[((np.cumsum(not_none[:, ::-1], axis=-1) == 1) & (not_none == True)[:,::-1])[:,::-1]]
        else:
            # look where tracks are defined
            not_none = np.all(~np.isnan(self.Track_array), axis=-1)
            if ignore_nans:
                # savety margin for kernel computation. worst case-> always take last frames
                not_none_f = np.array(not_none)
                not_none_f[:, -len(kernel):] = True
                # get cumsum of not_none (value is len of track at the specific timestep)
                f_sum = np.cumsum(not_none_f, axis=-1)
                # iterate over kernel. when index i of kernel value equals cumsum AND value is not none (i-th occurence)
                first_vals = np.nansum([k * array[(f_sum == (i+1)) & (not_none == True)] for i, k in enumerate(kernel)],
                          axis=0)
                # same but with flipped axis
                not_none_l = np.array(not_none)
                not_none_l[:, :len(kernel)-1] = True
                l_sum = np.cumsum(not_none_l[:, ::-1], axis=-1)[:, ::-1]
                last_vals = np.nansum([k * array[(l_sum == (i+1)) & (not_none == True)] for i, k in enumerate(kernel)],
                          axis=0)
            else:
                # get cumsum of not_none.
                not_none_sum = np.cumsum(not_none, axis=-1).astype(bool)
                # savety margin for kernel computation. worst case-> always take last frames
                not_none_sum[:,-len(kernel):] = True
                #then cumsum of cumsum>0 -> value is index from first track occurence
                f_sum = np.cumsum(not_none_sum, axis=-1)
                # iterate over kernel. when index i of kernel value equals cumsum (i-th occurence)
                # don't take care of nan-values. These will be ommitted in nansum
                first_vals = np.nansum([k*array[(f_sum == (i+1))] for i, k in enumerate(kernel)], axis=0)
                # same with flipped axis
                # get cumsum of not_none.
                not_none_sum = np.cumsum(not_none[:, ::-1], axis=-1).astype(bool)
                # savety margin for kernel computation. worst case-> always take first frames
                not_none_sum[:,-len(kernel):] = True
                l_sum = np.cumsum(not_none_sum, axis=-1)[:, ::-1]
                last_vals = np.nansum([k * array[(l_sum == (i+1))] for i, k in enumerate(kernel)], axis=0)
        return first_vals[None, :, :] - last_vals[:, None, :]

    def vel_delta(self, kernel=None, ignore_nans=False):
        Frames = np.cumsum(np.ones_like(self.Track_array[:,:,0]), axis=-1)
        if kernel is None:
            kernel = [1,-1]
        else:
            kernel = np.convolve(kernel, [1,-1])
        return self.pos_delta(kernel=kernel, ignore_nans=ignore_nans)/self.pos_delta(kernel=kernel, ignore_nans=ignore_nans, array=Frames)

    def spatial_diff(self, kernel=None, ignore_nans=False):
        return np.linalg.norm(self.pos_delta(kernel=kernel, ignore_nans=ignore_nans), axis=-1)

    def temporal_diff(self, kernel=None, ignore_nans=False):
        Frames = np.cumsum(np.ones_like(self.Track_array[:,:,0]), axis=-1)
        return self.pos_delta(kernel=kernel, ignore_nans=ignore_nans, array=Frames)

    def _stitch_(self, cost, threshold=np.inf, dummy=False):
        print(np.mean(cost>=threshold))
        cost [cost==np.amax(cost)] == np.nextafter(np.inf, 0)/(max(cost.shape)**2)

        if self.greedy or self.method == "greedy":
            rows, cols = greedy_assignment(cost)
        elif self.method == "network":
            if threshold == np.inf:
                Warning("Threshold is too high, network assignment will cost more computational time than hungarian!")
            rows, cols = network_assignment(cost, threshold=threshold)
        else:
            rows, cols = hungarian_assignment(cost)

        print("Assigned Tracks!")

        Frames = np.cumsum(np.ones_like(self.Track_array[:,:,0]), axis=-1)
        # look where tracks are defined
        not_none = np.all(~np.isnan(self.Track_array), axis=-1)
        # get positions where cumsum of not_none is 1 and track is defined (first occurence of track)
        first_frame = Frames[(np.cumsum(not_none, axis=-1) == 1) & (not_none == True)]

        # dr = dict(np.array([rows, cols]).T)
        # dc = dict(np.array([cols, rows]).T)
        Tracks = reverse_dict(self.Track_Dict)
        stitched = {}
        for j, k in enumerate(rows):
            if not cost[rows[j],cols[j]] >= threshold and not rows[j]==cols[j]:
                print("preparing to stitch...")
                a = rows[j]
                b = cols[j]
                if self.Track_Dict[a] not in Tracks:
                    # try forward search:
                    try:
                        while self.Track_Dict[a] not in Tracks:
                            a = stitched[a]
                    # elso do backward search
                    except KeyError:
                        while self.Track_Dict[a] not in Tracks:
                            a = reverse_dict(stitched)[a]
                if self.Track_Dict[b] not in Tracks:
                    # try backward search:
                    try:
                        while self.Track_Dict[b] not in Tracks:
                            b = reverse_dict(stitched)[b]
                    # else do forward search:
                    except KeyError:
                        while self.Track_Dict[b] not in Tracks:
                            b = stitched[b]
                appended_id, erased_id = np.array([a,b])[np.argsort([first_frame[a], first_frame[b]])]
                stitched.update({a:b})
                print("stitching %s and %s"%(appended_id, erased_id))
                self.__stitch__(self.Track_Dict[appended_id], self.Track_Dict[erased_id], dummy=dummy)
                Tracks.pop(self.Track_Dict[erased_id])

    def _stitch_from_rc(self, rows, cols, dummy=False):
        Frames = np.cumsum(np.ones_like(self.Track_array[:,:,0]), axis=-1)
        # look where tracks are defined
        not_none = np.all(~np.isnan(self.Track_array), axis=-1)
        # get positions where cumsum of not_none is 1 and track is defined (first occurence of track)
        first_frame = Frames[(np.cumsum(not_none, axis=-1) == 1) & (not_none == True)]

        dr = dict(np.array([rows, cols]).T)
        dc = dict(np.array([cols, rows]).T)
        Tracks = reverse_dict(self.Track_Dict)
        for j, k in enumerate(rows):
            if not rows[j]==cols[j]:
                print("preparing to stitch...")
                a = rows[j]
                b = cols[j]
                while self.Track_Dict[a] not in Tracks:
                    try:
                        a = dc[a]
                    except KeyError:
                        a = dr[a]
                while self.Track_Dict[b] not in Tracks:
                    try:
                        b = dr[b]
                    except KeyError:
                        b = dc[b]
                appended_id, erased_id = np.array([a,b])[np.argsort([first_frame[a], first_frame[b]])]
                print("stitching %s and %s"%(appended_id, erased_id))
                self.__stitch__(self.Track_Dict[appended_id], self.Track_Dict[erased_id], dummy=dummy)
                Tracks.pop(self.Track_Dict[erased_id])

    def __stitch__(self, appended_id, erased_id, dummy=False):
        i = reverse_dict(self.Track_Dict)[appended_id]
        j = reverse_dict(self.Track_Dict)[erased_id]
        if self.db:
            if not dummy:
                self.db.getTrack(appended_id).merge(erased_id, mode="average")
            # look where tracks are defined
            not_none = np.all(~np.isnan(self.Track_array[j]), axis=-1)
            # get positions where cumsum of not_none is 1 and track is defined (first occurence of track)
            frame = (np.cumsum(not_none, axis=-1) == 1) & (not_none == True)
            x, y = self.Track_array[j][frame].T
            try:
                self.db.setMarker(x=x, y=y,
                                  type=self.stiched_type,
                                  image=self.db.getImages(frame=np.where(frame)[0])[0],
                                  text="%s & %s"%(appended_id, erased_id))
            except TypeError:
                print("could not set debug marker!")
        not_none = ~(np.all(np.isnan(self.Track_array[appended_id]), axis=-1)&
                     np.all(np.isnan(self.Track_array[erased_id]), axis=-1))
        self.Track_array[appended_id][not_none] = np.nanmean([self.Track_array[appended_id][not_none],
                                                              self.Track_array[erased_id][not_none]], axis=0)
        self.Track_array[erased_id] = np.nan
        print("stitch!", appended_id, erased_id)

    def stitch(self):
        pass

    def write_to_db(self, db, marker_type):
        ids = []
        x = []
        y = []
        frames = []
        types = []
        Frames = np.cumsum(np.ones_like(self.Track_array[0,:,0]), axis=-1)
        for k, array in enumerate(self.Track_array):
            print(k)
            if np.any(~np.isnan(array)):
                db_track = db.setTrack(type=marker_type)
                print("as ", db_track.id)
                mask = np.all(~np.isnan(array), axis=-1)
                ids.extend(db_track.id * np.ones_like(Frames)[mask])
                x.extend(array[:,0][mask])
                y.extend(array[:,1][mask])
                frames.extend(Frames[mask].astype(int)-1)
                types.extend([marker_type for i in  np.ones_like(Frames)[mask]])
                db.setMarkers(frame=frames, x=x, y=y, type=types, track=ids)
                ids = []
                x = []
                y = []
                frames = []
                types = []
        # frame = None, x = None, y = None, type = None, track = None

class Richter_Stitcher(new_Stitcher):
    def __init__(self, greedy=False,
        inner_dist_threshold=25,
        outer_dist_threshold=50,
        inner_temporal_threshold=10,
        outer_temporal_threshold=20):
        super(Richter_Stitcher, self).__init__(greedy=greedy)

        self.inner_dist_threshold = float(inner_dist_threshold)
        self.outer_dist_threshold = float(outer_dist_threshold)
        self.inner_temporal_threshold = int(inner_temporal_threshold)
        self.outer_temporal_threshold = int(outer_temporal_threshold)

    def stitch(self, dummy=False):
        # array of temporal and spatial differences between all tracks
        temp = self.temporal_diff()[:,:,0]
        space = self.spatial_diff()
        # dict to store results
        row_col = {}
        #Do for each track
        for i in self.Track_Dict:
            print("doing ", i)
            # dict to store ids of matching candidates
            all_ids = {i}
            # ids of horizontal candidates (match track i)
            ids = np.where((temp[i]>0)&(temp[i]<self.outer_temporal_threshold))[0]
            all_ids.update(ids)
            # ids of vertical candidates (match tracks in ids)
            ids2 = np.where((temp[ids]>0)&(temp[ids]<self.outer_temporal_threshold))[1]
            all_ids.update(ids2)
            # dictionary to translate beetween sliced array and main arrays
            inner_dict = dict(enumerate(sorted(all_ids)))

            inner_time = temp[sorted(all_ids)][:, sorted(all_ids)]

            inner_array = space[sorted(all_ids)][:, sorted(all_ids)]
            inner_array[np.diag(np.ones(len(all_ids), dtype=bool))] = np.nextafter(np.inf, 0.)
            inner_array[inner_array>self.outer_dist_threshold] = np.nextafter(np.inf, 0.)
            inner_array[(inner_time>0)&(inner_time<self.outer_temporal_threshold)] = np.nextafter(np.inf, 0.)

            if self.greedy:
                r, c = greedy_assignment(inner_array)
            else:
                r, c = linear_sum_assignment(inner_array)

            for R,C in zip(r,c):
                # translate to entries in main arrays
                R = inner_dict[R]
                C = inner_dict[C]
                if ((space[C, R] <= self.inner_dist_threshold) and
                        (temp[C, R] <= self.inner_temporal_threshold) and
                        (temp[C, R] >= 0)):
                    if R not in row_col:
                        row_col.update({R: C})
                        print("Match!")
                    elif space[row_col[R], R] > space[C,R]:
                        row_col.update({R: C})
                        print("Overwriting Match!")

        rows, cols = np.array(list(row_col.items())).T
        self._stitch_from_rc(rows, cols, dummy=dummy)
        # for r in row_col:
        #     self.__stitch__(self.Track_Dict[r], self.Track_Dict[row_col[r]])


class new_DistanceStitcher(new_Stitcher):
    def __init__(self,*args, max_velocity, max_frames=3, **kwargs):
            super(new_DistanceStitcher, self).__init__(*args, **kwargs)
            self.MaxV = float(max_velocity)
            self.MaxF = int(max_frames)
            self.name ="DistanceStitcher"

    def stitch(self, dummy=False):
        s = self.spatial_diff()
        t = self.temporal_diff()[:,:,0]
        cost = s/t
        # velocity[np.abs(velocity)>=self.MaxV] = np.nan
        cost[np.diag(np.ones(len(cost), dtype=bool))] = self.MaxV
        cost[cost>self.MaxV]=self.MaxV
        cost[cost<0]=self.MaxV
        cost[np.abs(t)>self.MaxF]=self.MaxV
        self._stitch_(cost, threshold=self.MaxV, dummy=dummy)

class new_MotionStitcher(new_Stitcher):
    def __init__(self, window_length, max_diff, *args, max_distance=np.inf, max_delay=np.inf, **kwargs):
        self.kernel = np.ones(int(window_length))/float(window_length)
        super(new_MotionStitcher, self).__init__(*args, **kwargs)
        self.max_diff = float(max_diff)
        self.MaxDist=float(max_distance)
        self.MaxDelay=float(max_delay)
        self.name = "MotionStitcher"

    def stitch(self, dummy=False):
        cost = np.linalg.norm(self.vel_delta(kernel=self.kernel), axis=-1)
        s = self.pos_delta()
        s_abs = np.linalg.norm(s, axis=-1)
        t = self.temporal_diff()[:,:,0]
        cost[cost>self.max_diff] = self.max_diff
        cost[np.isnan(cost)] = self.max_diff
        cost[s_abs>self.MaxDist] =  self.max_diff
        # cost[np.abs(t)>self.MaxDelay] =  self.max_diff
        cost[t>self.MaxDelay] =  self.max_diff
        cost[t<=0] = self.max_diff
        self._stitch_(cost, threshold=self.max_diff, dummy=dummy)

class new_expDistanceStitcher(new_Stitcher):
    def __init__(self, k, w, *args, max_distance=np.inf, max_delay=np.inf, dim=(1,), **kwargs):
        super(new_expDistanceStitcher, self).__init__(*args, **kwargs)
        k=np.array(k, ndmin=2)
        if len(k)>1:
            self.K = np.array(k, ndmin=2)
        else:
            self.K = k * np.ones(dim).T
        self.W = float(w)
        self.MaxDist=float(max_distance)
        self.MaxDelay=float(max_delay)
        self.Dim = dim
        self.name ="expDistanceStitcher"

    def stitch(self, dummy=False):
        s = self.pos_delta()
        s_abs = np.linalg.norm(s,axis=-1)
        t = self.temporal_diff()[:,:,None]
        cost = np.exp((np.dot(self.K, np.abs(s)) + self.W*np.abs(t)))[0].T[0].T
        max_cost = np.exp((np.dot(self.K, np.abs(self.MaxDist)) + self.W * np.abs(self.MaxDelay)))[0,0]
        cost[np.diag(np.ones(len(cost), dtype=bool))] = max_cost
        cost[s_abs>self.MaxDist] = max_cost
        cost[t.T[0].T>self.MaxDelay] = max_cost
        cost[t.T[0].T<0] = max_cost
        self._stitch_(cost, threshold=max_cost, dummy=dummy)

class Stitcher(object):
    def __init__(self, greedy=False):
        self.Tracks = {}
        self.db = None
        self.track_dict = {}
        self.name = "Stitcher"
        self.greedy = bool(greedy)

    def add_PT_Tracks(self, tracks):
        self.Tracks.update(add_PT_Tracks(tracks))

    def add_PT_Tracks_from_Tracker(self, tracks):
        track_dict, tracks_object = add_PT_Tracks_from_Tracker(tracks)
        self.track_dict.update(tracks_object)

    def load_tracks_from_clickpoints(self, path, type):
        db_object, tracks_object = load_tracks_from_clickpoints(path, type, tracker_name=None)
        self.db = db_object
        self.stiched_type = self.db.setMarkerType(name=self.name, color="F0F0FF")
        self.Tracks = tracks_object
        self.track_dict = db_object.track_dict
        self.db_track_dict = db_object.db_track_dict
        print("Tracks loaded!")

    def load_measurements_from_clickpoints(self, path, type, measured_variables=["PositionX", "PositionY"]):
        db_object, tracks_object = load_measurements_from_clickpoints(path, type, measured_variables=measured_variables)
        self.db = db_object
        self.stiched_type = self.db.setMarkerType(name="Stitched", color="F0F0FF")
        self.Tracks = tracks_object
        self.track_dict = db_object.track_dict
        print("Tracks loaded!")

    def vel_delta(self, kernel=np.ones(1)):
        first_vel = np.zeros(
            (len(self.Tracks), len(self.Tracks), self.Tracks[min(self.Tracks.keys())].Model.State_dim, 1))
        last_vel = np.zeros(
            (len(self.Tracks), len(self.Tracks), self.Tracks[min(self.Tracks.keys())].Model.State_dim, 1))
        for i in range(len(self.Tracks)):
            X = self.Tracks[self.track_dict[i]].X
            if len(X)<1:
                continue
            frames = sorted(list(X.keys()))
            l = min(len(kernel), len(frames))
            first_vals = np.array([X[k] for k in frames[-1-l:]])/np.array([k for k in frames[-1-l:]])
            last_vals = np.array([X[k] for k in frames[:l+1]])/np.array([k for k in frames[:l+1]])

            first_vals = first_vals[1:]-first_vals[:-1]
            last_vals = last_vals[1:]-last_vals[:-1]
            try:
                first_vel[:,i] = np.array([np.dot(val, kernel[:l]) for val in first_vals.T[0]]).T[:, None]
            except ValueError:
                first_vel[:,i] = np.array([np.dot(val, kernel[:l-1]) for val in first_vals.T[0]]).T[:, None]
            try:
                last_vel[i:,] = np.array([np.dot(val, kernel[:l]) for val in last_vals.T[0]]).T[:, None]
            except ValueError:
                last_vel[i:,] = np.array([np.dot(val, kernel[:l-1]) for val in last_vals.T[0]]).T[:, None]
        return last_vel-first_vel

    def pos_delta(self, kernel=np.ones(1)):
        first_pos = np.zeros(
            (len(self.Tracks), len(self.Tracks), self.Tracks[min(self.Tracks.keys())].Model.State_dim, 1))
        last_pos = np.zeros(
            (len(self.Tracks), len(self.Tracks), self.Tracks[min(self.Tracks.keys())].Model.State_dim, 1))
        for i in range(len(self.Tracks)):
            X = self.Tracks[self.track_dict[i]].X
            if len(X)<1:
                continue
            frames = sorted(list(X.keys()))
            l = min(len(kernel), len(frames))
            first_vals = np.array([X[k] for k in frames[:l]])
            last_vals = np.array([X[k] for k in frames[-l:]])
            first_pos[:,i] = np.array([np.dot(val, kernel[:l]) for val in first_vals.T[0]]).T[:, None]
            last_pos[i,:] = np.array([np.dot(val, kernel[:l]) for val in last_vals.T[0]]).T[:, None]
        return last_pos-first_pos

    def spatial_diff(self):
        return np.linalg.norm(self.pos_delta(), axis=(2,3))

    def temporal_diff(self):
        first_time = np.zeros((len(self.Tracks), len(self.Tracks)))
        last_time = np.zeros((len(self.Tracks), len(self.Tracks)))
        for i in range(len(self.Tracks)):
            if len(self.Tracks[self.track_dict[i]].X) <1:
                continue
            first_time[:,i] = min([k for k in self.Tracks[self.track_dict[i]].X])
            last_time[i,:] = max([k for k in self.Tracks[self.track_dict[i]].X])
        return first_time-last_time

    def _stitch_(self, cost, threshold=np.inf):
        print(np.mean(cost>=threshold))
        # if np.any(cost==np.inf):
        #     cost[cost==np.inf]=np.amax(cost[cost!=np.inf])
        #     if threshold==np.inf:
        #         threshold=np.amax(cost)
        cost [cost==np.amax(cost)] == np.nextafter(np.inf, 0)/(max(cost.shape)**2)

        if self.greedy:
            rows, cols = greedy_assignment(cost)
        else:
            rows, cols = linear_sum_assignment(cost)

        print("Assigned Tracks!")
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
        times = set(self.Tracks[i].X.keys())
        times.update(self.Tracks[j].X.keys())
        self.Tracks[idx].X.update(self.Tracks[n_idx].X)
        self.Tracks[idx].X_error.update(self.Tracks[n_idx].X_error)
        self.Tracks[idx].Predicted_X.update(self.Tracks[n_idx].Predicted_X)
        self.Tracks[idx].Predicted_X_error.update(self.Tracks[n_idx].Predicted_X_error)
        self.Tracks[idx].Measurements.update(self.Tracks[n_idx].Measurements)
        if self.db:
            self.db.getTrack(self.db_track_dict[idx]).merge(self.db_track_dict[n_idx], mode="average")
            iii = min(self.Tracks[n_idx].X)
            m = self.Tracks[idx].X[iii]
            if isinstance(self.Tracks[idx], all_filters):
                m = self.Tracks[idx].Model.measure(m)
                i_x = self.Tracks[idx].Model.Measured_Variables.index("PositionX")
                i_y = self.Tracks[idx].Model.Measured_Variables.index("PositionY")
            self.db.setMarker(x=m[i_y], y=m[i_x], type=self.stiched_type, image=self.db.getImages(frame=iii)[0])
        print("stitch!", i, j)
        self.Tracks.pop(n_idx)


    def stitch(self):
        pass

    def save_tracks_to_db(self, path, type, function=None, flip=False):
        if function is None:
            function = lambda x : x
        db = DataFileExtended(path)

        track_set = []
        for track in self.Tracks:
            db_track = db.setTrack(type=type)
            for i in self.Tracks[track].X:
                meas = self.Tracks[track].Measurements[i]
                pos = self.Tracks[track].Model.vec_from_meas(meas)
                pos = function(pos)
                if flip:
                    track_set.append(dict(track=db_track.id, type=type, frame=i, x=pos[0], y=pos[1]))
                else:
                    track_set.append(dict(track=db_track.id, type=type, frame=i, x=pos[1], y=pos[0]))
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
        self.name ="DistanceStitcher"

    def stitch(self):
        s = self.spatial_diff()
        t = self.temporal_diff()
        cost = s/t
        # velocity[np.abs(velocity)>=self.MaxV] = np.nan
        cost[np.diag(np.ones(len(cost), dtype=bool))] = self.MaxV
        cost[cost>self.MaxV]=self.MaxV
        cost[cost<0]=self.MaxV
        cost[np.abs(t)>self.MaxF]=self.MaxV
        self._stitch_(cost, threshold=self.MaxV)

class MotionStitcher(Stitcher):
    def __init__(self, window_length, max_diff, max_distance=np.inf, max_delay=np.inf):
        self.kernel = np.ones(int(window_length))/float(window_length)
        super(MotionStitcher, self).__init__()
        self.max_diff = float(max_diff)
        self.MaxDist=float(max_distance)
        self.MaxDelay=float(max_delay)
        self.name ="MotionStitcher"

    def stitch(self):
        cost = np.linalg.norm(self.vel_delta(kernel=self.kernel), axis=(2,3))
        s = self.pos_delta()
        s_abs = np.linalg.norm(s,axis=(2,3))
        t = self.temporal_diff()
        cost[cost>self.max_diff] = self.max_diff
        cost[np.isnan(cost)] = self.max_diff
        cost[s_abs>self.MaxDist] =  self.max_diff
        # cost[np.abs(t)>self.MaxDelay] =  self.max_diff
        cost[t>self.MaxDelay] =  self.max_diff
        cost[t<=0] = self.max_diff
        self._stitch_(cost, threshold=self.max_diff)

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
        self.name ="expDistanceStitcher"

    def stitch(self):
        s = self.pos_delta()
        s_abs = np.linalg.norm(s,axis=(2,3))
        t = self.temporal_diff()[:,:,None]
        cost = np.exp((np.dot(self.K, np.abs(s)) + self.W*np.abs(t)))[0].T[0].T
        max_cost = np.exp((np.dot(self.K, np.abs(self.MaxDist)) + self.W * np.abs(self.MaxDelay)))[0,0]
        cost[np.diag(np.ones(len(cost), dtype=bool))] = max_cost
        cost[s_abs>self.MaxDist] = max_cost
        cost[t.T[0].T>self.MaxDelay] = max_cost
        cost[t.T[0].T<0] = max_cost
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
        self.name ="Heublein_Stitcher"

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
            if len(track_data.X)<1:
                continue
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
        self.end_frame = max([max(track.Measurements) for track in self.Tracks.values() if len(track.Measurements)])
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
                        if frame in track1.Measurements:
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

class new_Splitter(new_Stitcher):
    def __init__(self):
        super(new_Splitter, self).__init__()
        self.name = "Splitter"

    def v(self):
        Frames = np.cumsum(np.ones_like(self.Track_array[:,:,0]), axis=-1)[:,:,None]
        return np.diff(self.Track_array, axis=1)/np.diff(Frames, axis=1)

    def pos(self):
        return np.nancumsum(self.v(), axis=1)

    def abs_v(self):
        return np.linalg.norm(self.v(), axis=-1)

    def abs_pos(self):
        return np.linalg.norm(self.pos(), axis=-1)

    def _split_(self, row, collumn):
        Frames = np.cumsum(np.ones_like(self.Track_array[0,:,0]), axis=-1)
        for r,c in zip(row,collumn):
            track_id = self.Track_Dict[r]
            time = c
            new_Track = np.ones_like(self.Track_array[r])[None,:,:]*np.nan
            new_Track[0, c:] = self.Track_array[r,c:]
            self.Track_array = np.vstack((self.Track_array, new_Track))
            self.Track_array[r, c:] = np.nan
            if self.db is not None:
                db_track = self.db.getTrack(track_id)
                db_times = np.array([m.image.sort_index for m in db_track.markers])
                # db_times = Frames[np.all(~np.isnan(self.Track_array[r]), axis=-1)]
                if len(db_times)<1:
                    pass
                else:
                    db_time = db_times[np.argmin((db_times-time)**2)]
                    m = self.db.getMarkers(track=track_id, frame=int(db_time))[0]
                    self.db.setMarker(x=m.x, y=m.y, frame=db_time, type=self.stiched_type)
                    db_track.split(m)
                self.Track_Dict.update({len(self.Track_array): db_track.id})
            print("splitted ", track_id)


class new_Motion_Splitter(new_Splitter):
    def __init__(self,  mode_threshold=100, window_size=3):
        super(new_Motion_Splitter, self).__init__()
        self.name = "Motion_Splitter"
        self.mode_threshold = float(mode_threshold)
        self.window_size = int(window_size)

    def switch_mode(self):
        window_size = self.window_size
        margin = 2*window_size-1
        win = np.ones(window_size)
        #best practice kernel computing second derivative
        kernel = np.convolve(np.convolve(np.convolve(win,np.hstack((win,-win))),[1,-1]), win)
        kernel = kernel

        # function reacts quadratic to window-size change -> rescale to fit thresholds
        sm = np.abs([np.convolve(x, kernel, mode="same") for x in self.abs_pos()])*9/window_size**2
        sm[:, :margin] = 0.
        sm[:, -margin:] = 0.
        return sm

    def _temp_local_max_(self, array, threshold):
        mat = np.array(array)
        mat[mat < threshold] = np.amin(mat)
        b = np.zeros_like(mat, dtype=bool)
        b[:, :-1] = mat[:, 1:] < mat[:, :-1]
        b[:, 1:] &= mat[:, :-1] < mat[:, 1:]
        return np.where(b)

    def split(self):
        cost = self.switch_mode()

        r, c = self._temp_local_max_(cost, self.mode_threshold)
        self._split_(r, c)

class Splitter(Stitcher):
    def __init__(self):
        super(Splitter, self).__init__()
        self.name = "Splitter"

    def v(self):
        T = set()
        for track in self.Tracks.values():
            T.update(track.X.keys())
        t_dict=dict(enumerate(T))
        x_1 = np.zeros((len(self.track_dict), len(t_dict), 2, 1))

        for i in range(len(self.Tracks)):
            X = self.Tracks[self.track_dict[i]].X
            if len(X)<1:
                continue
            x_1[i, sorted(X.keys())[1:]] = np.array(list(X.values()), dtype=float)[1:,:2]-np.array(list(X.values()), dtype=float)[:-1,:2]
        return x_1.T[0].T

    def pos(self):
        return np.cumsum(self.v(), axis=1)

    def abs_v(self):
        return np.linalg.norm(self.v(), axis=-1)

    def abs_pos(self):
        return np.linalg.norm(self.pos(), axis=-1)

    def switch_mode(self):
        # sm = np.array([np.abs(np.convolve(np.abs(x - np.cumsum(x) / np.arange(len(x))),
        #                              [-0.1, 0., 0., 0., 0., 0.1], mode="same")) for x in self.abs_pos()])
        # kernel = np.convolve(np.convolve((np.convolve(np.ones(window_size),np.ones(window_size)),np.ones(window_size)),[1.,0,0,-1])
        sm = np.abs([np.convolve(x, [ 1,  2,  3,  0, -3, -6, -3,  0,  3,  2,  1], mode="same") for x in self.abs_pos()])
        return sm

    def _temp_local_max_(self, array, threshold):
        mat = np.array(array)
        mat[mat<threshold] = np.amin(mat)
        b = np.zeros_like(mat, dtype=bool)
        b[:, :-1] = mat[:, 1:] < mat[:, :-1]
        b[:, 1:] &= mat[:, :-1] < mat[:, 1:]
        return np.where(b)

    def _split_(self, row, collumn):
        for r,c in zip(row,collumn):
            track_id = self.db_track_dict[self.track_dict[r]]
            time = c
            new_id = max(self.Tracks.keys())+1
            new_track = copy(self.Tracks[track_id])
            self.Tracks.update({new_id: new_track})
            times = list(self.Tracks[track_id].X.keys())
            for t in times:
                if t<time:
                    new_track.downfilter(t)
                else:
                    self.Tracks[track_id].downfilter(t)
            if self.db is not None:
                db_track = self.db.getTrack(track_id)
                db_times = np.array([m.image.sort_index for m in db_track.markers])
                if len(db_times)<1:
                    pass
                else:
                    db_time = db_times[np.argmin((db_times-time)**2)]
                    m = self.db.getMarkers(track=track_id, frame=db_time)[0]
                    self.db.setMarker(x=m.x, y=m.y, frame=db_time, type=self.stiched_type)
                    db_track.split(m)
                print("splitted ", track_id)

    # def temporal_delta(self):
    #     e = self.existence()
    #     o = (e.astype(int)-(~e).astype(int))
    #     out = np.cumsum(o, axis=1)
    #     return (out-np.amin(out, axis=1)) *o
    #     # T = set()
    #     # for track in self.Tracks:
    #     #     T.update(self.Tracks[track].X.keys())
    #     # temporal_diff = np.ones((len(self.Tracks), int(max(T)-min(T))))*np.arange(int(min(T)), int(max(T)))
    #     # temporal_diff -= temporal_diff[:, 0][:, None]
    #     # return temporal_diff

    def existence(self):
        T = set()
        for track in self.Tracks.values():
            T.update(track.X.keys())
        t_dict=dict(enumerate(T))
        e = np.zeros((len(self.track_dict), len(t_dict), 1), dtype=bool)

        for i in range(len(self.Tracks)):
            X = self.Tracks[self.track_dict[i]].X
            e[i, sorted(X.keys())] = True
        return e



