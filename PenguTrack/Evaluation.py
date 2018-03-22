from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import clickpoints

from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import Model
from PenguTrack import Filters
from PenguTrack.DataFileExtended import DataFileExtended,\
    add_PT_Tracks, add_PT_Tracks_from_Tracker, load_tracks_from_clickpoints, load_measurements_from_clickpoints,\
    reverse_dict

import inspect
import pandas

all_filters = tuple([v for v in dict(inspect.getmembers(Filters)).values() if type(v)==type])

class new_Evaluator(object):
    def __init__(self):
        self.System_Track_Dict = {}
        self.GT_Track_Dict = {}
        self.System_array = None
        self.GT_array = None
        self.system_db = None
        self.gt_db = None

    def load_tracks_from_clickpoints(self, path, type):
        db = clickpoints.DataFile(path)
        tracks = db.getTracks(type=type)
        track_dict = dict(enumerate([t.id for t in tracks]))
        array = np.array([db.db.execute_sql("select (SELECT x from marker as m WHERE m.image_id = i.id AND m.track_id=?) as x, (SELECT y from marker as m WHERE m.image_id = i.id AND m.track_id=?) as y from image as i order by sort_index",[track_dict[k],track_dict[k]]).fetchall() for k in sorted(track_dict.keys())], dtype=float)
        print("Tracks loaded!")
        return db, array, track_dict

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
        return track_dict, array

    def add_PT_Tracks_from_Tracker_to_GT(self, tracks):
        self.GT_Track_Dict, self.GT_array = self.add_PT_Tracks_from_Tracker(tracks)

    def add_PT_Tracks_from_Tracker_to_System(self, tracks):
        self.System_Track_Dict, self.System_array = self.add_PT_Tracks_from_Tracker(tracks)

    def load_GT_tracks_from_clickpoints(self, path, type):
        self.gt_db, self.GT_array, self.GT_Track_Dict = self.load_tracks_from_clickpoints(path, type)

    def load_System_tracks_from_clickpoints(self, path, type):
        self.system_db, self.System_array, self.System_Track_Dict = self.load_tracks_from_clickpoints(path, type)

    def match(self, method=None):
        if method is None or method == "default" or method == "euclidic":
            self.Matches = {}
            for i, t in enumerate(self.GT_array):
                matched = np.where(np.sum(np.linalg.norm(t-self.System_array, axis=-1)<(self.Object_Size/2.), axis=-1)>self.PointThreshold)[0]
                self.Matches.update({self.GT_Track_Dict[i]: [self.System_Track_Dict[m] for m in matched]})
                print("Matched ", self.GT_Track_Dict[i])
        else:
            raise ValueError("No matching method %s known!"%method)

class new_Yin_Evaluator(new_Evaluator):
    def __init__(self, object_size, *args, **kwargs):
        self.TempThreshold = kwargs.pop("temporal_threshold", 0.1)
        self.SpaceThreshold = kwargs.pop("spacial_threshold", 0.1)
        self.PointThreshold = kwargs.pop("point_threshold", -1)
        self.Object_Size = object_size
        self.Matches = {}
        super(new_Yin_Evaluator, self).__init__(*args, **kwargs)
        
    def match(self, method=None):
        if method == "bbox":
            self.Matches = {}
            o = self.Object_Size / 2.
            for i, t in enumerate(self.GT_array):
                r, l = np.sort([self.System_array[:, :, 0], np.tile(t[:, 0], (len(self.System_Track_Dict), 1))], axis=0)
                l -= o
                r += o
                t, b = np.sort([self.System_array[:, :, 1], np.tile(t[:, 1], (len(self.System_Track_Dict), 1))], axis=0)
                t += o
                b -= o
                I = (t - b) * (r - l) * (((r - l) > 0) & ((t - b) > 0))
                O = I/(2*self.Object_Size**2-I)
                matched = np.where(np.nansum(O>self.SpaceThreshold, axis=-1)>self.PointThreshold)[0]
                self.Matches.update({self.GT_Track_Dict[i]: [self.System_Track_Dict[m] for m in matched]})
                print("Matched ", self.GT_Track_Dict[i])
        else:
            super(new_Yin_Evaluator, self).match(method=method)

    def temporal_overlap(self, sys_track, gt_track):
        sys_id = reverse_dict(self.System_Track_Dict)[sys_track]
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_frames = np.all(~np.isnan(self.System_array[sys_id]), axis=-1)
        gt_frames = np.all(~np.isnan(self.GT_array[gt_id]), axis=-1)
        return np.sum(sys_frames & gt_frames, dtype=float)/np.sum(sys_frames | gt_frames, dtype=float)

    def spatial_intersect(self, sys_track, gt_track):
        sys_id = reverse_dict(self.System_Track_Dict)[sys_track]
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        o = self.Object_Size/2.
        r, l = np.sort([self.System_array[sys_id,:,0],self.GT_array[gt_id,:,0]], axis=0)
        l -= o
        r += o
        t, b = np.sort([self.System_array[sys_id,:,1],self.GT_array[gt_id,:,1]], axis=0)
        t += o
        b -=o
        return (t - b) * (r - l) * (((r - l) > 0) & ((t - b) > 0))

    def spatial_overlap(self, sys_track, gt_track):
        I = self.spatial_intersect(sys_track, gt_track)
        return I/(2*self.Object_Size**2-I)

    def spatial_dist(self, sys_track, gt_track):
        sys_id = reverse_dict(self.System_Track_Dict)[sys_track]
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        return np.linalg.norm(self.GT_array[gt_id] - self.System_array[sys_id], axis=-1)

    def TE(self, sys_track, gt_track):
        return self.spatial_dist(sys_track, gt_track)

    def TME(self, sys_track, gt_track):
        return np.nanmean(self.TE(sys_track, gt_track))

    def TMED(self, sys_track, gt_track):
        return np.std(self.TE(sys_track, gt_track))

    def TMEMT(self, gt_track):
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_ids = [reverse_dict(self.System_Track_Dict)[s] for s in self.Matches[gt_track]]
        mask = np.zeros(len(self.System_Track_Dict), dtype=bool)
        mask[sys_ids] = True
        return (np.nansum(np.nanmean(np.linalg.norm(self.GT_array[gt_id]-self.System_array, axis=-1), axis=-1)[mask], axis=-1)/
        np.nansum(np.sum(np.all(~np.isnan(self.System_array), axis=-1), axis=-1)[mask], axis=-1))

    def TMEMTD(self, gt_track):
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_ids = [reverse_dict(self.System_Track_Dict)[s] for s in self.Matches[gt_track]]
        mask = np.zeros(len(self.System_Track_Dict), dtype=bool)
        mask[sys_ids] = True
        return (np.nansum(np.nanstd(np.linalg.norm(self.GT_array[gt_id]-self.System_array, axis=-1), axis=-1)[mask], axis=-1)/
        np.nansum(np.sum(np.all(~np.isnan(self.System_array), axis=-1), axis=-1)[mask], axis=-1))

    def TC(self, gt_track):
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_ids = [reverse_dict(self.System_Track_Dict)[s] for s in self.Matches[gt_track]]
        mask = np.zeros(len(self.System_Track_Dict), dtype=bool)
        mask[sys_ids] = True

        sys_frames = np.all(~np.isnan(self.System_array), axis=-1)
        gt_frames = np.all(~np.isnan(self.GT_array[gt_id]), axis=-1)
        return np.sum(np.any(sys_frames&mask[:, None]&gt_frames[None,:], axis=0), dtype=float)/np.sum(gt_frames, dtype=float)

    def R2(self, gt_track):
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_ids = [reverse_dict(self.System_Track_Dict)[s] for s in self.Matches[gt_track]]
        mask = np.zeros(len(self.System_Track_Dict), dtype=bool)
        mask[sys_ids] = True

        sys_frames = np.all(~np.isnan(self.System_array), axis=-1)
        gt_frames = np.all(~np.isnan(self.GT_array[gt_id]), axis=-1)
        return np.sum(np.sum(sys_frames&mask[:, None], axis=0)==2, dtype=float)/np.sum(gt_frames, dtype=float)

    def R3(self, gt_track):
        gt_id = reverse_dict(self.GT_Track_Dict)[gt_track]
        sys_ids = [reverse_dict(self.System_Track_Dict)[s] for s in self.Matches[gt_track]]
        mask = np.zeros(len(self.System_Track_Dict), dtype=bool)
        mask[sys_ids] = True

        sys_frames = np.all(~np.isnan(self.System_array), axis=-1)
        gt_frames = np.all(~np.isnan(self.GT_array[gt_id]), axis=-1)
        return np.sum(np.sum(sys_frames&mask[:, None], axis=0)==3, dtype=float)/np.sum(gt_frames, dtype=float)

    def CTM(self, gt_track):
        ct = []
        l = []
        for sys_track in self.Matches[gt_track]:
            o = self.spatial_overlap(gt_track, sys_track)
            mask = ~np.isnan(o)
            ct.extend(o[mask])
            l.append(np.sum(mask))
        return np.nansum(ct)/np.nansum(l)

    def CTD(self, gt_track):
        ct = []
        l = []
        for sys_track in self.Matches[gt_track]:
            o = self.spatial_overlap(gt_track, sys_track)
            mask = ~np.isnan(o)
            ct.extend(np.std(o[mask])*np.sum(mask))
            l.append(np.sum(mask))
        return np.nansum(ct)/np.nansum(l)

    def LT(self, gt_track):
        pass

    def IDC(self, gt_track):
        return len(self.Matches[gt_track])

    def MIX(self, gt_track):
        return np.sum([np.sum([k in self.Matches[b] for b in self.Matches if b!=gt_track]) for k in self.Matches[gt_track]])

    def mixture(self, gt_track):
        return self.MIX(gt_track)


class Evaluator(object):
    def __init__(self):
        self.System_Tracks = {}
        self.GT_Tracks = {}
        self.gt_db = None
        self.system_db = None
        self.gt_track_dict = {}
        self.system_track_dict = {}
        self.DF = None

    def built_DataFrame(self):

        super_list = [[i, j, *self.System_Tracks[i].X[j].T[0], "System"]
                                                     for i in self.System_Tracks.keys()
                                                     for j in self.System_Tracks[i].X.keys()]
        super_list.extend([[i, j, *self.GT_Tracks[i].X[j].T[0], "GT"]
                                                     for i in self.GT_Tracks.keys()
                                                     for j in self.GT_Tracks[i].X.keys()])
        self.DF = pandas.DataFrame(super_list,columns=["Track","Frame","X","Y", "Type"])

    def add_PT_Tracks(self, tracks):
        return add_PT_Tracks(tracks)

    def add_PT_Tracks_from_Tracker(self, tracks):
        tracks_dict, tracks_object = add_PT_Tracks_from_Tracker(tracks)
        return tracks_object

    def load_tracks_from_clickpoints(self, path, type, use_cp_tracks=False):
        db_object, tracks_object = load_tracks_from_clickpoints(path, type, tracker_name=None,
                                                                use_cp_tracks=use_cp_tracks)
        print("Tracks loaded!")
        return db_object, tracks_object


    def load_measurements_from_clickpoints(self, path, type, measured_variables=["PositionX", "PositionY"]):
        db_object, tracks_object = load_measurements_from_clickpoints(path, type,
                                                                      measured_variables=measured_variables)
        return db_object, tracks_object

    def add_PT_Tracks_to_GT(self, tracks):
        self.GT_Tracks = self.add_PT_Tracks(tracks)

    def add_PT_Tracks_to_System(self, tracks):
        self.System_Tracks = self.add_PT_Tracks(tracks)

    def add_PT_Tracks_from_Tracker_to_GT(self, tracks):
        self.GT_Tracks = self.add_PT_Tracks_from_Tracker(tracks)

    def add_PT_Tracks_from_Tracker_to_System(self, tracks):
        self.System_Tracks = self.add_PT_Tracks_from_Tracker(tracks)

    def load_GT_tracks_from_clickpoints(self, path, type):
        self.gt_db, self.GT_Tracks = self.load_tracks_from_clickpoints(path, type)
        self.gt_track_dict = self.gt_db.track_dict

    def load_System_tracks_from_clickpoints(self, path, type, use_cp_tracks=False):
        self.system_db, self.System_Tracks =  self.load_tracks_from_clickpoints(path, type,
                                                                                use_cp_tracks=use_cp_tracks)
        self.system_track_dict = self.system_db.track_dict


class Yin_Evaluator(Evaluator):
    def __init__(self, object_size, *args, **kwargs):
        self.TempThreshold = kwargs.pop("temporal_threshold", 0.1)
        self.SpaceThreshold = kwargs.pop("spacial_threshold", 0.1)
        self.PointThreshold = kwargs.pop("point_threshold", -1)
        self.Object_Size = object_size
        self.Matches = {}
        super(Yin_Evaluator, self).__init__(*args, **kwargs)

    def temporal_overlap(self, trackA, trackB):
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        setA = set(trackA.X.keys())
        setB = set(trackB.X.keys())
        print("calculated delta T!")
        return float(len(setA.intersection(setB)))/len(setA.union(setB))
        # intersect = set(trackA.X.keys()).intersection(trackB.X.keys())
        # return max(len(intersect)/len(trackA.X.keys()), len(intersect)/len(trackB.X.keys()))
        # return len(set(trackA.X.keys()).intersection(trackB.X.keys()))/len(set(trackA.X.keys()).union(set(trackB.X.keys())))
        # return min(max(trackA.X.keys()),max(trackB.X.keys()))-max(min(trackA.X.keys()),min(trackB.X.keys()))

    def spatial_overlap(self, trackA, trackB):
        # id_A = trackA
        # id_B = trackB
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        # frames = set(self.System_DF[self.System_DF["Track"] == id_A].Frame).intersection(
        #     set(self.GT_DF[self.GT_DF["Track"] == id_B].Frame))
        frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
        if len(frames) == 0:
            return {}
        # A = self.System_DF[self.System_DF.Track==id_A& self.System_DF.Frame.isin(frames)]
        # B = self.GT_DF[self.GT_DF.Track==id_B& self.GT_DF.Frame.isin(frames)]
        # I = self._intersect_many_(A.X, A.Y, B.X, B.Y)
        X = np.array([[trackA.X[f][0], trackA.X[f][1], trackB.X[f][0], trackB.X[f][1]] for f in frames], ndmin=3)
        I = self._intersect_many_(*X.T[0])
        return dict(zip(frames, I/(2*self.Object_Size**2-I)))
        # pointsA = [trackA.X[f] for f in frames]
        # pointsB = [trackB.X[f] for f in frames]
        # return dict(zip(frames,[self._intersect_(p1,p2)/self._union_(p1,p2) for p1,p2 in zip(pointsA,pointsB)]))

    def _intersect_(self, p1, p2):
        o = self.Object_Size/2.
        l = max(p1[0]-o, p2[0]-o)
        r = min(p1[0]+o, p2[0]+o)
        t = min(p1[1]+o, p2[1]+o)
        b = max(p1[1]-o, p2[1]-o)
        return (t-b)*(r-l) if ((r-l)>0 and (t-b)>0) else 0

    def _intersect_many_(self, x0, y0, x1, y1):
        o = self.Object_Size/2.
        r,l = np.sort([x0,x1], axis=0)
        t,b = np.sort([y0,y1], axis=0)
        r += o
        l -= o
        t += o
        b -=o
        w = r-l
        h = t-b
        return w*h*(w > 0)*(h > 0)

    def _union_many_(self, x0, y0, x1, y1):
        return 2*self.Object_Size**2-self._intersect_many_(x0, y0, x1, y1)

    def _union_(self, p1, p2):
        return 2*self.Object_Size**2-self._intersect_(p1,p2)

    def _handle_Track_(self, track, tracks=None):
        if not isinstance(track, Filter):
            try:
                A = int(track)
            except TypeError:
                raise ValueError("%s is not a track!"%track)
            if A in self.GT_Tracks and (tracks is None or tracks==self.GT_Tracks):
                 return self.GT_Tracks[A]
            elif A in self.System_Tracks and (tracks is None or tracks==self.System_Tracks):
                return self.System_Tracks[A]
            else:
                raise ValueError("%s is not a track!" % track)
        else:
            return track

    def _handle_Track_id_(self, track, tracks=None):
        if isinstance(track, Filter):
            if track in self.GT_Tracks.values() and (tracks is None or tracks==self.GT_Tracks):
                 return [k for k in self.GT_Tracks if self.GT_Tracks[k]==track][0]
            elif track in self.System_Tracks.values() and (tracks is None or tracks==self.System_Tracks):
                return [k for k in self.System_Tracks if self.System_Tracks[k]==track][0]
            else:
                raise ValueError("%s is not a track!" % track)
        else:
            return int(track)

    def pandas_spatial_overlap(self, st, gt):
        frames = self.pandas_frames(st, gt)
        positions = self.pandas_positions(st, gt)
        grouped = positions.groupby("Frame")
        o = self.Object_Size / 2.
        r = grouped.X.min() + o
        l = grouped.X.max() - o
        t = grouped.Y.min() + o
        b = grouped.Y.max() - o
        w = r - l
        h = t - b
        A = w * h * (w > 0) * (h > 0)
        return A/(2*self.Object_Size**2-A)

    def pandas_frames(self, st, gt):
         return set(self.DF[(self.DF["Track"] == st) & (self.DF.Type == "System")].Frame).intersection(
            set(self.DF[(self.DF["Track"] == gt) & (self.DF.Type == "GT")].Frame))

    def pandas_positions(self, st, gt):
        frames = self.pandas_frames(st, gt)
        return self.DF[
            (((self.DF.Track == st) & (self.DF.Type == "System")) | ((self.DF.Track == gt) & (self.DF.Type == "GT")))
            & self.DF.Frame.isin(frames)]


    def corse_spatial_overlap(self, st, gt):
        positions_st = self.DF[((self.DF.Track == st) & (self.DF.Type == "System"))]
        positions_gt = self.DF[((self.DF.Track == gt) & (self.DF.Type == "GT"))]
        o = self.Object_Size/2.
        l1 = positions_gt.X.min()
        l2 = positions_st.X.min()
        r1 = positions_gt.X.max()
        r2 = positions_st.X.max()
        b1 = positions_gt.Y.min()
        b2 = positions_st.Y.min()
        t1 = positions_gt.Y.max()
        t2 = positions_st.Y.max()
        r = min(r1, r2) + o
        l = max(l1, l2) - o
        t = min(t1, t2) + o
        b = max(b1, b2) - o

        w = r - l
        h = t - b
        return (w * h * (w > 0) * (h > 0))>0

    def match(self):
        if self.DF is None:
            try:
                self.built_DataFrame()
            except:
                self._old_match_()

        pos_sys = self.DF[self.DF.Type == "System"].groupby("Track")
        r1 = pos_sys.X.max()
        l1 = pos_sys.X.min()
        t1 = pos_sys.Y.max()
        b1 = pos_sys.Y.min()

        pos_gt = self.DF[self.DF.Type == "GT"].groupby("Track")
        r2 = pos_gt.X.max()
        l2 = pos_gt.X.min()
        t2 = pos_gt.Y.max()
        b2 = pos_gt.Y.min()
        o = self.Object_Size/2.

        r = r1.values[None,:] < r2.values[:,None]
        r = r1.values[None,:]*r+r2.values[:,None]*(~r)+o

        l = l1.values[None,:] > l2.values[:,None]
        l = l1.values[None,:]*l+l2.values[:,None]*(~l)-o

        t = t1.values[None,:] < t2.values[:,None]
        t = t1.values[None,:]*t+t2.values[:,None]*(~t)+o

        b = b1.values[None,:] > b2.values[:,None]
        b = b1.values[None,:]*b+b2.values[:,None]*(~b)-o

        w = r - l
        h = t - b
        lookup = (w * h * (w > 0) * (h > 0))>0

        st_dict = dict(zip(l1.keys(), range(len(l1.keys()))))
        gt_dict = dict(zip(l2.keys(), range(len(l2.keys()))))
        for gt in self.GT_Tracks:
            m = []
            for st in self.System_Tracks:
                print("calculating!", st, gt)
                if not lookup[gt_dict[gt], st_dict[st]] == True:
                    continue
                spatial = self.pandas_spatial_overlap(st, gt)
                if self.PointThreshold > 0:
                    if self.temporal_overlap(st, gt)>self.TempThreshold and \
                        (spatial>self.SpaceThreshold).sum()>self.PointThreshold:
                        m.append(st)
                else:
                    if spatial.mean()>self.SpaceThreshold and self.temporal_overlap(st,gt)>self.TempThreshold:
                        m.append(st)
            self.Matches.update({gt: m})
                # B = self.DF[self.DF.Track == gt & self.DF.Frame.isin(frames) & self.DF.Type=="GT"]
                # I = self._intersect_many_(A.X, A.Y, B.X, B.Y)


        # spatial_overlap =

    def _old_match_(self):
        for gt in self.GT_Tracks:
            self.Matches.update({gt: [st for st in self.System_Tracks if (np.nanmean(list(self.spatial_overlap(st,gt).values()))>self.SpaceThreshold and
                                                self.temporal_overlap(st,gt)>self.TempThreshold)]})

    def FAT(self, trackA, trackB):
        pass

    def TF(self, trackA):
        pass

    def TE(self, trackA, trackB):
        if self.DF is not None:
            id_A = self._handle_Track_id_(trackA)
            id_B = self._handle_Track_id_(trackB)
            positions = self.pandas_positions(id_A, id_B)
            A = positions[positions.Type == "System"].set_index("Frame")
            B = positions[positions.Type == "GT"].set_index("Frame")
            return dict(((A.X-B.X)**2+(A.Y-B.Y)**2)**0.5)
        else:
            trackA = self._handle_Track_(trackA)
            trackB = self._handle_Track_(trackB)
            frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
            return dict([[f,np.linalg.norm(trackA.X[f]-trackB.X[f])] for f in frames])

    def TME(self, trackA, trackB):
        return np.nanmean(self.TE(trackA, trackB))

    def TMED(self, trackA, trackB):
        return np.std(self.TE(trackA, trackB))

    def TMEMT(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tmemt=[]
        l=[]
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            te = list(self.TE(trackA, trackB).values())
            tmemt.append(np.nanmean(te)*len(te))
            l.append(len(te))
        return np.nansum(tmemt)/np.nansum(l)

    def TMEMTD(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k] == trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tmemt = []
        l = []
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            te = list(self.TE(trackA, trackB).values())
            tmemt.append(np.std(te)*len(te))
            l.append(len(te))
        return np.nansum(tmemt) / np.nansum(l)

    def TC(self, trackA):
        trackA = self._handle_Track_(trackA, tracks=self.GT_Tracks)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tc = []
        for m in self.Matches[key]:
            tc.extend(self.System_Tracks[m].X.keys())
        try:
            return len(set(tc).intersection(set(trackA.X.keys())))/float(len(trackA.X))
        except ZeroDivisionError:
            return 0.

    def R2(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tc = []
        t2 = []
        for m in self.Matches[key]:
            t2.extend(set(self.System_Tracks[m].X.keys()).intersection(set(tc)))
            tc.extend(self.System_Tracks[m].X.keys())
        try:
            return len(set(t2))/float(len(trackA.X))
        except ZeroDivisionError:
            return 0.

    def R3(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tc = []
        t2 = []
        t3 = []
        for m in self.Matches[key]:
            t3.extend(set(self.System_Tracks[m].X.keys()).intersection(set(t2)))
            t2.extend(set(self.System_Tracks[m].X.keys()).intersection(set(tc)))
            tc.extend(self.System_Tracks[m].X.keys())
        try:
            return len(set(t3))/float(len(trackA.X))
        except ZeroDivisionError:
            return 0.

    def CTM(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        ct = []
        l = []
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            o = list(self.spatial_overlap(trackA, trackB).values())
            ct.extend(o)
            l.append(len(o))
        return np.nansum(ct)/np.nansum(l)

    def CTD(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        ct = []
        l = []
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            o=list(self.spatial_overlap(trackA, trackB).values())
            ct.append(len(o)*np.std(o))
            l.append(len(o))
        return np.nansum(ct)/np.nansum(l)

    def LT(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        try:
            first_sys = min([min(self._handle_Track_(m).X.keys()) for m in self.Matches[key]])
        except ValueError:
            return max(trackA.X.keys())-min(trackA.X.keys())
        return first_sys-min(trackA.X.keys())

    def IDC(self, trackA):
        id_A = self._handle_Track_id_(trackA)
        return len(self.Matches[id_A])
        # trackA = self._handle_Track_(trackA)
        # if trackA in self.GT_Tracks.values():
        #     key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        # else:
        #     raise KeyError("Track is not a Ground Truth Track")
        #
        # if self.DF is not None:
        #     # self.DF[]
        #     return len(self.Matches[key])
        # else:
        #     ct = []
        #     l = []
        #     frames = set()
        #     for m in self.Matches[key]:
        #         trackB = self._handle_Track_(m)
        #         frames.symmetric_difference_update(trackB.X.keys())
        #     IDC_j = 1
        #     id = None
        #     for f in frames:
        #         if id is None:
        #             id=[k for k in self.Matches[key] if f in self.System_Tracks[k].X]
        #             if len(id)>0:
        #                 id=id[0]
        #             else:
        #                 id = None
        #                 continue
        #         if not f in self.System_Tracks[id].X:
        #             IDC_j +=1
        #             ids=[k for k in self.Matches[key] if f not in self.System_Tracks[k].X]
        #             func = lambda i : len([fr for fr in frames if f<fr and fr<[fr for fr in frames if fr>f and not f in self.System_Tracks[id].X][0]])
        #             try:
        #                 id = max(ids, key=func)
        #             except ValueError:
        #                 id = None
        #                 continue
        #     return IDC_j

    def velocity(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        from datetime import timedelta
        t = np.array(sorted(trackA.X.keys()))
        x = np.array([trackA.X[f] for f in t], dtype=float)
        delta_x = x[1:]-x[:-1]
        try:
            delta_t = np.array([delta.seconds for delta in t[1:]-t[:-1]], dtype=float)
        except AttributeError:
            delta_t = np.array([delta for delta in t[1:]-t[:-1]], dtype=float)
        return delta_x.T/delta_t

    def activity(self, trackA):
        v=self.velocity(trackA)
        v_n = np.linalg.norm(v, axis=0)
        return np.nansum(v_n>self.Object_Size/2.)/float(len(v_n))

    def mixture(self, trackA):
        trackA = self._handle_Track_(trackA)
        a = self._handle_Track_id_(trackA)
        return np.sum([np.sum([k in self.Matches[b] for b in self.Matches if b!=a]) for k in self.Matches[a]])

class Alex_Evaluator(Yin_Evaluator):
    def __init__(self, object_size, camera_height, camera_tilt, image_height, sensor_height, focal_length, tolerance =1., **kwargs):
        self.Camera_Height = float(camera_height)
        self.Image_Height=float(image_height)
        self.Sensor_Height= float(sensor_height)
        self.Focal_Length = float(focal_length)
        self.Camera_Tilt = float(camera_tilt)
        self.Object_Size_m = float(object_size)
        self.Tolerance = float(tolerance)
        super(Alex_Evaluator, self).__init__(self.Object_Size_m, **kwargs)

    def size(self, y_px):
        dist_m = self.Camera_Height * np.tan(np.arctan(
            (self.Image_Height / 2. - y_px) / self.Image_Height * self.Sensor_Height / self.Focal_Length) - self.Camera_Tilt + np.pi / 2.)
        return 2 * np.arctan(self.Object_Size_m / 2. / dist_m) * self.Image_Height / (2 * np.arctan(self.Sensor_Height / 2. / self.Focal_Length))


    def _intersect_(self, p1, p2):
        o = self.Tolerance*self.size((p1[1]+p2[1])/2.)/2.
        l = max(p1[0]-o, p2[0]-o)
        r = min(p1[0]+o, p2[0]+o)
        t = min(p1[1]+o, p2[1]+o)
        b = max(p1[1]-o, p2[1]-o)
        return (t-b)*(r-l) if ((r-l)>0 and (t-b)>0) else 0

    def _union_(self, p1, p2):
        return 2*(self.Tolerance*self.size((p1[1]+p2[1])/2.))**2-self._intersect_(p1,p2)

    def relative_velocity(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        from datetime import timedelta
        t = np.array(sorted(trackA.X.keys()))
        x = np.array([trackA.X[f] for f in t], dtype=float)
        sizes = self.size(x[1:].T[1])
        delta_x = ((x[1:]-x[:-1]).T/sizes).T
        try:
            delta_t = np.array([delta.seconds for delta in t[1:]-t[:-1]], dtype=float)
        except AttributeError:
            delta_t = (t[1:] - t[:-1]).astype(float)
        return delta_x.T/delta_t

    def activity(self, trackA, thresh=.25):
        v=self.relative_velocity(trackA)
        v_n = np.linalg.norm(v, axis=0)
        return np.nansum(v_n>thresh)/float(len(v_n))

    def activities(self, trackA, thresh=.25):
        v=self.relative_velocity(trackA)
        v_n = np.linalg.norm(v, axis=0)
        v_max = np.amax(v_n)
        return v_n<thresh

    def rTE(self, trackA, trackB):
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
        return dict([[f, np.linalg.norm(trackA.X[f] - trackB.X[f])/self.size((trackA.X[f][1]+trackB.X[f][1])/2.)] for f in frames])

    def rTME(self, trackA, trackB):
        return np.nanmean(list(self.rTE(trackA,trackB).values()))

    def rTMED(self, trackA, trackB):
        return np.std(list(self.rTE(trackA,trackB).values()))

    def rTMEMT(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k] == trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tmemt = []
        l = []
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            te = self.rTE(trackA, trackB)
            tmemt.append(np.nanmean(list(te.values())) * len(te))
            l.append(len(te))
        return np.nansum(tmemt) / np.nansum(l)

    def rTMEMTD(self, trackA):
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k] == trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        tmemt = []
        l = []
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            te = self.rTE(trackA, trackB)
            tmemt.append(np.std(list(te.values())) * len(te))
            l.append(len(te))
        return np.nansum(tmemt) / np.nansum(l)


if __name__ == "__main__":
    import json
    evaluation = Yin_Evaluator(50., temporal_threshold=0., spacial_threshold=np.nextafter(0.,1.), point_threshold=50)#np.nextafter(0.,1.))
    evaluation.load_System_tracks_from_clickpoints(
        path="/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData3.cdb",
        # path=r"C:\Users\Alex\Documents/Promotion/AdeliesAntavia/AdelieTrack/AdelieData2.cdb",
        type="Track_Marker")
    evaluation.load_GT_tracks_from_clickpoints(path="/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData_GroundTruth_interpolated.cdb",
    # evaluation.load_GT_tracks_from_clickpoints(path=r"C:\Users\Alex\Documents/Promotion/AdeliesAntavia/AdelieTrack/AdelieData_GroundTruth.cdb",
                                               type="GroundTruth")
    # evaluation.built_DataFrame()
    evaluation.match()
    with open("/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData3_matches.json", "w") as fp:
        json.dump(evaluation.Matches, fp, indent=4, sort_keys=True)


    with open("/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData3_matches.json", "r") as fp:
        evaluation.Matches = json.load(fp, parse_int=int)
        # evaluation.Matches = dict([[int(l), loaded[l]] for l in loaded])
    print(evaluation.Matches)
    tmemt = []
    tmemtd = []
    tc = []
    r2 = []
    r3 = []
    ctm = []
    ctd = []
    lt = []
    idc = []
    a = []
    mix = []
    text_file = "/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData3.txt"
    with open(text_file, "w") as myfile:
        myfile.write("Tracking evaluation\n")


    def print_save(*args):
        try:
            string = "\t".join([str(l) for l in args])
        except TypeError:
            string = str(args)
        with open(text_file, "a") as my_file:
            my_file.write(string)
            my_file.write("\n")
        print(string)
    for m in evaluation.Matches:
        print_save("------------")
        print_save(m)
        tmemt.append(evaluation.TMEMT(m))
        # print("Debug1")
        # tmemtd.append(evaluation.TMEMTD(m))
        print("Debug2")
        tc.append(evaluation.TC(m))
        print("Debug3")
        # r2.append(evaluation.R2(m))
        # r3.append(evaluation.R3(m))
        # ctm.append(evaluation.CTM(m))
        # ctd.append(evaluation.CTD(m))
        # lt.append(evaluation.LT(m))
        mix.append(evaluation.mixture(m))
        print("Debug4")
        idc.append(evaluation.IDC(m))
        print("Debug5")
        # a.append(evaluation.activity(m))
        # print_save("TMEMT", tmemt[-1], tmemtd[-1])
        # print_save("TMEMT coeff", tmemtd[-1] / tmemt[-1])
        print_save("TC", tc[-1])
        # print_save("corrected TC", tc[-1] / a[-1])
        # print_save("R2", r2[-1])
        # print_save("R3", r3[-1])
        # print_save("CTM", ctm[-1])
        # print_save("CTD", ctd[-1])
        # print_save("LT", lt[-1])
        print_save("IDC", idc[-1])
        print_save("MIX", mix[-1])

