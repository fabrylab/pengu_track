from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import clickpoints

from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import Model
from PenguTrack import Filters
from PenguTrack.DataFileExtended import DataFileExtended

import inspect
import pandas

all_filters = tuple([v for v in dict(inspect.getmembers(Filters)).values() if type(v)==type])

class Evaluator(object):
    def __init__(self):
        self.System_Tracks = {}
        self.GT_Tracks = {}
        self.gt_db = None
        self.system_db = None
        self.gt_track_dict = {}
        self.system_track_dict = {}

    def add_PT_Tracks(self, tracks):
        tracks_object = {}
        for track in tracks:
            if len(tracks_object) == 0:
                i = 0
            else:
                i = max(tracks_object.keys()) + 1
                tracks_object.update({i: track})
        return tracks_object

    def add_PT_Tracks_from_Tracker(self, tracks):
        tracks_object = {}
        for j,i in enumerate(tracks):
            self.track_dict.update({j:i})
            tracks_object.update({i: tracks[i]})
        return tracks_object

    def load_tracks_from_clickpoints(self, path, type):
        tracks_object = {}
        db_object = clickpoints.DataFile(path)
        if "DataFileExtendedVersion" in [item.key for item in db_object.table_meta.select()]:
            while not db_object.db.is_closed():
                print("trying to close...")
                db_object.db.close()
            print("closed")
            db_object = DataFileExtended(path)
            tracks_object = db_object.tracker_from_db()[0].Filters
            self.track_dict = dict(zip(range(len(tracks_object)), tracks_object.keys()))
            db_object.track_dict = dict(db_object.db.execute_sql('select id, track_id from filter').fetchall())
        else:
            tracks = db_object.getTracks(type=type)
            for track in tracks:
                filter = Filters.Filter(Model())
                X_raw = db_object.db.execute_sql(
                    'select sort_index, y, x from marker m inner join image i on i.id ==m.image_id where m.track_id = ?',
                    [track.id])

                filter.X.update(dict([[v[0], np.array([[v[1]],[v[2]]], dtype=float)] for v in X_raw]))
                tracks_object.update({track.id:filter})
            self.track_dict = dict(zip(range(len(tracks_object)), tracks_object.keys()))
            db_object.track_dict = dict(zip(tracks_object.keys(), tracks_object.keys()))

        print("Tracks loaded!")
        return db_object, tracks_object


    def load_measurements_from_clickpoints(self, path, type,
                                           # db_object, tracks_object,
                                           measured_variables=["PositionX", "PositionY"]):
        db_object = DataFileExtended(path)
        db=db_object
        tracks_object = {}
        tracks = db.getTracks(type=type)
        all_markers = db.db.execute_sql('SELECT track_id, (SELECT sort_index FROM image WHERE image.id = image_id) as sort_index, measurement.x, measurement.y, z FROM marker JOIN measurement ON marker.id = measurement.marker_id WHERE type_id = ?', str(db.getMarkerType(name=type).id)).fetchall()
        all_markers = np.array(all_markers)

        for track in tracks:
            track = track.id
            n = len(self.track_dict)
            self.track_dict.update({n: track})
            tracks_object.update({track: Filters.Filter(Model(state_dim=2, meas_dim=2),
                                                 no_dist=True,
                                                 prob_update=False)})
            tracks_object[track].Model.Measured_Variables = measured_variables
            track_markers = all_markers[all_markers[:,0]==track]
            X = dict(zip(track_markers.T[1].astype(int), track_markers[:, 2:, None]))
            for i in sorted(X):
                try:
                    tracks_object[track].predict(i)
                except:
                    pass
                tracks_object[track].update(i=i, z=Measurement(1., X[i]))
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

    def load_System_tracks_from_clickpoints(self, path, type):
        self.system_db, self.System_Tracks =  self.load_tracks_from_clickpoints(path, type)


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
        setA = set(trackA.X.keys())
        setB = set(trackB.X.keys())
        print("calculated delta T!")
        return float(len(setA.intersection(setB)))/len(setA.union(setB))
        # intersect = set(trackA.X.keys()).intersection(trackB.X.keys())
        # return max(len(intersect)/len(trackA.X.keys()), len(intersect)/len(trackB.X.keys()))
        # return len(set(trackA.X.keys()).intersection(trackB.X.keys()))/len(set(trackA.X.keys()).union(set(trackB.X.keys())))
        # return min(max(trackA.X.keys()),max(trackB.X.keys()))-max(min(trackA.X.keys()),min(trackB.X.keys()))

    def spatial_overlap(self, trackA, trackB):
        id_A = trackA
        id_B = trackB
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        frames = set(self.System_DF[self.System_DF["Track"] == id_A].Frame).intersection(
            set(self.GT_DF[self.GT_DF["Track"] == id_B].Frame))
        # frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
        if len(frames) == 0:
            return {}
        A = self.System_DF[self.System_DF.Track==id_A& self.System_DF.Frame.isin(frames)]
        B = self.GT_DF[self.GT_DF.Track==id_B& self.GT_DF.Frame.isin(frames)]
        I = self._intersect_many_(A.X, A.Y, B.X, B.Y)
        X = np.array([[trackA.X[f][0], trackA.X[f][1], trackB.X[f][0], trackB.X[f][1]] for f in frames], ndmin=3)
        # I = self._intersect_many_(*X.T[0])
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
                return  self.System_Tracks[A]
            else:
                raise ValueError("%s is not a track!" % track)
        else:
            return track

    def pandas_spatial_overlap(self, st, gt):
        frames = set(self.DF[(self.DF["Track"] == st) & (self.DF.Type == "System")].Frame).intersection(
            set(self.DF[(self.DF["Track"] == gt) & (self.DF.Type == "GT")].Frame))
        positions = self.DF[
            (((self.DF.Track == st) & (self.DF.Type == "System")) | ((self.DF.Track == gt) & (self.DF.Type == "GT")))
            & self.DF.Frame.isin(frames)]
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
        super_list = [[i, j, *self.System_Tracks[i].X[j].T[0], "System"]
                                                     for i in self.System_Tracks.keys()
                                                     for j in self.System_Tracks[i].X.keys()]
        super_list.extend([[i, j, *self.GT_Tracks[i].X[j].T[0], "GT"]
                                                     for i in self.GT_Tracks.keys()
                                                     for j in self.GT_Tracks[i].X.keys()])
        self.DF = pandas.DataFrame(super_list,columns=["Track","Frame","X","Y", "Type"])

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
                if self.pandas_spatial_overlap(st, gt).mean()>self.SpaceThreshold and self.temporal_overlap(st,gt)>self.TempThreshold:
                    m.append(st)
            self.Matches.update({gt: m})
                # B = self.DF[self.DF.Track == gt & self.DF.Frame.isin(frames) & self.DF.Type=="GT"]
                # I = self._intersect_many_(A.X, A.Y, B.X, B.Y)


        # spatial_overlap =
        # for gt in self.GT_Tracks:
        #     self.Matches.update({gt: [st for st in self.System_Tracks if (np.mean(list(self.spatial_overlap(st,gt).values()))>self.SpaceThreshold and
        #                                         self.temporal_overlap(st,gt)>self.TempThreshold)]})

    def FAT(self, trackA, trackB):
        pass

    def TF(self, trackA):
        pass

    def TE(self, trackA, trackB):
        trackA = self._handle_Track_(trackA)
        trackB = self._handle_Track_(trackB)
        frames = set(trackA.X.keys()).intersection(set(trackB.X.keys()))
        return dict([[f,np.linalg.norm(trackA.X[f]-trackB.X[f])] for f in frames])

    def TME(self, trackA, trackB):
        return np.mean(self.TE(trackA, trackB))

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
            te = self.TE(trackA, trackB).values()
            tmemt.append(np.mean(te)*len(te))
            l.append(len(te))
        return np.sum(tmemt)/np.sum(l)

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
            te = self.TE(trackA, trackB).values()
            tmemt.append(np.std(te)*len(te))
            l.append(len(te))
        return np.sum(tmemt) / np.sum(l)

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
            o=self.spatial_overlap(trackA, trackB).values()
            ct.extend(o)
            l.append(len(o))
        return np.sum(ct)/np.sum(l)

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
            o=self.spatial_overlap(trackA, trackB).values()
            ct.append(len(o)*np.std(o))
            l.append(len(o))
        return np.sum(ct)/np.sum(l)

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
        trackA = self._handle_Track_(trackA)
        if trackA in self.GT_Tracks.values():
            key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        else:
            raise KeyError("Track is not a Ground Truth Track")
        ct = []
        l = []
        frames = set()
        for m in self.Matches[key]:
            trackB = self._handle_Track_(m)
            frames.symmetric_difference_update(trackB.X.keys())
        IDC_j = 1
        id = None
        for f in frames:
            if id is None:
                id=[k for k in self.Matches[key] if f in self.System_Tracks[k].X]
                if len(id)>0:
                    id=id[0]
                else:
                    id = None
                    continue
            if not f in self.System_Tracks[id].X:
                IDC_j +=1
                ids=[k for k in self.Matches[key] if f not in self.System_Tracks[k].X]
                func = lambda i : len([fr for fr in frames if f<fr and fr<[fr for fr in frames if fr>f and not f in self.System_Tracks[id].X][0]])
                try:
                    id = max(ids, key=func)
                except ValueError:
                    id = None
                    continue
        return IDC_j

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
        delta_t = np.array([delta.seconds for delta in t[1:]-t[:-1]], dtype=float)
        return delta_x.T/delta_t

    def activity(self, trackA):
        v=self.velocity(trackA)
        v_n = np.linalg.norm(v, axis=0)
        return np.sum(v_n>self.Object_Size/2.)/float(len(v_n))

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
        return np.sum(v_n>thresh)/float(len(v_n))

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
        return np.mean(self.rTE(trackA,trackB).values())

    def rTMED(self, trackA, trackB):
        return np.std(self.rTE(trackA,trackB).values())

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
            tmemt.append(np.mean(te.values()) * len(te))
            l.append(len(te))
        return np.sum(tmemt) / np.sum(l)

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
            tmemt.append(np.std(te.values()) * len(te))
            l.append(len(te))
        return np.sum(tmemt) / np.sum(l)


if __name__ == "__main__":
    evaluation = Yin_Evaluator(25.)
    evaluation.load_System_tracks_from_clickpoints(
        # path="/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData3.cdb",
        path=r"C:\Users\Alex\Documents/Promotion/AdeliesAntavia/AdelieTrack/AdelieData2.cdb",
        type="Track_Marker")
    # evaluation.load_GT_tracks_from_clickpoints(path="/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData_GroundTruth_interpolated.cdb",
    evaluation.load_GT_tracks_from_clickpoints(path=r"C:\Users\Alex\Documents/Promotion/AdeliesAntavia/AdelieTrack/AdelieData_GroundTruth.cdb",
                                               type="GroundTruth")

    evaluation.match()
    print(evaluation.Matches)

