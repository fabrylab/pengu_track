from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import clickpoints

from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import VariableSpeed
from PenguTrack.DataFileExtended import DataFileExtended

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

    def load_GT_tracks_from_clickpoints(self, path, type, use_measurements=False):
        self.GT_db = DataFileExtended(path)
        if self.GT_db.getTracks(type=type)[0].markers[0].measurement is not None and \
            len(self.GT_db.getTracks(type=type)[0].markers[0].measurement)>0 and \
                use_measurements:
            for track in self.GT_db.getTracks(type=type):
                self.GT_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z])
                    # self.GT_Tracks[track.id].update(z=meas, i=m.image.timestamp)
                    if np.any([im.timestamp is None for im in self.GT_db.getImages()]):
                        self.GT_Tracks[track.id].update(z=meas, i=m.image.filename)
                    else:
                        self.GT_Tracks[track.id].update(z=meas, i=m.image.timestamp)

        else:
            for track in self.GT_db.getTracks(type=type):
                self.GT_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.x,
                                            m.y,
                                            0])

                    if np.any([im.timestamp is None for im in self.GT_db.getImages()]):
                        self.GT_Tracks[track.id].update(z=meas, i=m.image.sort_index)
                    else:
                        self.GT_Tracks[track.id].update(z=meas, i=m.image.timestamp)


    def load_System_tracks_from_clickpoints(self, path, type, use_measurements=False):
        self.System_db = DataFileExtended(path)
        if self.System_db.getTracks(type=type)[0].markers[0].measurement is not None and \
            len(self.System_db.getTracks(type=type)[0].markers[0].measurement)>0 and \
            use_measurements:
            for track in self.System_db.getTracks(type=type):
                self.System_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z],
                                       frame=m.image.sort_index)
                    # self.System_Tracks[track.id].update(z=meas, i=m.image.timestamp)
                    if np.any([im.timestamp is None for im in self.System_db.getImages()]):
                        self.System_Tracks[track.id].update(z=meas, i=m.image.sort_index)
                    else:
                        self.System_Tracks[track.id].update(z=meas, i=m.image.timestamp)

        else:
            for track in self.System_db.getTracks(type=type):
                self.System_Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.x,
                                            m.y,
                                            0], frame=m.image.sort_index)
                    # self.System_Tracks[track.id].update(z=meas, i=m.image.timestamp)
                    if np.any([im.timestamp is None for im in self.System_db.getImages()]):
                        self.System_Tracks[track.id].update(z=meas, i=m.image.sort_index)
                    else:
                        self.System_Tracks[track.id].update(z=meas, i=m.image.timestamp)


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
            for track in self.GT_Tracks:
                # self.db.deleteTracks(id=track)
                GT_db_track = self.GT_db.setTrack(type=type, id=track)
                for m in self.GT_Tracks[track].Measurements:
                    meas = self.GT_Tracks[track].Measurements[m]
                    pos = [meas.PositionX, meas.PositionY, meas.PositionZ]
                    pos = function(pos)
                    try:
                        marker = self.GT_db.setMarker(type=type, frame=self.GT_db.getImages(timestamp=m)[0].sort_index, x=pos[0], y=pos[1], track=track)
                    except clickpoints.ImageDoesNotExist:
                        continue
                    except IndexError:
                        continue
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
                    marker = self.System_db.setMarker(type=type, frame=self.System_db.getImages(timestamp=m)[0].sort_index, x=pos[0], y=pos[1], track=track)
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
        intersect = set(trackA.X.keys()).intersection(trackB.X.keys())
        # return max(len(intersect)/len(trackA.X.keys()), len(intersect)/len(trackB.X.keys()))
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

    def _handle_Track_(self, track, tracks=None):
        if not isinstance(track, Filter):
            try:
                A = int(track)
            except TypeError:
                raise ValueError("%s is not a track!"%track)
            if self.GT_Tracks.has_key(A) and (tracks is None or tracks==self.GT_Tracks):
                 return self.GT_Tracks[A]
            elif self.System_Tracks.has_key(A) and (tracks is None or tracks==self.System_Tracks):
                return  self.System_Tracks[A]
            else:
                raise ValueError("%s is not a track!" % track)
        else:
            return track

    def match(self):
        for gt in self.GT_Tracks:
            self.Matches.update({gt: [st for st in self.System_Tracks if (np.mean(self.spatial_overlap(st,gt).values())>self.SpaceThreshold and
                                                self.temporal_overlap(st,gt)>self.TempThreshold)]})

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
                id=[k for k in self.Matches[key] if self.System_Tracks[k].X.has_key(f)]
                if len(id)>0:
                    id=id[0]
                else:
                    id = None
                    continue
                # id=[k for k in self.Matches[key] if self.System_Tracks[k].X.has_key(f)][0]
            if not self.System_Tracks[id].X.has_key(f):
                IDC_j +=1
                ids=[k for k in self.Matches[key] if self.System_Tracks[k].X.has_key(f)]
                func = lambda i : len([fr for fr in frames if f<fr and fr<[fr for fr in frames if fr>f and not self.System_Tracks[id].X.has_key(f)][0]])
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
        # trackA = self._handle_Track_(trackA)
        # if trackA in self.GT_Tracks.values():
        #     key = [k for k in self.GT_Tracks if self.GT_Tracks[k]==trackA][0]
        # else:
        #     raise KeyError("Track is not a Ground Truth Track")
        # from datetime import timedelta
        # x = np.array(trackA.X.values(), dtype=float)
        # delta_x = x[1:]-x[:-1]
        # t = np.array(sorted(trackA.X.keys()))
        # delta_t = np.array([delta.seconds for delta in t[1:]-t[:-1]], dtype=float)
        # v = delta_x.T/delta_t
        v=self.velocity(trackA)
        v_n = np.linalg.norm(v, axis=0)
        # from skimage.filters import threshold_otsu
        # val = threshold_otsu(v_n, nbins=100)
        # print(v_n, v_n.shape)
        # plt.figure()
        # plt.hist(v_n.T, bins=100)
        # print(np.amin(v_n), np.amax(v_n), val)
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

    version = "adelie"
    if version == "adelie":

        text_file = "/home/alex/Desktop/TrackingEvaluation_adelie.txt"
        with open("/home/alex/Desktop/TrackingEvaluation_adelie.txt", "w") as myfile:
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

        evaluation = Alex_Evaluator(0.525, 37.9, 0.24, 2592, 9e-3, 14e-3
                                    ,temporal_threshold=0.01, spacial_threshold=0.4, tolerance=1)
        evaluation.load_GT_tracks_from_clickpoints(path="/home/birdflight/Desktop/Adelie_Evaluation/252Horizon.cdb", type="GT_under_limit")
        evaluation.load_System_tracks_from_clickpoints(path="/home/birdflight/Desktop/Adelie_Evaluation/PT_Test_full_n3_r7_A20.cdb", type="PT_Track_Marker")
        evaluation.load_System_tracks_from_clickpoints(path="/home/birdflight/Desktop/Adelie_Evaluation/PT_Test_full_n3_r7_A20.cdb", type="PT_Track_Marker")
        evaluation.system_db = clickpoints.DataFile("/home/birdflight/Desktop/Adelie_Evaluation/PT_Test_full_n3_r7_A20.cdb")
        evaluation.system_db.setMarkerType(name="GT_under_limit", color="#FFFFFF", mode=evaluation.system_db.TYPE_Track)
        evaluation.system_db.setMarkerType(name="Match", color="#FF8800", mode=evaluation.system_db.TYPE_Track)
        evaluation.save_GT_tracks_to_db(path="/home/birdflight/Desktop/Adelie_Evaluation/PT_Test_full_n3_r7_A20.cdb", type="GT_under_limit")

        # for f in sorted(evaluation.GT_Tracks.values()[0].X.keys())[:20]:
        for f in sorted(set(np.hstack([track.X.keys() for track in evaluation.GT_Tracks.values()])))[:20]:
        # for f in sorted(evaluation.GT_Tracks[1].X.keys())[:20]:
            for m in evaluation.GT_Tracks:
                evaluation.GT_Tracks[m].downfilter(f)
        evaluation.match()

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

        print(evaluation.Matches)
        for m in evaluation.Matches:
            print("------------")
            print(m)
            tmemt.append(evaluation.TMEMT(m))
            tmemtd.append(evaluation.TMEMTD(m))
            tc.append(evaluation.TC(m))
            r2.append(evaluation.R2(m))
            r3.append(evaluation.R3(m))
            ctm.append(evaluation.CTM(m))
            ctd.append(evaluation.CTD(m))
            lt.append(evaluation.LT(m))
            idc.append(evaluation.IDC(m))
            a.append(evaluation.activity(m))
            print("TMEMT", tmemt[-1], tmemtd[-1])
            print("TMEMT coeff", tmemtd[-1] / tmemt[-1])
            print("TC", tc[-1])
            print("corrected TC", tc[-1] / a[-1])
            print("R2", r2[-1])
            print("R3", r3[-1])
            print("CTM", ctm[-1])
            print("CTD", ctd[-1])
            print("LT", lt[-1])
            print("IDC", idc[-1])
            print("activity", a[-1])

        print("------------")
        print("----mean----")
        print("TMEMT", np.mean(tmemt), np.std(tmemtd) / len(tmemtd) ** 0.5)
        print("TMEMT coeff", np.mean(np.asarray(tmemtd) / np.asarray(tmemt)),
              np.std(np.asarray(tmemtd) / np.asarray(tmemt)) / len(tmemt) ** 0.5)
        print("TC", np.mean(tc), np.std(tc) / len(tc) ** 0.5)
        print("corrected TC", np.mean(np.asarray(tc) / np.asarray(a)),
              np.std(np.asarray(tc) / np.asarray(a)) / len(a) ** 0.5)
        print("R2", np.mean(r2), np.std(r2) / len(r2) ** 0.5)
        print("R3", np.mean(r3), np.std(r3) / len(r3) ** 0.5)
        print("CTM", np.mean(ctm), np.std(ctm) / len(ctm) ** 0.5)
        print("CTD", np.mean(ctd), np.std(ctd) / len(ctd) ** 0.5)
        print("LT", np.mean([l.seconds for l in lt]), np.std([l.seconds for l in lt]) / len(lt) ** 0.5)
        print("IDC", np.mean(idc), np.std(idc) / len(idc) ** 0.5)
        print("activity", np.mean(a), np.std(a) / len(a) ** 0.5)

        from datetime import timedelta

        def total_hits(k):
            return evaluation.TC(k) * len(evaluation.GT_Tracks[k].X.keys()) if (len(evaluation.GT_Tracks[k].X.keys()) > 10
                                                                                and len(evaluation.Matches[k]) < 11) else 0.
            # return np.random.rand()

        plotted = dict([[k, evaluation.Matches[k]] for k in sorted(evaluation.Matches.keys(), key=total_hits)[-5:]])

        from datetime import timedelta

        fig, axes = plt.subplots(len(plotted), 1)

        all_times = set()
        # for m in evaluation.GT_Tracks:
        for m in plotted:
            all_times.update(evaluation.GT_Tracks[m].X.keys())
        all_times = sorted(all_times)
        x_min = min(all_times)
        x_max = max(all_times)
        max_len = max([len(v) for v in evaluation.Matches.values()])
        pal = sn.color_palette(n_colors=max_len)
        fig.suptitle("Track Coverage", y=1.)
        lines = {}
        for i, m in enumerate(plotted):
            axes[i].set_xlim([x_min, x_max])
            axes[i].set_ylim([0, 2])
            # axes[i].set_title("Ground Truth Track %s"%m, size=12)
            ACT = np.linalg.norm(evaluation.relative_velocity(m), axis=0)
            temps = sorted(evaluation._handle_Track_(m).X.keys())
            l1 = axes[i].fill_between([t for t in all_times if t in temps], [2 for t in all_times if t in temps], alpha=0.1,
                                      label="Penguin is Visible")
            l2 = axes[i].fill_between(temps[1:], ACT, alpha=0.2, label="Penguin Velocity")
            lines.update({"Penguin is Visible  ": l1, "Penguin Velocity  ": l2})
            for j, n in enumerate(evaluation.Matches[m]):
                overs = evaluation.rTE(m, n)
                temps = np.array(sorted(overs.keys()))
                x = [overs[t] for t in temps]
                t_0 = temps[0]
                lines.update({"System Track Error %s" % j: axes[i].plot(temps, x, color=pal[j])[0]})
                axes[i].set_ylabel("Distance\n (in Penguin Sizes)")
                my_plot.despine(axes[i])
                my_plot.setAxisSizeMM(fig, axes[i], 147, 180 / len(axes))
        axes[-1].set_xlabel("Timestamp")
        plt.tight_layout()
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
        fig.legend([lines[k] for k in sorted(lines.keys())], [k[:-2] for k in sorted(lines.keys())], ncol=3, loc="lower center",
                   prop={"size": 12})
        plt.savefig("/home/birdflight/Desktop/Adelie_Evaluation/Pictures/Adelie_TrackEvaluation.pdf")
        plt.savefig("/home/birdflight/Desktop/Adelie_Evaluation/Pictures/Adelie_TrackEvaluation.png")
        plt.show()

    elif version == "cell":
        text_file = "/home/alex/Desktop/TrackingEvaluation3d.txt"
        with open("/home/alex/Desktop/TrackingEvaluation3d.txt", "w") as myfile:
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

        # evaluation = Alex_Evaluator(0.525, 37.9, 0.24, 2592, 9e-3, 14e-3
        #                             , temporal_threshold=0.01, spacial_threshold=0.4, tolerance=1)
        evaluation = Yin_Evaluator(16, temporal_threshold=0.01, spacial_threshold=0.4)
        evaluation.load_GT_tracks_from_clickpoints(path="/home/alex/Desktop/PT_Cell_GT_Track.cdb",
                                                   type="GroundTruth")
        evaluation.load_System_tracks_from_clickpoints(
            path="/home/alex/Desktop/PT_Cell_T850_A75_inf_3d.cdb", type="PT_Track_Marker")
        evaluation.system_db = clickpoints.DataFile(
            "/home/alex/Desktop/PT_Cell_T850_A75_inf_3d.cdb")
        evaluation.system_db.setMarkerType(name="GT", color="#FFFFFF", mode=evaluation.system_db.TYPE_Track)
        evaluation.system_db.setMarkerType(name="Match", color="#FF8800", mode=evaluation.system_db.TYPE_Track)
        evaluation.save_GT_tracks_to_db(path="/home/alex/Desktop/PT_Cell_T850_A75_inf_3d.cdb",
                                        type="GT")

        # for f in sorted(evaluation.GT_Tracks[1].X.keys())[:20]:
        for f in sorted(set(np.hstack([t.X.keys() for t in evaluation.GT_Tracks.values()])))[:20]:
            for m in evaluation.GT_Tracks:
                evaluation.GT_Tracks[m].downfilter(f)
        evaluation.match()

        bad_ids = [m for m in evaluation.Matches if len(evaluation.Matches[m])<1]
        for b in bad_ids:
            evaluation.Matches.pop(b)

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

        print_save(evaluation.Matches)
        for m in evaluation.Matches:
            print_save("------------")
            print_save(m)
            tmemt.append(evaluation.TMEMT(m))
            tmemtd.append(evaluation.TMEMTD(m))
            tc.append(evaluation.TC(m))
            r2.append(evaluation.R2(m))
            r3.append(evaluation.R3(m))
            ctm.append(evaluation.CTM(m))
            ctd.append(evaluation.CTD(m))
            lt.append(evaluation.LT(m))
            idc.append(evaluation.IDC(m))
            a.append(evaluation.activity(m))
            print_save("TMEMT", tmemt[-1], tmemtd[-1])
            print_save("TMEMT coeff", tmemtd[-1] / tmemt[-1])
            print_save("TC", tc[-1])
            print_save("corrected TC", tc[-1] / a[-1])
            print_save("R2", r2[-1])
            print_save("R3", r3[-1])
            print_save("CTM", ctm[-1])
            print_save("CTD", ctd[-1])
            print_save("LT", lt[-1])
            print_save("IDC", idc[-1])
            print_save("activity", a[-1])

        print_save("------------")
        print_save("----mean----")
        print_save("TMEMT", np.mean(tmemt), np.std(tmemtd) / len(tmemtd) ** 0.5)
        print_save("TMEMT coeff", np.mean(np.asarray(tmemtd) / np.asarray(tmemt)),
              np.std(np.asarray(tmemtd) / np.asarray(tmemt)) / len(tmemt) ** 0.5)
        print_save("TC", np.mean(tc), np.std(tc) / len(tc) ** 0.5)
        print_save("corrected TC", np.mean(np.asarray(tc) / np.asarray(a)),
              np.std(np.asarray(tc) / np.asarray(a)) / len(a) ** 0.5)
        print_save("R2", np.mean(r2), np.std(r2) / len(r2) ** 0.5)
        print_save("R3", np.mean(r3), np.std(r3) / len(r3) ** 0.5)
        print_save("CTM", np.mean(ctm), np.std(ctm) / len(ctm) ** 0.5)
        print_save("CTD", np.mean(ctd), np.std(ctd) / len(ctd) ** 0.5)
        print_save("LT", np.mean([l.seconds for l in lt]), np.std([l.seconds for l in lt]) / len(lt) ** 0.5)
        print_save("IDC", np.mean(idc), np.std(idc) / len(idc) ** 0.5)
        print_save("activity", np.mean(a), np.std(a) / len(a) ** 0.5)

    # elif version == "beer":
if True:
    if True:
        eve = evaluation
        from datetime import timedelta
        import matplotlib.dates as md

        mean_vel = np.mean([np.mean(np.linalg.norm(evaluation.velocity(k))) for k in evaluation.Matches])


        def total_hits(k):
            # return evaluation.TC(k) * len(evaluation.GT_Tracks[k].X.keys()) if (len(evaluation.GT_Tracks[k].X.keys()) > 10
            #                                                                     and len(evaluation.Matches[k]) < 11) else 0.
            return 1. if k in [374,392,288, 346, 383] else 0.
            # return (np.mean(np.linalg.norm(evaluation.velocity(k), axis=0))-mean_vel)**2 if evaluation.TC(k)>0.7 and len(evaluation.GT_Tracks[k].X.keys())>20 else 0.
            # return np.mean(np.linalg.norm(evaluation.velocity(k), axis=0)) if (len(evaluation.Matches[k]) < 4 and evaluation.TC(k) >0.5) else 0.



        plotted = dict([[k, eve.Matches[k]] for k in sorted(eve.Matches.keys(), key=total_hits)[-5:]])

        fig, axes = plt.subplots(len(plotted.keys()), 1)

        all_times = set()
        for m in plotted:
            all_times.update(eve.GT_Tracks[m].X.keys())
        all_times = sorted(all_times)
        x_min = min(all_times)
        x_max = max(all_times)
        max_len = max([len(v) for v in plotted.values()])

        max_time_range = max([len(eve.GT_Tracks[k].X.keys()) for k in plotted]) + 1

        pal = sn.color_palette(n_colors=max_len)
        title = fig.suptitle("Track Coverage", y=1.)
        lines = {}
        for i, m in enumerate([346,383,392,374,288]):
            temps = sorted(eve._handle_Track_(m).X.keys())
            x_min_loc = min(temps)
            x_max_loc = max(temps)
            # if x_max_loc < x_max:
            # axes[i].set_xlim([x_min_loc, all_times[all_times.index(x_min_loc)] + timedelta(seconds=max_time_range)])
            # else:
            #     axes[i].set_xlim([x_min_loc, x_max])
            axes[i].set_xlim([x_min, x_max])

            axes[i].set_ylim([-0.1, 1.])
            # axes[i].set_title("Ground Truth Track %s"%m, size=12)
            ACT = np.linalg.norm(eve.velocity(m), axis=0)
            ACT[np.isinf(ACT)] = np.nan
            ACT /= eve.Object_Size#np.nanmax(ACT)
            ACT *= 60.
            l1 = axes[i].fill_between([t for t in all_times if t in temps], [100 for t in all_times if t in temps],
                                      alpha=0.1,
                                      label="Cell is Visible")
            l2 = axes[i].fill_between(temps[1:], ACT, -0.1, alpha=0.2, label="Cell Velocity")
            lines.update({"Cell is Visible  ": l1, "Cell Activity  ": l2})
            for j, n in enumerate(eve.Matches[m]):
                overs = eve.TE(m, n)
                temps = np.array(sorted(overs.keys()))
                x = np.array([overs[t] for t in temps])/eve.Object_Size
                t_0 = temps[0]
                lines.update({"System Track Error %s" % j: axes[i].plot(temps, x, color=pal[j])[0]})
            axes[i].set_ylabel("Distance\n in Cell Sizes")
            xfmt = md.DateFormatter('')
            axes[i].xaxis.set_major_formatter(xfmt)
            my_plot.despine(axes[i])
            my_plot.setAxisSizeMM(fig, axes[i], width=147, height=115 / len(axes))
            # box = axes[i].get_position()
            # axes[i].set_position([box.x0, box.y0 + box.height * 0.1,
            #                  box.width, box.height * 0.9])
        xfmt = md.DateFormatter('%M:%S')
        axes[-1].xaxis.set_major_formatter(xfmt)
        my_plot.despine(axes[i])
        axes[-1].set_xlabel("Timestamp")
        # axes[-1].
        # axes[i].
        # fig.autofmt_xdate()
        lgd = fig.legend([lines[k] for k in sorted(lines.keys())], [k[:-2] for k in sorted(lines.keys())], ncol=2,
                   loc="lower center",
                   # loc='upper center',
                   bbox_to_anchor=(0.5, -0.025),
                   prop={"size": 12})

        plt.tight_layout(pad=1.2)
        plt.savefig("/home/alex/Masterarbeit/Pictures/Cell_TrackEvaluation3d.pdf", bbox_extra_artists=(lgd, title,), bbox_inches='tight')
        plt.savefig("/home/alex/Masterarbeit/Pictures/Cell_TrackEvaluation3d.png", bbox_extra_artists=(lgd, title,), bbox_inches='tight')
        plt.show()

    elif version == "bird":
        import clickpoints
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sn
        import my_plot

        eve = Yin_Evaluator(10, spacial_threshold=0.01, temporal_threshold=0.1)
        eve.load_GT_tracks_from_clickpoints("/mnt/mmap/GT_Starter.cdb", "GT_Bird")
        # eve.load_System_tracks_from_clickpoints("/mnt/mmap/Starter_n3_3_r20.cdb", type="PT_Track_Marker")
        eve.load_System_tracks_from_clickpoints("/mnt/mmap/Starter_n3_3_r20_F5_q200_r100.cdb", type="PT_Track_Marker")
        eve.match()
        print(eve.Matches)

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

        print(eve.Matches)
        TC = {}
        for m in eve.Matches:
            print("------------")
            print(m)
            tmemt.append(eve.TMEMT(m))
            tmemtd.append(eve.TMEMTD(m))
            tc.append(eve.TC(m))
            r2.append(eve.R2(m))
            r3.append(eve.R3(m))
            ctm.append(eve.CTM(m))
            ctd.append(eve.CTD(m))
            lt.append(eve.LT(m))
            idc.append(eve.IDC(m))
            a.append(eve.activity(m))
            print("TMEMT", tmemt[-1], tmemtd[-1])
            print("TMEMT coeff", tmemtd[-1] / tmemt[-1])
            print("TC", tc[-1])
            TC.update({m: tc[-1]})
            print("corrected TC", tc[-1] / a[-1])
            print("R2", r2[-1])
            print("R3", r3[-1])
            print("CTM", ctm[-1])
            print("CTD", ctd[-1])
            print("LT", lt[-1])
            print("IDC", idc[-1])
            print("activity", a[-1])

        print("------------")
        print("----mean----")
        print("TMEMT", np.mean(tmemt), np.std(tmemtd) / len(tmemtd) ** 0.5)
        print("TMEMT coeff", np.mean(np.asarray(tmemtd) / np.asarray(tmemt)),
              np.std(np.asarray(tmemtd) / np.asarray(tmemt)) / len(tmemt) ** 0.5)
        print("TC", np.mean(tc), np.std(tc) / len(tc) ** 0.5)
        print("corrected TC", np.mean(np.asarray(tc) / np.asarray(a)),
              np.std(np.asarray(tc) / np.asarray(a)) / len(a) ** 0.5)
        print("R2", np.mean(r2), np.std(r2) / len(r2) ** 0.5)
        print("R3", np.mean(r3), np.std(r3) / len(r3) ** 0.5)
        print("CTM", np.mean(ctm), np.std(ctm) / len(ctm) ** 0.5)
        print("CTD", np.mean(ctd), np.std(ctd) / len(ctd) ** 0.5)
        print("LT", np.mean([l.seconds for l in lt]), np.std([l.seconds for l in lt]) / len(lt) ** 0.5)
        print("IDC", np.mean(idc), np.std(idc) / len(idc) ** 0.5)
        print("activity", np.mean(a), np.std(a) / len(a) ** 0.5)

        from datetime import timedelta
        import matplotlib.dates as md


        def total_hits(k):
            return TC[k] if len(eve.GT_Tracks[k].X.keys()) > 10 else 0.


        plotted = dict([[k, eve.Matches[k]] for k in sorted(TC.keys(), key=total_hits)[-5:]])

        fig, axes = plt.subplots(len(plotted.keys()), 1)

        all_times = set()
        for m in plotted:
            all_times.update(eve.GT_Tracks[m].X.keys())
        all_times = sorted(all_times)
        x_min = min(all_times)
        x_max = max(all_times)
        max_len = max([len(v) for v in plotted.values()])

        max_time_range = max([len(eve.GT_Tracks[k].X.keys()) for k in plotted]) + 1

        pal = sn.color_palette(n_colors=max_len)
        fig.suptitle("Track Coverage", y=1.)
        lines = {}
        for i, m in enumerate(plotted):
            temps = sorted(eve._handle_Track_(m).X.keys())
            x_min_loc = min(temps)
            x_max_loc = max(temps)
            # if x_max_loc < x_max:
            axes[i].set_xlim([x_min_loc, all_times[all_times.index(x_min_loc)] + timedelta(seconds=max_time_range)])
            # else:
            #     axes[i].set_xlim([x_min_loc, x_max])

            axes[i].set_ylim([-0.1, 2])
            # axes[i].set_title("Ground Truth Track %s"%m, size=12)
            ACT = np.linalg.norm(eve.velocity(m), axis=0)
            ACT[np.isinf(ACT)] = np.nan
            ACT /= np.nanmax(ACT)
            l1 = axes[i].fill_between([t for t in all_times if t in temps], [100 for t in all_times if t in temps],
                                      alpha=0.1,
                                      label="Bird is Visible")
            l2 = axes[i].fill_between(temps[1:], ACT, -0.1, alpha=0.2, label="Bird Velocity")
            lines.update({"Bird is Visible  ": l1, "Bird Activity  ": l2})
            for j, n in enumerate(eve.Matches[m]):
                overs = eve.TE(m, n)
                temps = np.array(sorted(overs.keys()))
                x = [overs[t] for t in temps]
                t_0 = temps[0]
                lines.update({"System Track Error %s" % j: axes[i].plot(temps, x, color=pal[j])[0]})
            axes[i].set_ylabel("Distance\n in px")
            xfmt = md.DateFormatter('%M:%S')
            axes[i].xaxis.set_major_formatter(xfmt)
            my_plot.despine(axes[i])
            my_plot.setAxisSizeMM(fig, axes[i], width=147, height=115 / len(axes))
        axes[-1].set_xlabel("Timestamp")
        # axes[-1].
        # axes[i].
        # fig.autofmt_xdate()
        plt.tight_layout(pad=1.2)
        fig.legend([lines[k] for k in sorted(lines.keys())], [k[:-2] for k in sorted(lines.keys())], ncol=1,
                   loc="center right",
                   prop={"size": 12})
        plt.savefig("/home/birdflight/Desktop/Birdflight_TrackEvaluation.pdf")
        plt.savefig("/home/birdflight/Desktop/Birdflight_TrackEvaluation.png")
        plt.show()
