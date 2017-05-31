from __future__ import print_function, division
import clickpoints
import numpy as np
from Detectors import Measurement
from Filters import Filter
from Models import VariableSpeed
from DataFileExtended import DataFileExtended



class Stitcher(object):
    def __init__(self):
        self.Tracks = {}
        self.db = None

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
        self.db = DataFileExtended(path)
        if self.db.getTracks(type=type)[0].markers[0].measurement is not None:
            for track in self.db.getTracks(type=type):
                self.Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z])
                    self.Tracks[track.id].update(z=meas, i=m.image.sort_index)
        else:
            for track in self.db.getTracks(type=type):
                self.Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = Measurement(1., [m.x,
                                            m.y,
                                            0])
                    self.Tracks[track.id].update(z=meas, i=m.image.sort_index)

    def stitch(self):
        pass

    def save_tracks_to_db(self, path, type, function=None):
        if function is None:
            function = lambda x : x
        self.db = DataFileExtended(path)
        self.db.deleteTracks(type=type)
        with self.db.db.atomic() as transaction:
            for track in self.Tracks:
                # self.db.deleteTracks(id=track)
                db_track = self.db.setTrack(type=type, id=track)
                for m in self.Tracks[track].Measurements:
                    meas = self.Tracks[track].Measurements[m]
                    pos = [meas.PositionX, meas.PositionY, meas.PositionZ]
                    pos = function(pos)
                    marker = self.db.setMarker(frame=m, x=pos[0], y=pos[1], track=track)
                    meas = self.Tracks[track].Measurements[m]
                    self.db.setMeasurement(marker=marker, log=meas.Log_Probability,
                                           x=meas.PositionX, y=meas.PositionY, z=meas.PositionZ)


class Heublein_Stitcher(Stitcher):
    """
    Class for Stitching Tracks
    """

    def __init__(self, a1, a2, a3, a4, max_cost, limiter):
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
        # end_frame = self.end_frame
        end_frame = max([max(track.Measurements) for track in self.Tracks.values()])
        default = np.nan
        if track1["end"] == end_frame:
            x = default
        elif track1["id"] == track2["id"] or track1["end"] == track1["start"] or track2["end"] <= track1["end"]:
            x = default
        else:
            distxy = np.sqrt(np.sum((track2["start_pos"] - track1["end_pos"]) ** 2.))
            distz = np.abs(track2["z_start"] - track1["z_end"])
            dt = track2["start"] - track1["end"]
            # print(distxy, distz, dt)
            if dt < 0:
                x = 3 * (a1 * distxy + a2 * distz) / (-dt + 1) + a4 * (-dt)
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
        return matches

    def Delete_Tracks_with_length_1(self):
        """
        Delets tracks with a length of 1.
        """
        shorts = [track for track in self.Tracks if len(self.Tracks[track].Measurements)<=1]
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
        print ("-----------Done with stitching-----------")



if __name__ == '__main__':
    import shutil
    shutil.copy("/home/user/CellTracking/layers_testing4.cdb",
                "/home/user/CellTracking/layers_testing4_test.cdb")

    def resulution_correction(pos):
        x, y, z = pos
        res = 6.45/10
        return y/res, x/res, z/10.

    # stitcher1 = Stitcher(1,0.5,50,60,200,122,"PT_Track_Marker",
    #                      "/home/user/CellTracking/layers_testing4_test.cdb",
    #                      5)
    # stitcher1 = Stitcher(1,0.5,50,60,200,122)
    # stitcher1.load_ClickpointsDB("/home/user/CellTracking/layers_testing4_test.cdb", track_type="PT_Track_Marker")
    # stitcher1.load_txt("/home/user/CellTracking/tracks.csv")
    # stitcher1.stitch()
    # stitcher1.write_ClickpointsDB()<<
    stitcher = Heublein_Stitcher(1/0.645,0.5,50,60,200,5)
    stitcher.load_tracks_from_clickpoints("/home/user/CellTracking/layers_testing4_test.cdb", "PT_Track_Marker")
    # stitcher.stitch()
    # stitcher.save_tracks_to_db("/home/user/CellTracking/layers_testing4_test.cdb", "PT_Track_Marker", function=resulution_correction)
    # print ("-----------Written to DB-----------")

