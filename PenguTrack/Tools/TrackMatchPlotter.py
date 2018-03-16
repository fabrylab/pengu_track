import seaborn as sns
import numpy as np
from PenguTrack.DataFileExtended import DataFileExtended,\
    add_PT_Tracks, add_PT_Tracks_from_Tracker, load_tracks_from_clickpoints, load_measurements_from_clickpoints



class TrackMatchPlotter(object):
    def __init__(self, matches):
        self.System_Tracks = {}
        self.GT_Tracks = {}
        self.gt_db = None
        self.system_db = None
        self.gt_track_dict = {}
        self.system_track_dict = {}
        self.Matches = matches

    def add_PT_Tracks(self, tracks):
        return add_PT_Tracks(tracks)

    def add_PT_Tracks_from_Tracker(self, tracks):
        tracks_dict, tracks_object = add_PT_Tracks_from_Tracker(tracks)
        return tracks_object

    def load_tracks_from_clickpoints(self, path, type):
        db_object, tracks_object = load_tracks_from_clickpoints(path, type, tracker_name=None)
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

    def load_System_tracks_from_clickpoints(self, path, type):
        self.system_db, self.System_Tracks =  self.load_tracks_from_clickpoints(path, type)
        self.system_track_dict = self.system_db.track_dict

    def plot_img(self, fig, ax, *args, **kwargs):
        img = self.gt_db.getImage(0).data
        ax.imshow(img, *args, **kwargs)

    def plot_gt(self, gt, fig, ax, *args, **kwargs):
        X, Y = np.array([self.GT_Tracks[gt].X[f] for f in sorted(self.GT_Tracks[gt].X)]).T[0]
        ax.plot(Y, X, "-ko", *args, label="Ground Truth (%s)"%gt, **kwargs)

    def plot_tracklets(self, gt, fig, ax, *args, **kwargs):
        N = len(self.Matches[gt])
        cpal = sns.color_palette("hls", N)
        for i,st in enumerate(self.Matches[gt]):
            X1, Y1 = np.array([self.System_Tracks[st].X[f] for f in sorted(self.System_Tracks[st].X)]).T[0]
            ax.plot(Y1, X1, *args, color=cpal[i], label=str(st), **kwargs)

        #     ...
        # plt.savefig(bbox_inches = 'tight', pad_inches = 0, dpi = 300)
        # ax.legend()
        # plt.show()

