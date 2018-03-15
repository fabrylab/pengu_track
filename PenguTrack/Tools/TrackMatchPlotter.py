import seaborn as sns
import numpy as np


class TrackPlotter:
    def __init__(self, GT_Tracks, System_Tracks, matches):
        self.GT_Tracks = GT_Tracks
        self.System_Tracks = System_Tracks
        self.Matches = matches
    def plot_gt(self, gt, fig, ax, *args, **kwargs):
        X, Y = np.array([self.GT_Tracks[gt].X[f] for f in sorted(self.GT_Tracks[gt].X)]).T[0]
        ax.plot(X, Y, "-ko", label="Ground Truth (%s)"%gt)
        N = len(self.Matches[gt])
        cpal = sns.color_palette("hls", N)

        for i,st in enumerate(self.Matches[gt]):
            X1, Y1 = np.array([self.System_Tracks[st].X[f] for f in sorted(self.System_Tracks[st].X)]).T[0]
            ax.plot(X1, Y1, color=cpal[i], label=str(st))
        # ax.legend()
        # plt.show()

