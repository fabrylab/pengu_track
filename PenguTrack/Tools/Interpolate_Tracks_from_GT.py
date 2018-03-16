import clickpoints
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

db = clickpoints.DataFile("/home/alex/Promotion/AdeliesAntavia/AdelieTrack/AdelieData_GroundTruth_interpolated.cdb")

type0 = db.getMarkerType("GroundTruth")
type1 = db.setMarkerType(name="interpolated", color="#FF0000", mode=db.TYPE_Track)
db.deleteMarkers(type=type1)

frames = np.array([i.sort_index for i in db.getImages()])
images = np.array([db.getImage(frame=f)for f in frames])
args = np.argsort(frames)
frames = frames[args]
images = images[args]

for track0 in db.getTracks(type=type0):
    track1 = db.setTrack(type=type1)
    t, x0, y0 = np.array([[m.image.sort_index, m.x, m.y] for m in track0.track_markers]).T
    print("Track points loaded")
    args = np.argsort(t)
    t=t[args]
    x0=x0[args]
    y0=y0[args]
    X1=interp1d(t, np.array([x0, y0]), kind="linear")(frames[(t[0]<=frames)&(frames <= t[-1])])
    print("Interpolation Done")
    db.setMarkers(image=images[(t[0]<=frames)&(frames <= t[-1])], x=X1[0], y=X1[1], track=track1)
    print("done track %s"%track0.id)
    # plt.plot(X1.T[0], X1.T[1])
    # plt.show()
    # print(f, X1)
