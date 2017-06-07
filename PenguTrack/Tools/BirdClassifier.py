import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from PenguTrack.DataFileExtended import DataFileExtended

file_path = "/home/user/Desktop/Birdflight.cdb"
db = DataFileExtended(file_path)

PT_Track_Type = db.getMarkerType(name="PT_Track_Marker")
PT_Detection_Type = db.getMarkerType(name="PT_Detection_Marker")
PT_Prediction_Type = db.getMarkerType(name="PT_Prediction_Marker")
GT_Type = db.getMarkerType(name="GT")

Tracks = db.getTracks(type=PT_Track_Type)

mean_phi = []
phies = []
std_phi = []
mean_len = []
lens = []

for track in Tracks:
    if track.markers.count() < 3:
        continue
    x = np.asarray([[m.x, m.y] for m in track.markers])
    del_x = x[1:]-x[:-1]
    del_x_norm = np.linalg.norm(del_x, axis=1)
    try:
        # phi = np.tensordot(del_x[1:],del_x[:-1].T, axes=1)/del_x_norm[1:]/del_x_norm[:-1]
        phi = np.diag(np.tensordot(del_x[1:],del_x[:-1].T, axes=1))/del_x_norm[1:]/del_x_norm[:-1]
        phi = phi[True^(np.isnan(phi) or np.isinf(phi))]
        print(phi.mean(), phi.std(), phi.max(), phi.min())
        phies.extend(phi)
        mean_phi.append(phi.mean())
        std_phi.append(phi.std())
        del_x_norm = del_x_norm[True^(np.isnan(phi) or np.isinf(phi))]
        lens.extend(del_x_norm)
        mean_len.append(del_x_norm.mean())
    except ValueError:
        continue

# sn.jointplot(mean_phi, mean_len)
plt.hist2d(mean_phi, mean_len)
plt.figure()
plt.hist2d(std_phi, mean_len)