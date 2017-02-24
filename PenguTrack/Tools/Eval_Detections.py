from __future__ import print_function, division
import clickpoints
import numpy as np
# import sys
pingusize=12

db = clickpoints.DataFile("/home/alex/Masterarbeit/master_project/master_project/adelie_data/770_PANA/Neuer Ordner/new0.cdb")
GT_Type = db.getMarkerType(name="GT_Detections")
GT_Markers = db.getMarkers(type=GT_Type)
Auto_Type = db.getMarkerType(name="PT_Track_Marker")
Auto_Markers = db.getMarkers(type=Auto_Type, frame=0)

GT_pos = np.asarray([[marker.x, marker.y] for marker in GT_Markers])
auto_pos = np.asarray([[marker.x, marker.y] for marker in Auto_Markers])

dist_mat = np.linalg.norm(GT_pos[None,:].T-auto_pos[:,None].T, axis=0)

dists1 = np.amin(dist_mat, axis=1)
dists0 = np.amin(dist_mat, axis=0)
n_false_positive = len(dists0[dists0>pingusize])
n_correct = len(dists1[dists1<=pingusize])
n_not_found = dist_mat.shape[0]-n_correct
total_rms_err = np.sqrt(np.mean(np.square(dists1[dists1 < pingusize])))

print("N-Total: %s GT, %s Auto" % dist_mat.shape)
print("Correct Detections: %s absolute, %s relative %%"%(n_correct, 100*n_correct/dist_mat.shape[0]))
print("False Positives: %s absolute, %s relative %%"%(n_false_positive, 100*n_false_positive/dist_mat.shape[1]))
print("False Negative: %s absolute, %s relative %%"%(n_not_found, 100*n_not_found/dist_mat.shape[0]))
print("Total RMS-Error: %s absolute, %s relative %%"%(total_rms_err, 100*total_rms_err/pingusize))