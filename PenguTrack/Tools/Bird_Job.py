from __future__ import division, print_function

# import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import binary_dilation
import peewee
import sys

# import clickpoints
import platform
from os import path
from time import time

# Connect to database
# for p in sys.argv:
# 	print(p)
# file_path = str(sys.argv[1])
# q = float(sys.argv[2])
# r = float(sys.argv[3])
q = 200
r = 300
# if platform.system() != 'Linux':
#     file_path = file_path.replace("/mnt/jobs", r"//131.188.117.98/shared/jobs")
#path.normpath(file_path)

# import os
# # path = "/mnt/jobs/Pengu_Track_Evaluation/20150204/247.cdb"
# path = str(file_path)
# os.system("mkdir ~/Desktop/TODO")
# os.system("cp %s ~/Desktop/TODO/%s_done.cdb"%(path, path[-7:-4]))
# # os.system("cp -r %s/* ~/Desktop/TODO/"%path[:-7])
#
# file_path = "/home/alex/Desktop/TODO/%s_done.cdb"%path[-7:-4]



start_frame = 0

# Import PenguTrack
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.Filters import Filter
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
# from PenguTrack.Filters import ThreadedMultiFilter as MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import Measurement as Pengu_Meas
from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector
from PenguTrack.Detectors import rgb2gray
from PenguTrack.Stitchers import Heublein_Stitcher

import scipy.stats as ss

# Load Database
file_path = "/home/user/Desktop/Birdflight.cdb"
global db
db = DataFileExtended(file_path)

# Initialise PenguTrack
object_size = 1  # Object diameter (smallest)
object_number = 1  # Number of Objects in First Track
object_area = 3

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=2, timeconst=1.)

X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                       np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
# MultiKal.LogProbabilityThreshold = -300.
MultiKal.MeasurementProbabilityThreshold = 0.
# MultiKal = MultiFilter(Filter, model)
print("Initialized Tracker")

# Init_Background from Image_Median
# Initialize segmentation with init_image and start updating the first 10 frames.
N = db.getImages().count()
init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                           for j in np.arange(0,10)], axis=0), dtype=np.int)
# VB = ViBeSegmentation(n=2, init_image=init, n_min=2, r=25, phi=1)
VB = ViBeSegmentation(n=3, init_image=init, n_min=3, r=40, phi=1)
print("Debug")
for i in range(10,20):
    mask = VB.detect(db.getImage(frame=i).data, do_neighbours=False)
print("Detecting!")


import matplotlib.pyplot as plt
# for i in range(10306,10311):
#     mask = VB.detect(db.getImage(frame=i).data, do_neighbours=False)
#     fig, ax = plt.subplots(1)
#     # ax.imshow(np.vstack((mask*2**8, db.getImage(frame=i).data)))
#     ax.imshow(np.vstack((mask[:,16000:18000]*2**8, db.getImage(frame=i).data[:,16000:18000])))
#     plt.show()

# Initialize Detector
AD = AreaDetector(object_area, object_number, upper_limit=10, lower_limit=1)
print('Initialized')

# SetMaskType
if db.getMaskType(name="PT_Mask_Type"):
    PT_Mask_Type = db.getMaskType(name="PT_Mask_Type")
else:
    PT_Mask_Type = db.setMaskType(name="PT_Mask_Type", color="#FF6633")

# Define ClickPoints Marker
if db.getMarkerType(name="PT_Detection_Marker"):
    marker_type = db.getMarkerType(name="PT_Detection_Marker")
else:
    marker_type = db.setMarkerType(name="PT_Detection_Marker", color="#FFFF00", style='{"scale":0.8}')
if db.getMarkerType(name="PT_Track_Marker"):
    marker_type2 = db.getMarkerType(name="PT_Track_Marker")
else:
    marker_type2 = db.setMarkerType(name="PT_Track_Marker", color="#00FF00", mode=db.TYPE_Track)
if db.getMarkerType(name="PT_Prediction_Marker"):
    marker_type3 = db.getMarkerType(name= "PT_Prediction_Marker")
else:
    marker_type3 = db.setMarkerType(name="PT_Prediction_Marker", color="#0000FF")
if db.getMarkerType(name="PT_Stitch_Marker"):
    marker_type4 = db.getMarkerType(name="PT_Stitch_Marker")
else:
    marker_type4 = db.setMarkerType(name="PT_Stitch_Marker", color="#FF8800", mode=db.TYPE_Track)

# Delete Old Tracks
db.deleteMarkers(type=marker_type)
db.deleteMarkers(type=marker_type2)
db.deleteMarkers(type=marker_type3)
db.deleteMarkers(type=marker_type4)

db.deleteTracks(type=marker_type2)
db.deleteTracks(type=marker_type4)



# Start Iteration over Images
print('Starting Iteration')
images = db.getImageIterator(start_frame=20)#start_frame=start_frame, end_frame=3)
# images = db.getImageIterator(start_frame=10272, end_frame=10311)#start_frame=start_frame, end_frame=3)

start = time()
for image in images:
    print(time()-start)
    start = time()
    i = image.get_id()

    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Segmentation step
    SegMap = VB.detect(image.data, do_neighbours=False)

    print(time()-start)
    start = time()

    # print(SegMap.shape)
    # print(image.data.shape)

    # Setting Mask in ClickPoints
    db.setMask(image=image, data=(PT_Mask_Type.index*(~SegMap).astype(np.uint8)))
    print("Mask save")


    SegMap = db.getMask(image=image).data
    Mask = ~SegMap.astype(bool)
    Positions = AD.detect(Mask)
    X = np.asarray([[pos.PositionX, pos.PositionY] for pos in Positions])
    Positions = [pos for pos in Positions if np.sum(((pos.PositionX-X.T[0])**2+(pos.PositionY-X.T[1])**2)**0.5 < 200) < 10]
    print("Found %s animals!"%len(Positions))

    if len(Positions)==0:
        continue

    # for pos1 in Positions:
    #     a = float(pos1.Log_Probability)
    #     dists = [np.linalg.norm([pos1.PositionX-pos2.PositionX,
    #                                                          pos1.PositionY - pos2.PositionY]) for pos2 in Positions]
    #     pos1.Log_Probability -= np.log(np.mean(dists))
    #     print(str(a), str(pos1.Log_Probability), str(a-pos1.Log_Probability))
    with db.db.atomic() as transaction:
        print("Tracking")
        # Update Filter with new Detections
        MultiKal.update(z=Positions, i=i)

        # Get Tracks from Filters
        for k in MultiKal.ActiveFilters.keys():
            x = y = np.nan
            # Case 1: we tracked something in this filter
            if i in MultiKal.ActiveFilters[k].Measurements.keys():
                meas = MultiKal.ActiveFilters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                prob = MultiKal.ActiveFilters[k].log_prob(keys=[i], compare_bel=False)

            # Case 3: we want to see the prediction markers
            if i in MultiKal.ActiveFilters[k].Predicted_X.keys():
                pred_x, pred_y = MultiKal.Model.measure(MultiKal.ActiveFilters[k].Predicted_X[i])

            # For debugging detection step we set markers at the log-scale detections
            try:
                db.setMarker(image=image, x=y, y=x, text="Detection %s, %s"%(k, meas.Log_Probability), type=marker_type)
            except:
                pass

            # Write assigned tracks to ClickPoints DataBase
            if i in MultiKal.ActiveFilters[k].Predicted_X.keys():
                pred_marker = db.setMarker(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                       type=marker_type3)

            if np.isnan(x) or np.isnan(y):
                pass
            else:
                if db.getTrack(k+100):
                    track_marker = db.setMarker(image=image, type=marker_type2, track=(100+k), x=y, y=x,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    # print('Set Track(%s)-Marker at %s, %s' % ((100+k), x, y))
                else:
                    db.setTrack(marker_type2, id=100+k, hidden=False)
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=y, y=x,
                                     text='Track %s, Prob %.2f, CRITICAL' % ((100+k), prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=100+k, x=y, y=x,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    # print('Set new Track %s and Track-Marker at %s, %s' % ((100+k), x, y))

                # Save measurement in Database
                db.setMeasurement(marker=track_marker, log=meas.Log_Probability, x=x, y=y)
    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')

def trans_func(pos):
    x, y, z = pos
    return y, x, z

# # Initialize Stitcher
# stitcher = Heublein_Stitcher(25, 0., 50, 60, 200, 5)
# stitcher.add_PT_Tracks_from_Tracker(MultiKal.Filters)
# print("Initialized Stitcher")
# stitcher.stitch()
# stitcher.save_tracks_to_db(file_path, marker_type4, function=trans_func)
# print("Written Stitched Tracks to DB")
#
# db.deleteTracks(id=[track.id for track in db.getTracks(type=marker_type4) if len(track.markers) < 3])
# print("Deleted short tracks")

