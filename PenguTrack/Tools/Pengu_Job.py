from __future__ import division, print_function

import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import binary_dilation
import peewee
import sys

import clickpoints
import platform
from os import path
from time import time

# Connect to database
# for p in sys.argv:
# 	print(p)
# file_path = str(sys.argv[1])
# q = float(sys.argv[2])
# r = float(sys.argv[3])
q = 1
r = 1
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

file_path = "/home/user/Desktop/FullDayAnalysis/FullDay.cdb"
global db
db = clickpoints.DataFile(file_path)


start_frame = 0

#Initialise PenguTrack
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation
from PenguTrack.Detectors import BlobDetector
from PenguTrack.Detectors import Measurement as Pengu_Meas
from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector
from PenguTrack.Detectors import BlobSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation
from PenguTrack.Detectors import rgb2gray

import scipy.stats as ss


#Initialise PenguTrack
object_size = 0.5  # Object diameter (smallest)
penguin_height = 0.462#0.575
penguin_width = 0.21
object_number = 300  # Number of Objects in First Track
object_area = 55

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=2, timeconst=0.5)

X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                       np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

# MultiKal.LogProbabilityThreshold = -10.

# Init_Background from Image_Median
N = db.getImages().count()
init = np.array(np.median([np.asarray(rgb2gray(db.getImage(frame=j).data), dtype=np.int)
                           for j in np.random.randint(0, N, 10)], axis=0), dtype=np.int)

# Init Segmentation Module with Init_Image

# VB = ViBeSegmentation(init_image=init, n_min=18, r=20, phi=1)
# Load horizon-markers
horizont_type = db.getMarkerType(name="Horizon")
try:
    horizon_markers = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T
except ValueError:
    raise ValueError("No markers with name 'Horizon'!")
# Load penguin-markers
penguin_type = db.getMarkerType(name="Penguin_Size")
try:
    penguin_markers = np.array([[m.x1, m.y1, m.x2, m.y2] for m in db.getLines(type="Penguin_Size")]).T
except ValueError:
    raise ValueError("No markers with name 'Horizon'!")

# Initialize detector and start backwards.
VB = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3, 9e-3], penguin_markers, penguin_height, 500, n=5, init_image=init, n_min=3, r=10, phi=1)#, camera_h=44.)
# VB.camera_h = 25.

for i in range(1,10):
    mask = VB.detect(rgb2gray(db.getImage(frame=i).data), do_neighbours=False)

BS = BlobSegmentation(15, min_size=4)
imgdata = VB.horizontal_equalisation(rgb2gray(db.getImage(frame=0).data))

print("Detecting Penguins of size ", object_area, VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
AD = AreaDetector(object_area, object_number, upper_limit=80, lower_limit=20)#VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
print('Initialized')

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

# Delete Old Tracks
db.deleteMarkers(type=marker_type)
db.deleteMarkers(type=marker_type2)
db.deleteMarkers(type=marker_type3)

db.deleteTracks(type=marker_type3)

# append Database if necessary
import peewee

class Measurement(db.base_model):
    import peewee
    # full definition here - no need to use migrate
    marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name="measurement", on_delete='CASCADE') # reference to frame and track via marker!
    log = peewee.FloatField(default=0)
    x = peewee.FloatField()
    y = peewee.FloatField()

if "measurement" not in db.db.get_tables():
    try:
        db.db.connect()
    except peewee.OperationalError:
        pass
    Measurement.create_table()#  important to respect unique constraint

db.table_measurement = Measurement   # for consistency


def setMeasurement(marker=None, log=None, x=None, y=None):
    import peewee
    assert not (marker is None), "Measurement must refer to a marker."
    try:
        item = db.table_measurement.get(marker=marker)
    except peewee.DoesNotExist:
        item = db.table_measurement()

    dictionary = dict(marker=marker, log=log, x=x, y=y)
    for key in dictionary:
        if dictionary[key] is not None:
            setattr(item, key, dictionary[key])
    item.save()
    return item

db.setMeasurement = setMeasurement

# Start Iteration over Images
print('Starting Iteration')
images = db.getImageIterator(start_frame=11)#start_frame=start_frame, end_frame=3)
PredictionTimes = []
SegmentationTimes = []
DetectionTimes = []
TrackingTimes = []
TrackWritingTimes = []
for image in images:
    start = time()
    i = image.get_id()
    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
    PredictionTimes.append((time()-start))
    print("Time for Prediction: %s"%PredictionTimes[-1])
    start = time()

    # Segmentation step
    SegMap = VB.detect(rgb2gray(image.data), do_neighbours=False)

    SegmentationTimes.append((time()-start))
    print("Time for segmentation: %s"%SegmentationTimes[-1])

    # Setting Mask in ClickPoints
    db.setMask(image=image, data=(255*(~SegMap).astype(np.uint8)))
    print("Mask save")

    start = time()
    SegMap = db.getMask(image=image).data
    Mask = ~SegMap.astype(bool)

    Positions = AD.detect(Mask)

    DetectionTimes.append((time()-start))
    print("Time for Detection: %s"%DetectionTimes[-1])
    start = time()
    print("Found %s animals!"%len(Positions))

    # Project from log-scale map to ortho-map and rescale to metric coordinates
    for pos in Positions:
        pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY
                                                      , pos.PositionX])
        pos.PositionX *= (VB.Max_Dist/VB.height)
        pos.PositionY *= (VB.Max_Dist/VB.height)


    if np.all(Positions != np.array([])):
        print("Tracking")
        # Update Filter with new Detections
        try:
            MultiKal.update(z=Positions, i=i)
        except IndexError:
            continue
        TrackingTimes.append((time()-start))
        print("Time for tracking: %s"%TrackingTimes[-1])
        start = time()
        # Get Tracks from Filters
        for k in MultiKal.ActiveFilters.keys():
            x = y = np.nan
            # Case 1: we tracked something in this filter
            if i in MultiKal.ActiveFilters[k].Measurements.keys():
                meas = MultiKal.ActiveFilters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                # rescale to pixel coordinates
                x_px = x * (VB.height/VB.Max_Dist)
                y_px = y * (VB.height/VB.Max_Dist)
                prob = MultiKal.ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
            # Case 3: we want to see the prediction markers
            if i in MultiKal.ActiveFilters[k].Predicted_X.keys():
                pred_x, pred_y = MultiKal.Model.measure(MultiKal.ActiveFilters[k].Predicted_X[i])
                # rescale to pixel coordinates
                pred_x_px = pred_x * (VB.height/VB.Max_Dist)
                pred_y_px = pred_y * (VB.height/VB.Max_Dist)

            # For debugging detection step we set markers at the log-scale detections
            try:
                yy, xx = VB.orth_to_log([y_px, x_px])
                db.setMarker(image=image, x=yy, y=xx, text="Detection %s"%k, type=marker_type)
            except:
                pass

            # Warp back to image coordinates
            x_img, y_img = VB.warp_orth([VB.Res * (y_px - VB.width / 2.), VB.Res * (VB.height - x_px)])
            pred_x_img, pred_y_img = VB.warp_orth([VB.Res * (pred_y_px - VB.width / 2.), VB.Res * (VB.height - pred_x_px)])

            # Write assigned tracks to ClickPoints DataBase
            if i in MultiKal.ActiveFilters[k].Predicted_X.keys():
                pred_marker = db.setMarker(image=image, x=pred_x_img, y=pred_y_img, text="Track %s" % (100 + k),
                                       type=marker_type3)
            if np.isnan(x) or np.isnan(y):
                pass
            else:
                if db.getTrack(k+100):
                    track_marker = db.setMarker(image=image, type=marker_type2, track=(100+k), x=x_img, y=y_img,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set Track(%s)-Marker at %s, %s' % ((100+k), x_img, y_img))
                else:
                    db.setTrack(marker_type2, id=100+k, hidden=False)
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x_img, y=y_img,
                                     text='Track %s, Prob %.2f, CRITICAL' % ((100+k), prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=100+k, x=x_img, y=y_img,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % ((100+k), x_img, y_img))

                # Save measurement in Database
                db.setMeasurement(marker=track_marker, log=meas.Log_Probability, x=x, y=y)
    TrackWritingTimes.append((time()-start))
    print("Time for writing tracks: %s"%TrackWritingTimes[-1])
    start = time()
    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')

# for track in db.getTracks(type=marker_type2):
#     if len(db.getMarkers(track=track)) < 3:
#         db.deleteTracks(id=track)
db.deleteTracks(id=[track.id for track in db.getTracks() if len(db.getMarkers(track=track)) < 3])

# print([[np.mean(t), np.std(t), np.amin(t), np.amax(t)] for t in PredictionTimes, SegmentationTimes, DetectionTimes, TrackingTimes, TrackWritingTimes])
# os.system("cp ~/Desktop/TODO/%s_done.cdb %s "%(path[:-7], path[-7:-4]))

# os.system("rm -r ~/Desktop/TODO/*")
# track_length_table = np.asarray([[t.id, len(t.markers)] for t in db.getTracks(type=marker_type2)])
# db.deleteTracks(id=track_length_table.T[0][track_length_table.T[1]<4])
# print("Deleted shorter Tracks")
#
# GT_Type = db.getMarkerType(name="GT_Detections")
# GT_Markers = db.getMarkers(type=GT_Type)
# Auto_Type = db.getMarkerType(name="PT_Track_Marker")
# Auto_Markers = db.getMarkers(type=Auto_Type, frame=0)
#
# GT_pos = np.asarray([[marker.x, marker.y] for marker in GT_Markers])
# auto_pos = np.asarray([[marker.x, marker.y] for marker in Auto_Markers])
#
# dist_mat = np.linalg.norm(GT_pos[None,:].T-auto_pos[:,None].T, axis=0)
#
# dists = np.amin(dist_mat, axis=1)
# n_false_positive = len(dists[dists>VB.Penguin_Size])+dist_mat.shape[1]-dist_mat.shape[0]
# n_correct = len(dists[dists<VB.Penguin_Size])
# n_not_found = dist_mat.shape[0]-n_correct
# total_rms_err = np.sqrt(np.mean(np.square(dists[dists<VB.Penguin_Size])))
#
#
# with open("eval_0.txt","a") as myfile:
#     myfile.write("\n")
#     myfile.write("P-Faktor %s"%sys.argv[1])
#     myfile.write("\n")
#     myfile.write("N-Total: %s GT, %s Auto"%dist_mat.shape)
#     myfile.write("\n")
#     myfile.write("Correct Detections: %s absolute, %s relative %%"%(n_correct, 100*n_correct/dist_mat.shape[0]))
#     myfile.write("\n")
#     myfile.write("False Positives: %s absolute, %s relative %%"%(n_false_positive, 100*n_false_positive/dist_mat.shape[1]))
#     myfile.write("\n")
#     myfile.write("False Negative: %s absolute, %s relative %%"%(n_not_found, 100*n_not_found/dist_mat.shape[0]))
#     myfile.write("\n")
#     myfile.write("Total RMS-Error: %s absolute, %s relative %%"%(total_rms_err, 100*total_rms_err/VB.Penguin_Size))
#     myfile.write("\n")
