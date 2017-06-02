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
q = 100
r = 2
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
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import SiAdViBeSegmentation
from PenguTrack.Detectors import Measurement as Pengu_Meas
from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector
from PenguTrack.Detectors import rgb2gray
from PenguTrack.Stitchers import Heublein_Stitcher

import scipy.stats as ss

# Load Database
file_path = "/home/alex/Masterarbeit/770_PANA/blabla.cdb"
global db
db = DataFileExtended(file_path)

# Initialise PenguTrack
object_size = 0.5  # Object diameter (smallest)
penguin_height = 0.462#0.575
penguin_width = 0.21
object_number = 300  # Number of Objects in First Track
object_area = 55

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=3, timeconst=0.5)

X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                       np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

# Init_Background from Image_Median
N = db.getImages().count()
init = np.array(np.median([np.asarray(rgb2gray(db.getImage(frame=j).data), dtype=np.int)
                           for j in np.random.randint(0, N, 10)], axis=0), dtype=np.int)

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

# Initialize segmentation with init_image and start updating the first 10 frames.
VB = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3, 9e-3], penguin_markers, penguin_height, 500, n=5, init_image=init, n_min=3, r=10, phi=1)#, camera_h=44.)
for i in range(1,10):
    mask = VB.detect(rgb2gray(db.getImage(frame=i).data), do_neighbours=False)
print("Detecting Penguins of size ", object_area, VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)

# Initialize Detector
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
images = db.getImageIterator(start_frame=10)#start_frame=start_frame, end_frame=3)

for image in images:
    start = time()
    i = image.get_id()

    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Segmentation step
    SegMap = VB.detect(rgb2gray(image.data), do_neighbours=False)

    # Setting Mask in ClickPoints
    db.setMask(image=image, data=(255*(~SegMap).astype(np.uint8)))
    print("Mask save")

    #
    SegMap = db.getMask(image=image).data
    Mask = ~SegMap.astype(bool)
    Positions = AD.detect(Mask)
    print("Found %s animals!"%len(Positions))

    # Project from log-scale map to ortho-map and rescale to metric coordinates
    for pos in Positions:
        pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY
                                                      , pos.PositionX])
        pos.PositionX *= (VB.Max_Dist/VB.height)
        pos.PositionY *= (VB.Max_Dist/VB.height)
        pos.PositionZ = -VB.height


    #if np.all(Positions != np.array([])):
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

                # rescale to pixel coordinates
                x_px = x * (VB.height/VB.Max_Dist)
                y_px = y * (VB.height/VB.Max_Dist)
                prob = MultiKal.ActiveFilters[k].log_prob(keys=[i], compare_bel=False)

            # Case 3: we want to see the prediction markers
            if i in MultiKal.ActiveFilters[k].Predicted_X.keys():
                pred_x, pred_y, pred_z = MultiKal.Model.measure(MultiKal.ActiveFilters[k].Predicted_X[i])
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
    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')

def trans_func(pos):
    x, y, z = pos
    x_px = x * (VB.height/VB.Max_Dist)
    y_px = y * (VB.height/VB.Max_Dist)
    x_new, y_new = VB.warp_orth([VB.Res * (y_px - VB.width / 2.), VB.Res * (VB.height - x_px)])
    return x_new, y_new, z

# Initialize Stitcher
stitcher = Heublein_Stitcher(25, 0., 50, 60, 200, 5)
stitcher.add_PT_Tracks_from_Tracker(MultiKal.Filters)
print("Initialized Stitcher")
stitcher.stitch()
stitcher.save_tracks_to_db(file_path, marker_type4, function=trans_func)
print("Written Stitched Tracks to DB")

db.deleteTracks(id=[track.id for track in db.getTracks(type=marker_type4) if len(track.markers) < 3])
print("Deleted short tracks")

