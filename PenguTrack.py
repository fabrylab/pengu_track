from __future__ import division, print_function
import resource
resource.setrlimit(resource.RLIMIT_AS, (12000 * 1048576L, -1L))

import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import peewee
import sys

import clickpoints

# Connect to database
db = clickpoints.DataFile("/home/alex/Masterarbeit/master_project/master_project/adelie_data/770_PANA/Neuer Ordner/gsfd.cdb")
start_frame = 0

#Initialise PenguTrack
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation
from PenguTrack.Detectors import BlobDetector

import scipy.stats as ss

object_size = 22  # Object diameter (smallest)
object_number = 50  # Number of Objects in First Track

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=2, timeconst=0.5)

uncertainty = 8 * object_size
X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([uncertainty, uncertainty])  # Prediction uncertainty
R = np.diag([uncertainty * 2, uncertainty * 2])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.array([uncertainty, uncertainty]),
                       np.array([uncertainty, uncertainty]), meas_dist=Meas_Dist, state_dist=State_Dist)

# Init_Background from Image_Median
N = db.getImages().count()
init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                           for j in np.random.randint(0, N, 10)], axis=0), dtype=np.int)

# Init Segmentation Module with Init_Image
# VB = ViBeSegmentation(init_image=init, n_min=18, r=20, phi=1)

# Load horizon-markers, rotate them
horizont_type = db.getMarkerType(name="Horizon")
try:
    x, y = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T
except ValueError:
    raise ValueError("No markers with name 'Horizon'!")
VB = SiAdViBeSegmentation([x,y], 14e-3, [17e-3,13e-3], 40, 0.6, 500, n=2, init_image=init, n_min=18, r=20, phi=1)
imgdata = VB.horizontal_equalisation(db.getImage(frame=0).data, VB.Horizonmarkers, VB.F, VB.Sensor_Size, VB.H, VB.h_p, max_dist=VB.Max_Dist)
# import matplotlib.pyplot as plt
# plt.imshow(imgdata)
# plt.show()
# Init Detection Module
BD = BlobDetector(object_size, object_number)
print('Initialized')

# Define ClickPoints Marker

if db.getMarkerType(name="PT_Detection_Marker"):
    marker_type = db.getMarkerType(name="PT_Detection_Marker")
else:
    marker_type = db.setMarkerType(name="PT_Detection_Marker", color="#FF0000", style='{"scale":1.2}')
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
    # full definition here - no need to use migrate
    marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name="measurement", on_delete='CASCADE') # reference to frame and track via marker!
    log = peewee.FloatField(default=0)
    x = peewee.FloatField()
    y = peewee.FloatField()


if "measurement" not in db.db.get_tables():
    db.db.connect()
    Measurement.create_table()#  important to respect unique constraint

db.table_measurement = Measurement   # for consistency


# Start Iteration over Images
print('Starting Iteration')
images = db.getImageIterator(start_frame=start_frame, end_frame=3)
for image in images:

    i = image.get_id()
    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Detection step
    SegMap = VB.detect(image.data, do_neighbours=False)
    Positions = BD.detect(SegMap)

    # Setting Mask in ClickPoints
    db.setMask(image=image, data=(~SegMap).astype(np.uint8))
    print("Mask save")
    n = 1

    if np.all(Positions != np.array([])):

        # Update Filter with new Detections
        MultiKal.update(z=Positions, i=i)

        # Get Tracks from Filter (a little dirty)
        for k in MultiKal.Filters.keys():
            x = y = np.nan
            if i in MultiKal.Filters[k].Measurements.keys():
                meas = MultiKal.Filters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)
            elif i in MultiKal.Filters[k].X.keys():
                meas = None
                x, y = MultiKal.Model.measure(MultiKal.Filters[k].X[i])
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

            if i in MultiKal.Filters[k].Measurements.keys():
                pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

            # Write assigned tracks to ClickPoints DataBase
            if np.isnan(x) or np.isnan(y):
                pass
            else:
                pred_marker = db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s" % k, type=marker_type3)
                try:
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % (k, prob))
                    print('Set Track(%s)-Marker at %s, %s' % (k, x, y))
                except:
                    db.setTrack(marker_type2, id=k)
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % (k, prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % (k, x, y))

                db.db.connect()
                meas_entry = Measurement(marker=track_marker, log=prob, x=x, y=y)
                meas_entry.save()

    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')