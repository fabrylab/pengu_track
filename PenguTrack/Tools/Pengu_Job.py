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

# Connect to database
for p in sys.argv:
	print(p)
file_path = str(sys.argv[1])
if platform.system() != 'Linux':
    file_path = file_path.replace("/mnt/jobs", r"//131.188.117.98/shared/jobs")
#path.normpath(file_path)
db = clickpoints.DataFile(file_path)
start_frame = 0

#Initialise PenguTrack
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation
from PenguTrack.Detectors import BlobDetector
from PenguTrack.Detectors import AreaDetector
from PenguTrack.Detectors import Measurement as Pengu_Meas
from PenguTrack.Detectors import AreaDetector
from PenguTrack.Detectors import BlobSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation

import scipy.stats as ss


#Initialise PenguTrack
object_size = 0.5  # Object diameter (smallest)
penguin_height = 0.462#0.575
penguin_width = 0.21
object_number = 500  # Number of Objects in First Track

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=2, timeconst=0.5)

uncertainty = 4 * object_size
X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([uncertainty, uncertainty])  # Prediction uncertainty
R = np.diag([object_size, object_size])  # Measurement uncertainty

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
VB = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3, 9e-3], penguin_markers, penguin_height, 500, n=2, init_image=init, n_min=2, r=10, phi=1)

#for i in range(1,20)[::-1]:
#    VB.detect(db.getImage(frame=i).data, do_neighbours=False)

BS = BlobSegmentation(15, min_size=4)
imgdata = VB.horizontal_equalisation(db.getImage(frame=0).data)

# Init Detection Module
# BD = BlobDetector(object_size, object_number)
print("Detecting Penguins of size ", 42, VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
AD = AreaDetector(42)#VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
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
    assert not (marker is None), "Measurement must refer to a marker."
    try:
        item = db.table_measurement.get(marker=marker)
    except peewee.DoesNotExist:
        item = db.table_measurement()

    dictionary = dict(marker=marker, x=x, y=y)
    for key in dictionary:
        if dictionary[key] is not None:
            setattr(item, key, dictionary[key])
    item.save()
    return item

db.setMeasurement = setMeasurement

# Start Iteration over Images
print('Starting Iteration')
images = db.getImageIterator(start_frame=21, end_frame=555)#start_frame=start_frame, end_frame=3)
for image in images:

    i = image.get_id()
    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Segmentation step
    # SegMap = VB.detect(image.data, do_neighbours=False)
    SegMap = db.getMask(image=image).data

    # Setting Mask in ClickPoints
    # db.setMask(image=image, data=(255*(~SegMap).astype(np.uint8)))
    # print("Mask save")

    Mask = ~SegMap.astype(bool)

    # Detection of regions with distinct areas
    from skimage.morphology import binary_closing
    selem4 = np.array([[0,1,1,1,0],
                   [0,1,1,1,0],
                   [0,1,1,1,0],
                   [0,1,1,1,0],
                   [0,1,1,1,0]])
    Positions = AD.detect(binary_closing(Mask, selem=selem4))
    #print(Positions)

    # Project from log-scale map to ortho-map and rescale to metric coordinates
    for pos in Positions:
        pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY
                                                      , pos.PositionX])
        pos.PositionX *= (VB.Max_Dist/VB.height)
        pos.PositionY *= (VB.Max_Dist/VB.height)


    if np.all(Positions != np.array([])):
        # Update Filter with new Detections
        try:
            MultiKal.update(z=Positions, i=i)
        except IndexError:
            continue
        # Get Tracks from Filters
        for k in MultiKal.Filters.keys():
            x = y = np.nan
            # Case 1: we tracked something in this filter
            if i in MultiKal.Filters[k].Measurements.keys():
                meas = MultiKal.Filters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                # rescale to pixel coordinates
                x_px = x * (VB.height/VB.Max_Dist)
                y_px = y * (VB.height/VB.Max_Dist)
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

            # Case 3: we want to see the prediction markers
            if i in MultiKal.Filters[k].Predicted_X.keys():
                pred_x, pred_y = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
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
            if i in MultiKal.Filters[k].Predicted_X.keys():
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
                    db.setTrack(marker_type2, id=100+k)
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x_img, y=y_img,
                                     text='Track %s, Prob %.2f, CRITICAL' % ((100+k), prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=100+k, x=x_img, y=y_img,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % ((100+k), x_img, y_img))

                # Save measurement in Database
                db.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')

GT_Type = db.getMarkerType(name="GT_Detections")
GT_Markers = db.getMarkers(type=GT_Type)
Auto_Type = db.getMarkerType(name="PT_Track_Marker")
Auto_Markers = db.getMarkers(type=Auto_Type, frame=0)

GT_pos = np.asarray([[marker.x, marker.y] for marker in GT_Markers])
auto_pos = np.asarray([[marker.x, marker.y] for marker in Auto_Markers])

dist_mat = np.linalg.norm(GT_pos[None,:].T-auto_pos[:,None].T, axis=0)

dists = np.amin(dist_mat, axis=1)
n_false_positive = len(dists[dists>VB.Penguin_Size])+dist_mat.shape[1]-dist_mat.shape[0]
n_correct = len(dists[dists<VB.Penguin_Size])
n_not_found = dist_mat.shape[0]-n_correct
total_rms_err = np.sqrt(np.mean(np.square(dists[dists<VB.Penguin_Size])))


with open("eval_0.txt","a") as myfile:
    myfile.write("\n")
    myfile.write("P-Faktor %s"%sys.argv[1])
    myfile.write("\n")
    myfile.write("N-Total: %s GT, %s Auto"%dist_mat.shape)
    myfile.write("\n")
    myfile.write("Correct Detections: %s absolute, %s relative %%"%(n_correct, 100*n_correct/dist_mat.shape[0]))
    myfile.write("\n")
    myfile.write("False Positives: %s absolute, %s relative %%"%(n_false_positive, 100*n_false_positive/dist_mat.shape[1]))
    myfile.write("\n")
    myfile.write("False Negative: %s absolute, %s relative %%"%(n_not_found, 100*n_not_found/dist_mat.shape[0]))
    myfile.write("\n")
    myfile.write("Total RMS-Error: %s absolute, %s relative %%"%(total_rms_err, 100*total_rms_err/VB.Penguin_Size))
    myfile.write("\n")
