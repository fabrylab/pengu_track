from __future__ import division, print_function
#import resource

# import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from skimage.morphology import binary_dilation
import peewee
import sys

import clickpoints

from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
# from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation
# from PenguTrack.Detectors import BlobDetector
from PenguTrack.Detectors import AreaDetector
from PenguTrack.Detectors import BlobSegmentation
# from PenguTrack.Detectors import Measurement as Pengu_Meas

import scipy.stats as ss

#resource.setrlimit(resource.RLIMIT_AS, (12000 * 1048576L, -1L))

# Connect to database
db = clickpoints.DataFile("C:\\Users\\User\\Desktop\\241.cdb")
start_frame = 0

#Initialise PenguTrack
object_size = 2  # Object diameter (smallest)
penguin_height = 0.462#0.575
penguin_width = 0.21
object_number = 100  # Number of Objects in First Track

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

#for i in range(1,10)[::-1]:
#    VB.detect(db.getImage(frame=i).data, do_neighbours=False)

BS = BlobSegmentation(15, min_size=4)
imgdata = VB.horizontal_equalisation(db.getImage(frame=0).data)

# Init Detection Module
# BD = BlobDetector(object_size, object_number)
print("Detecting Penguins of size ", 100, VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
AD = AreaDetector(100)#VB.Penguin_Size*penguin_width*VB.Penguin_Size/penguin_height)
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
    db.db.connect()
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
images = db.getImageIterator()#start_frame=start_frame, end_frame=3)
for image in images:

    i = image.get_id()
    # Prediction step
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Detection step
    SegMap = VB.detect(image.data, do_neighbours=False)
    # import matplotlib.pyplot as plt
    # plt.imshow(SegMap)
    # plt.figure()
    # from scipy import ndimage
    # k = np.zeros((object_size+2, VB.Penguin_Size+2))
    # k[1:-1, 1:-1] = 1.
    # SegMap = ndimage.convolve(SegMap, k.astype(bool).T, mode="constant", cval=0.)
    # plt.imshow(SegMap)
    # plt.figure()
    import matplotlib.pyplot as plt
    # SegMap = binary_dilation(SegMap)
    # SegMap = np.asarray(SegMap).astype(bool)
    # plt.imshow(SegMap)
    # SegMap2 = BS.detect(VB.horizontal_equalisation(image.data))
    # SegMap2 = np.asarray(SegMap2).astype(bool)
    # plt.figure()
    # plt.imshow(SegMap2)
    # SegMap = SegMap & SegMap2
    # plt.figure()
    # plt.imshow(SegMap)
    # plt.show()

    from skimage.measure import label, regionprops
    # labeled = label(SegMap, connectivity=2)
    # bad_ids = [prop.label for prop in regionprops(labeled) if prop.area < VB.Penguin_Size]
    # for id in bad_ids:
    #     labeled[labeled == id] = 0
    # plt.imshow(label(SegMap, connectivity=2))
    # plt.show()

    # labeled[labeled != 0] = 1

    db.setMask(image=image, data=(255*(~SegMap).astype(np.uint8)))
    Positions = AD.detect(~db.getMask(image=image).data.astype(bool))
    print(Positions)

    # def trafo(x):
    #     x -= VB.width
    #     x *= (VB.Max_Dist/VB.height)/(VB.h_p/VB.Penguin_Size)
    #     x += VB.width
    #     return x
    for pos in Positions:
        pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY
                                                      , pos.PositionX])

    # Positions = [VB.back_warp_orth(VB.warp_log([VB.Res * (pos.PositionY - VB.width / 2.),
    #                                             VB.Res * (VB.height - pos.PositionX)])) for pos in Positions]
    # Positions = [Pengu_Meas(1., [pos[0], pos[1]]) for pos in Positions]
    # xxyy = np.array([[pos.PositionX, pos.PositionY] for pos in Positions])
    # plt.scatter(xxyy.T[0], xxyy.T[1])
    # plt.show()
    # x_p, y_p = Positions
    # x_p = x_p-
    # Positions = VB.warp2(Positions)
    # xy = np.array(VB.grid)
    # xx, yy = VB.warp_orth(VB.back_warp_orth(xy))
    # plt.scatter(xx, yy)
    # plt.show()
    # Setting Mask in ClickPoints
    print("Mask save")
    n = 1

    if np.all(Positions != np.array([])):

        # Update Filter with new Detections
        try:
            MultiKal.update(z=Positions, i=i)
        except IndexError:
            continue
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

            # x, y = VB.warp_orth([x, y])
            # x = VB.Res*(x-VB.width/2.)
            # y = VB.height-y
            try:
                yy, xx = VB.orth_to_log([y,x])
                db.setMarker(image=image, x=yy, y=xx, text="Detection %s"%k, type=marker_type)
            except:
                pass
            x, y = VB.warp_orth([VB.Res * (y - VB.width / 2.), VB.Res * (VB.height - x)])
            pred_x, pred_y = VB.warp_orth([VB.Res * (pred_y - VB.width / 2.), VB.Res * (VB.height - pred_x)])


            # Write assigned tracks to ClickPoints DataBase
            if np.isnan(x) or np.isnan(y):
                pass
            else:
                pred_marker = db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s" % (100+k), type=marker_type3)
                if db.getTrack(k+100):
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    #if k == MultiKal.CriticalIndex:
                    #    db.setMarker(image=image, type=marker_type, x=x, y=y,
                    #                 text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=(100+k), x=x, y=y,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set Track(%s)-Marker at %s, %s' % ((100+k), x, y))
                else:
                    db.setTrack(marker_type2, id=100+k)
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % ((100+k), prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=100+k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % ((100+k), x, y))

                # db.db.connect()
                # meas_entry = Measurement(marker=track_marker, log=prob, x=x, y=y)
                # meas_entry.save()
                db.setMeasurement(marker=track_marker, log=prob, x=x, y=y)

    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')