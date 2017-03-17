from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import AreaDetector
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Filters import MultiFilter
from PenguTrack.Filters import KalmanFilter

from PenguTrack.Detectors import Measurement as PT_Measurement

import clickpoints
import numpy as np
import peewee
import scipy.stats as ss

MaxP_db = clickpoints.DataFile("./max_Proj.cdb")
MinP_db = clickpoints.DataFile("./min_Proj.cdb")
MaxI_db = clickpoints.DataFile("./max_Indizes.cdb")
db = MaxP_db  # for convenience

#Initialise PenguTrack
object_size = 15  # Object diameter (smallest)
object_number = 200  # Number of Objects in First Track

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=3, timeconst=0.5)

uncertainty = 8 * object_size
X = np.zeros(6).T  # Initial Value for Position
Q = np.diag([uncertainty, uncertainty, uncertainty])  # Prediction uncertainty
R = np.diag([uncertainty * 2, uncertainty * 2, uncertainty * 2])  # Measurement uncertainty

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

VB = ViBeSegmentation(init_image=init, n=2, n_min=2, r=10, phi=1)

#for i in range(1,10)[::-1]:
#    VB.detect(db.getImage(frame=i).data, do_neighbours=False)

# Init Detection Module
# BD = BlobDetector(object_size, object_number)
print("Detecting Cells of size ", 400)
AD = AreaDetector(400)
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
    z = peewee.FloatField()

if "measurement" not in db.db.get_tables():
    db.db.connect()
    Measurement.create_table()#  important to respect unique constraint

db.table_measurement = Measurement   # for consistency


def setMeasurement(marker=None, log=None, x=None, y=None, z=None):
    assert not (marker is None), "Measurement must refer to a marker."
    try:
        item = db.table_measurement.get(marker=marker)
    except peewee.DoesNotExist:
        item = db.table_measurement()

    dictionary = dict(marker=marker, x=x, y=y, z=z)
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
    Index_Image = MaxI_db.getImage(frame=i).data
    MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

    # Detection step
    SegMap = VB.detect(image.data, do_neighbours=False)

    # Setting Mask in ClickPoints
    db.setMask(image=image, data=(255*(~SegMap).astype(np.uint8)))
    print("Mask save")

    # Detection Step
    Positions2D = AD.detect(~db.getMask(image=image).data.astype(bool))
    Positions3D = []
    for pos in Positions2D:
        posZ = Index_Image[int(pos.PositionX), int(pos.PositionY)]
        Positions3D.append(PT_Measurement(pos.Log_Probability,
                                          [pos.PositionX, pos.PositionY, posZ],
                                          frame=pos.Frame,
                                          track_id=pos.Track_Id))
    Positions = Positions3D  # convenience
    print("Detected %s objects"%len(Positions))

    if np.all(Positions != np.array([])):

        # Update Filter with new Detections
        try:
            MultiKal.update(z=Positions, i=i)
        except IndexError:
            continue
        # Get Tracks from Filter (a little dirty)
        for k in MultiKal.Filters.keys():
            x = y = z = np.nan
            if i in MultiKal.Filters[k].Measurements.keys():
                meas = MultiKal.Filters[k].Measurements[i]
                x = meas.PositionX
                y = meas.PositionY
                z = meas.PositionZ
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)
            elif i in MultiKal.Filters[k].X.keys():
                meas = None
                x, y, z = MultiKal.Model.measure(MultiKal.Filters[k].X[i])
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

            if i in MultiKal.Filters[k].Measurements.keys():
                pred_x, pred_y, pred_z = MultiKal.Model.measure(MultiKal.Filters[k].Predicted_X[i])
                prob = MultiKal.Filters[k].log_prob(keys=[i], compare_bel=False)

            try:
                yy, xx = y, x
                db.setMarker(image=image, x=yy, y=xx, text="Detection %s"%k, type=marker_type)
            except:
                pass
            # Write assigned tracks to ClickPoints DataBase
            if np.isnan(x) or np.isnan(y):
                pass
            else:
                pred_marker = db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s" % (100+k), type=marker_type3)
                if db.getTrack(k+100):
                    track_marker = db.setMarker(image=image, type=marker_type2, track=(100+k), x=x, y=y,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set Track(%s)-Marker at %s, %s' % ((100+k), x, y))
                else:
                    db.setTrack(marker_type2, id=100+k)
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % ((100+k), prob))
                    track_marker = db.setMarker(image=image, type=marker_type2, track=100+k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % ((100+k), prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % ((100+k), x, y))

                db.setMeasurement(marker=track_marker, log=prob, x=x, y=y, z=z)

    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')