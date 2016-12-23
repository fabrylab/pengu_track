from __future__ import division, print_function
import cv2
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import peewee
import sys

import clickpoints

# Connect to database
db = clickpoints.DataFile("penguin_data.cdb")
start_frame = 0

#Initialise PenguTrack
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import ViBeSegmentation
from PenguTrack.Detectors import BlobDetector

import scipy.stats as ss

object_size = 11  # Object diameter (smallest)
object_number = 2  # Number of Objects in First Track

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
                           for j in np.random.randint(0, N, 20)], axis=0), dtype=np.int)

# Init Segmentation Module with Init_Image
VB = ViBeSegmentation(init_image=init, n_min=18, r=20, phi=1)
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
    pass

if "measurement" not in db.db.get_tables():
    db.db.connect()
    db.db.create_table(Measurement)

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
                marker = db.setMarker(image=image, x=pred_x, y=pred_y, text="Track %s" % k, type=marker_type3)
                try:
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                    db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % (k, prob))
                    print('Set Track(%s)-Marker at %s, %s' % (k, x, y))
                except:
                    db.setTrack(marker_type2, id=k)
                    # db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y, text='Track %s, Prob %.2f'%(k, prob))
                    if k == MultiKal.CriticalIndex:
                        db.setMarker(image=image, type=marker_type, x=x, y=y,
                                     text='Track %s, Prob %.2f, CRITICAL' % (k, prob))
                    db.setMarker(image=image, type=marker_type2, track=k, x=x, y=y,
                                 text='Track %s, Prob %.2f' % (k, prob))
                    print('Set new Track %s and Track-Marker at %s, %s' % (k, x, y))


            # add all measurement entries to measurements table
            # iterate over all attributes of measurement
            for attr in meas.__dict__.keys():
                # test if collumn exists
                if not attr in [col.name for col in db.db.get_columns("measurement")]:
                    #if not use mirgration tool
                    import playhouse.migrate
                    import peewee
                    migrator = playhouse.migrate.SqliteMigrator(db.db)

                    # add column with adequate dtype
                    if type(meas.__dict__[attr]) in [float, np.float, np.float16, np.float32, np.float64]:
                        col = peewee.FloatField(default=0., null=True)
                    elif type(meas.__dict__[attr]) in [int, np.int, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64]:
                        col = peewee.IntegerField(default=0, null=True)
                    elif type(meas.__dict__[attr]) == type(None):
                        col = peewee.IntegerField(default=0, null=True)
                    else:
                        print(attr, type(meas.__dict__[attr]))
                        raise TypeError("Not a database type!")

                    #do migration
                    playhouse.migrate.migrate(migrator.add_column("measurement", attr, col),)

            # i wanted to do this, it failed
            db.db.connect()
            meas_entry = Measurement()
            for key in meas.__dict__.keys():
                setattr(meas_entry, key, meas.__dict__[key])
                print(getattr(meas_entry, key), meas.__dict__[key])
            # i tried to fall back to hard coding o spot the error, but it did also not work out
            # meas_entry = Measurement.create(Log_Probability=1., PositionX=1., PositionY=1., Frame=10, Track_Id=1)
            # save the entry
            meas_entry.save()

    print("Got %s Filters" % len(MultiKal.ActiveFilters.keys()))

print('done with Tracking')