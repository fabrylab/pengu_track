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
# init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
#                            for j in np.arange(0,10)], axis=0), dtype=np.int)
init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                           for j in np.arange(10252,10262)], axis=0), dtype=np.int)
# VB = ViBeSegmentation(n=2, init_image=init, n_min=2, r=25, phi=1)
# n_multi = 2
# width_multi = int(init.shape[1]/n_multi)
# print(width_multi)
# VBs = []
# for i in range(n_multi):
#     VBs.append(ViBeSegmentation(n=3, init_image=init[:,i*width_multi:(i+1)*width_multi], n_min=3, r=40, phi=1))
VB = ViBeSegmentation(n=3, init_image=init, n_min=3, r=40, phi=1)
print("Debug")
# def seg(arg):
#     i, image = arg
#     return VBs[i].detect(image[:,i*width_multi:(i+1)*width_multi], do_neighbours=False)
# from multiprocessing import Pool
# segmentation_pool = Pool(n_multi)
# detection_pool = Pool(n_multi)
for i in range(10262,10272):
    img = db.getImage(frame=i).data
    # segmentation_pool.map(seg, [[j, img] for j in range(n_multi)])
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
    PT_Detection_Type = db.getMarkerType(name="PT_Detection_Marker")
else:
    PT_Detection_Type = db.setMarkerType(name="PT_Detection_Marker", color="#FFFF00", style='{"scale":0.8}')
if db.getMarkerType(name="PT_Track_Marker"):
    PT_Track_Type = db.getMarkerType(name="PT_Track_Marker")
else:
    PT_Track_Type = db.setMarkerType(name="PT_Track_Marker", color="#00FF00", mode=db.TYPE_Track)
if db.getMarkerType(name="PT_Prediction_Marker"):
    PT_Prediction_Type = db.getMarkerType(name= "PT_Prediction_Marker")
else:
    PT_Prediction_Type = db.setMarkerType(name="PT_Prediction_Marker", color="#0000FF")
if db.getMarkerType(name="PT_Stitch_Marker"):
    PT_Stitch_Type = db.getMarkerType(name="PT_Stitch_Marker")
else:
    PT_Stitch_Type = db.setMarkerType(name="PT_Stitch_Marker", color="#FF8800", mode=db.TYPE_Track)

# Delete Old Tracks
db.deleteMarkers(type=PT_Detection_Type)
db.deleteMarkers(type=PT_Track_Type)
db.deleteMarkers(type=PT_Prediction_Type)
db.deleteMarkers(type=PT_Stitch_Type)

db.deleteTracks(type=PT_Track_Type)
db.deleteTracks(type=PT_Stitch_Type)



# Start Iteration over Images
print('Starting Iteration')
# images = db.getImageIterator(start_frame=20, end_frame=30)#start_frame=start_frame, end_frame=3)
images = db.getImageIterator(start_frame=10272, end_frame=10311)#start_frame=start_frame, end_frame=3)
# images = db.getImageIterator(start_frame=10272, end_frame=10279)#start_frame=start_frame, end_frame=3)

from multiprocessing import Process,Queue,Pipe

segmentation_queue = Queue(10)
SegMap_write_queue = Queue()
Detection_write_queue = Queue()
Track_write_queue = Queue()

# segmentation_pipe_in, segmentation_pipe_out = Pipe()
detection_pipe_in, detection_pipe_out = Pipe()
tracking_pipe_in, tracking_pipe_out = Pipe()
# writing_pipe_in, writing_pipe_out = Pipe()

Timer_in = Queue()
Timer_out = Queue()


def load(images):
    # images = args
    for i, img in enumerate(images):
        # while True:
        #     if not segmentation_pipe_out.poll():
        #         segmentation_pipe_in.send([i, img.data])
        #         break
        segmentation_queue.put([img.sort_index, img.data])
        Timer_in.put([img.sort_index, time()])
        print("loaded image %s" % img.sort_index)
        # while True:
        #     if not segmentation_queue.full():
        #         segmentation_queue.put()
        #         segmentation_queue.put([img.sort_index, img.data])
        #         Timer_in.put([img.sort_index, time()])
        #         print("loaded image %s"%img.sort_index)
        #         # print(Timer_in.keys())
        #         break


def segmentate():
    while True:
        i, img = segmentation_queue.get()
        print("starting Segmentation %s"%i)
        # i, img = segmentation_pipe_out.recv()
        #  SegMap = segmentation_pool.map(seg, [[j, img] for j in range(n_multi)])
        #  SegMap = np.hstack(SegMap)
        SegMap = VB.detect(img, do_neighbours=False)
        # SegMap_write_queue.put([i, SegMap])
        detection_pipe_in.send([i, SegMap])
        print("Segmentated Image %s"%i)


        # if segmentation_queue.empty() and not loading.is_alive():
        #     break


def detect():
    while True:
        # i, SegMap = detection_queue.get()
        i, SegMap = detection_pipe_out.recv()
        Positions = AD.detect(SegMap)
        X = np.asarray([[pos.PositionX, pos.PositionY] for pos in Positions])
        Positions = [pos for pos in Positions if np.sum(((pos.PositionX-X.T[0])**2+(pos.PositionY-X.T[1])**2)**0.5 < 200) < 10]
        # Detection_write_queue.put([i, Positions])
        tracking_pipe_in.send([i, Positions])
        print("Found %s animals in %s!"%(len(Positions), i))

        # if not detection_pipe_out.poll(1) and not segmentation.is_alive():
        #     break


def track():
    # i, Positions = tracking_queue.get()
    while True:
        i, Positions = tracking_pipe_out.recv()
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
        if len(Positions) > 0:
            print("Tracking %s"%i)
            # Update Filter with new Detections
            MultiKal.update(z=Positions, i=i)
            Track_write_queue.put([i, MultiKal.ActiveFilters])
        else:
            Track_write_queue.put([i,{}])

                # writing_pipe_in.send([i, MultiKal.ActiveFilters.keys()])
        print("Got %s Filters in frame %s" % (len(MultiKal.ActiveFilters.keys()), i))
        # if not tracking_pipe_in.poll(1) and not detection.is_alive():
        #     break


# def mask_writing():
#     while True:
#         try:
#             with db.db.atomic() as transaction:
#                 while not SegMap_write_queue.empty():
#                     i, Mask = SegMap_write_queue.get()
#                     db.setMask(frame=i, data=(PT_Mask_Type.index*(~Mask).astype(np.uint8)))
#                     print("Masks set! %s"%i)
#         except peewee.OperationalError:
#             pass
#         # if SegMap_write_queue.empty() and not segmentation.is_alive():
#         #     break
#
#
# def detection_writing():
#     while True:
#         try:
#             with db.db.atomic() as transaction:
#                 while not Detection_write_queue.empty():
#                     i, Positions = Detection_write_queue.get()
#                     for pos in Positions:
#                         detection_marker = db.setMarker(frame=i,
#                                                 x=pos.PositionY, y=pos.PositionX,
#                                                 text="Detection  %.2f" % (pos.Log_Probability),
#                                                 type=PT_Detection_Type)
#                         db.setMeasurement(marker=detection_marker, log=pos.Log_Probability, x=pos.PositionX, y=pos.PositionY)
#                     print("Detections written! %s"%i)
#         except peewee.OperationalError:
#             pass
#         # if Detection_write_queue.empty() and not detection.is_alive():
#         #     break
#
#
# def track_writing():
#     while True:
#         try:
#             with db.db.atomic() as transaction:
#             # for kkk in range(1):
#                 while not Track_write_queue.empty():
#                 # for kk in range(1):
#                     i, ActiveFilters = Track_write_queue.get()
#                     for k in ActiveFilters:
#                         if not db.getTrack(k+100):
#                             track = db.setTrack(type=PT_Track_Type, id=100+k)
#                         else:
#                             track = db.getTrack(id=100+k)
#                         if ActiveFilters[k].Measurements.has_key(i):
#                             meas = ActiveFilters[k].Measurements[i]
#                             x = meas.PositionX
#                             y = meas.PositionY
#                             prob = ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
#                             db.setMarker(frame=i, x=y, y=x,
#                                          track=track,
#                                          text="Track %s, Prob %.2f" % (k, prob),
#                                          type=PT_Track_Type)
#                         if ActiveFilters[k].Predicted_X.has_key(i):
#                             pred_x, pred_y = MultiKal.Model.measure(ActiveFilters[k].Predicted_X[i])
#                             db.setMarker(frame=i, x=pred_y, y=pred_x,
#                                          text="Prediction %s" % (k),
#                                          type=PT_Prediction_Type)
#                     Timer_out.put([i, time()])
#                     print("Tracks written! %s"%i)
#         except peewee.OperationalError:
#             pass
        # if Track_write_queue.empty() and not tracking.is_alive():
        #     break
def DB_write():
    while True:
        try:
            with db.db.atomic() as transaction:
                while not SegMap_write_queue.empty():
                    i, Mask = SegMap_write_queue.get()
                    db.setMask(frame=i, data=(PT_Mask_Type.index * (~Mask).astype(np.uint8)))
                    print("Masks set! %s" % i)
                while not Detection_write_queue.empty():
                    i, Positions = Detection_write_queue.get()
                    for pos in Positions:
                        detection_marker = db.setMarker(frame=i,
                                                x=pos.PositionY, y=pos.PositionX,
                                                text="Detection  %.2f" % (pos.Log_Probability),
                                                type=PT_Detection_Type)
                        db.setMeasurement(marker=detection_marker, log=pos.Log_Probability, x=pos.PositionX, y=pos.PositionY)
                    print("Detections written! %s"%i)
                while not Track_write_queue.empty():
                    i, ActiveFilters = Track_write_queue.get()
                    for k in ActiveFilters:
                        if not db.getTrack(k+100):
                            track = db.setTrack(type=PT_Track_Type, id=100+k)
                        else:
                            track = db.getTrack(id=100+k)

                        if ActiveFilters[k].Measurements.has_key(i):
                            meas = ActiveFilters[k].Measurements[i]
                            x = meas.PositionX
                            y = meas.PositionY
                            prob = ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
                            db.setMarker(frame=i, x=y, y=x,
                                         track=track,
                                         text="Track %s, Prob %.2f" % (k, prob),
                                         type=PT_Track_Type)
                        if ActiveFilters[k].Predicted_X.has_key(i):
                            pred_x, pred_y = MultiKal.Model.measure(ActiveFilters[k].Predicted_X[i])
                            db.setMarker(frame=i, x=pred_y, y=pred_x,
                                         text="Prediction %s" % (k),
                                         type=PT_Prediction_Type)
                    Timer_out.put([i, time()])
                    print("Tracks written! %s"%i)

        except peewee.OperationalError:
            pass


loading = Process(target=load, args=(images,))
segmentation = Process(target=segmentate)
detection = Process(target=detect)
tracking = Process(target=track)
# writing_SegMaps = Process(target=mask_writing)
# writing_Detections = Process(target=detection_writing)
# writing_Tracks = Process(target=track_writing)
writind_DB = Process(target=DB_write)

loading.start()
segmentation.start()
detection.start()
tracking.start()
# writing_SegMaps.start()
# writing_Detections.start()
# writing_Tracks.start()
writind_DB.start()

times1 = {}
times2 = {}
start = time()
while True:
    if not Timer_in.empty():
        i, value = Timer_in.get()
        times1.update({i: value})

    if not Timer_out.empty():
        i, value = Timer_out.get()
        times2.update({i: value})
        for i in times2:
            if times1.has_key(i):
                print("Time for image %s is %s seconds!"%(i, times2[i]-times1[i]))
            else:
                print("DAMN!!")

    if np.all([times2.has_key(k) for k in range(10272, 10311)]):
        full_time=time()-start
        print(full_time/len(times2.keys()))
        break

print(times1.keys())
print(times2.keys())

loading.terminate()
segmentation.terminate()
detection.terminate()
tracking.terminate()
# writing_SegMaps.terminate()
# writing_Detections.terminate()
# writing_Tracks.terminate()
writind_DB.terminate()


# while True:
    # print(np.sum([segmentation_queue.empty(),
    #       SegMap_write_queue.empty(),
    #       Detection_write_queue.empty(),
    #       Track_write_queue.empty(),
    #       not detection_pipe_in.poll(),
    #       not detection_pipe_out.poll(),
    #       not tracking_pipe_in.poll(),
    #       not tracking_pipe_out.poll()]) < 8)
    # pass
# loading.terminate()
# segmentation.terminate()
# detection.terminate()
# detection.terminate()
# tracking.terminate()
# writing_SegMaps.terminate()
# writing_Detections.terminate()
# writing_Tracks.terminate()

    # print(segmentation_queue.empty(),
    #       SegMap_write_queue.empty(),
    #       Detection_write_queue.empty(),
    #       Track_write_queue.empty(),
    #       not detection_pipe_in.poll(),
    #       not detection_pipe_out.poll(),
    #       not tracking_pipe_in.poll(),
    #       not tracking_pipe_out.poll())
    # print(loading.is_alive(),
    #       segmentation.is_alive(),
    #       detection.is_alive(),
    #       tracking.is_alive(),
    #       writing_SegMaps.is_alive(),
    #       writing_Detections.is_alive(),
    #       writing_Tracks.is_alive())
    # print("---------%s-----------"%np.sum([segmentation_queue.empty(),
    #       SegMap_write_queue.empty(),
    #       Detection_write_queue.empty(),
    #       Track_write_queue.empty(),
    #       not detection_pipe_in.poll(),
    #       not detection_pipe_out.poll(),
    #       not tracking_pipe_in.poll(),
    #       not tracking_pipe_out.poll()]))
    #
    # pass
# loading.join()
# segmentation.join()
# detection.join()
# tracking.join()

#
# start = time()
# while True:
#     i, keys = writing_pipe_out.recv()
#     print("Done with %s in %s s"%(i, time()-start))
#     start = time()
# start = time()
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

