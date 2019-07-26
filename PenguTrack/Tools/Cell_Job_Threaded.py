from __future__ import division, print_function

import numpy as np
import peewee

import os
import sys
import time

from qtpy import QtGui, QtCore, QtWidgets
from qimage2ndarray import array2qimage


input_file = "/home/alex/2017-03-10_Tzellen_microwells_bestdata/30sec/max_Proj.cdb"
input_file2 = "/home/alex/2017-03-10_Tzellen_microwells_bestdata/30sec/min_Indizes.cdb"


def int8(input):
    return np.asarray(input,ndmin=2,dtype=np.uint8,copy=True)


# seg_cam = GigECam("/home/birdflight/birdflight/src/python/cfg/segmem.xml")
print("Initialized Camera")

# Import PenguTrack
import clickpoints
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import RandomWalk
from PenguTrack.Detectors import rgb2gray, TresholdSegmentation
from PenguTrack.Detectors import RegionPropDetector, RegionFilter, ExtendedRegionProps

from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
from skimage.morphology import disk
SELEM = disk(2,dtype=bool)

import scipy.stats as ss

# Load Database
a_min=75
a_max=np.inf
file_path = "/home/alex/Desktop/PT_Cell_T850_A%s_%s_3d.cdb"%(a_min,a_max)

global db
db = DataFileExtended(file_path,"w")

db_start = DataFileExtended(input_file)
db_start2 = DataFileExtended(input_file2)
images = db_start.getImageIterator()

def getImage():
    try:
        im = images.next()
    except StopIteration:
        # print("Done! First!")
        return None, None, None
    fname = im.filename
    d = im.timestamp
    print(fname)
    print(fname.replace("MaxProj","MinIndices289"))
    try:
        indizes = db_start2.getImages(filename=fname.replace("MaxProj","MinIndices289"))[0]
    except IndexError:
        try:
            indizes = db_start2.getImages(filename=fname.replace("MaxProj", "MinIndices290"))[0]
        except IndexError:
            try:
                indizes = db_start2.getImages(filename=fname.replace("MaxProj", "MinIndices288"))[0]
            except IndexError:
                indizes = db_start2.getImages(filename=fname.replace("MaxProj", "MinIndices2"))[0]
    time_unix = np.uint32(time.mktime(d.timetuple()))
    time_ms = 0
    meta = {'time': time_unix,
            'time_ms': time_ms,
            'file_name': fname,
            'path': im.path.path}
    return im.data,indizes.data, meta

# Tracking Parameters
q = 3
r = 1

object_size=20

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = RandomWalk(dim=3)

X = np.zeros(6).T  # Initial Value for Position
Q = np.diag([q*object_size,q*object_size, q*object_size])  # Prediction uncertainty
R = np.diag([r*object_size, r*object_size, r*object_size])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                       np.diag(R))#, meas_dist=Meas_Dist, state_dist=State_Dist)
MultiKal.LogProbabilityThreshold = -1000.
MultiKal.FilterThreshold = 2
MultiKal.MeasurementProbabilityThreshold = 0.
MultiKal.AssignmentProbabilityThreshold = 0.


VB = TresholdSegmentation(850)

import matplotlib.pyplot as plt

# Initialize Detector
print('Initialized Tracker')
# AD = AreaDetector(object_area, object_number, upper_limit=10, lower_limit=0)
from PenguTrack.Detectors import RegionFilter, RegionPropDetector

rf = RegionFilter("area",200,var=108.**2, lower_limit=a_min, upper_limit=a_max)
# rf = RegionFilter("area",200,var=40.**2, upper_limit=300)
# rf2 = RegionFilter("solidity",0.98,var=0.04**2)#, lower_limit=0.8)
# rf3 = RegionFilter("eccentricity",0.51,var=0.31**2)#, upper_limit=0.95)
# rf4 = RegionFilter("extent",0.66,var=0.07**2)#, lower_limit=0.5, upper_limit=0.9)
# rf5 = RegionFilter("InOutContrast2", 0.89, var=0.13**2)#, lower_limit=0.9)
# rf6 = RegionFilter("mean_intensity", 60., var=17.**2)#, lower_limit=25.)
# rf7 = RegionFilter("max_intensity", 124, var=56)#, lower_limit=40)
# rf8 = RegionFilter("min_intensity", 21, var=14)#, lower_limit=0, upper_limit=70)

AD = RegionPropDetector([rf])#,rf2,rf3,rf5,rf6])


print("--------------------------")
print("Parameters")
print("--------------------------")
print(VB.__dict__)
print(AD.__dict__)
print([f.__dict__ for f in AD.Filters])
print(MultiKal.__dict__)
print(MultiKal.Model.__dict__)
print("--------------------------")
print("--------------------------")

# Set Mask Type
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

from multiprocessing import Process,Queue,Pipe

# segmentation_queue = Queue(10)
Image_write_queue = Queue()
SegMap_write_queue = Queue()
Detection_write_queue = Queue()
Track_write_queue = Queue()

segmentation_pipe_in, segmentation_pipe_out = Pipe()
detection_pipe_in, detection_pipe_out = Pipe()
tracking_pipe_in, tracking_pipe_out = Pipe()

Timer_in = Queue()
Timer_out = Queue()

from datetime import datetime, timedelta

# TODO: wtf is this "Exception RuntimeError: RuntimeError('main thread is not in main loop',) in <bound method PhotoImage.__del__ of <Tkinter.PhotoImage instance at 0x7f516ca8f830>> ignored
# Tcl_AsyncDelete: async handler deleted by the wrong thread


def load(load_cam):
    i = 1
    while True:
        img, index_img, meta = getImage()
        if img is not None:
            timestamp = datetime.fromtimestamp(meta["time"])
            timestamp += timedelta(milliseconds=int(meta["time_ms"]))
            segmentation_pipe_in.send([i, img, index_img])
            Image_write_queue.put([timestamp, i, meta])
            Timer_in.put([i, time.time()])
            print("loaded image %s" % i)
            # print(meta)
            i+=1
        # else:
        #     Image_write_queue.put([None,None,None])
        #     segmentation_pipe_in.send([None,None])
        #     Timer_in.put([None,None])
        #     break
        if img is None and meta is None:
            break
    print("Done!Loading!")


def segmentate():
    LastMap = None
    while True:
        i, img, index_image = segmentation_pipe_out.recv()
        # if i is None and img is None:
        #     SegMap_write_queue.put([None, None])
        #     detection_pipe_in.send([None,None,None])
        #     break
        print("starting Segmentation %s"%i)
        SegMap = VB.segmentate(img)
        SegMap = binary_opening(SegMap)
        SegMap_write_queue.put([i, SegMap])
        detection_pipe_in.send([i, SegMap, img, index_image])
        print("Segmentated Image %s"%i)
    print("Done Segmenation!")


def detect():
    Map = None
    while True:
        i, SegMap, img, index_image = detection_pipe_out.recv()
        # if i is None and SegMap is None and img is None:
        #     Detection_write_queue.put([None,None])
        #     tracking_pipe_in.send([None,None])
        #     break
        combi = np.zeros_like(index_image)
        combi[SegMap] = index_image[SegMap]
        from skimage.measure import label
        combi = np.sum(label(np.asarray([combi==z+1 for z in range(int(np.amax(combi)))], dtype=bool)), axis=0)
        # plt.ioff()
        # plt.imshow(combi)
        # plt.show()
        Positions = AD.detect(combi)#, intensity_image=img)
        New_Positions = []
        from PenguTrack.Detectors import Measurement as PT_Measurement
        for pos in Positions:
            z = index_image[int(pos.PositionX), int(pos.PositionY)]
            New_Positions.append(PT_Measurement(pos.Log_Probability, [pos.PositionX,pos.PositionY, z*13/0.645],
                                                data=pos.Data, frame=pos.Frame, track_id=pos.Track_Id))
        # for pos in Positions:
        #     pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY/float(VB.SubSampling)
        #                                                       , pos.PositionX/float(VB.SubSampling)])
        #     pos.PositionX *= (VB.Max_Dist / VB.height)
        #     pos.PositionY *= (VB.Max_Dist / VB.height)
        Detection_write_queue.put([i, Positions])
        tracking_pipe_in.send([i, New_Positions])
        print("Found %s animals in %s!"%(len(Positions), i))
    print("DoneDetection!")

        # if not detection_pipe_out.poll(1) and not segmentation.is_alive():
        #     break


def track():
    while True:
        try:
            with np.errstate(all="raise"):
                i, Positions = tracking_pipe_out.recv()
                # if i is None and Positions is None:
                #     Track_write_queue.put([None,None])
                #     raise StopIteration
                MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
                if len(Positions) > 2:
                    print("Tracking %s" % i)
                    # Update Filter with new Detections
                    MultiKal.update(z=Positions, i=i)
                    # Track_write_queue.put([i, dict(MultiKal.ActiveFilters)])
                    filters = MultiKal.ActiveFilters
                    Track_write_queue.put([i, dict([[k, [filters[k].X.get(i, None),
                                                         filters[k].Predicted_X.get(i, None),
                                                         filters[k].Measurements.get(i, None),
                                                         filters[k].log_prob(keys=[i])]] for k in filters])])
                else:
                    print("empty track at %s"%i)
                    # Track_write_queue.put([i, {}])
                print("Got %s Filters in frame %s" % (len(MultiKal.ActiveFilters.keys()), i))
        except StopIteration:
            break
    print("Done Tracking!")


def dummy_write():
    while True:
        if not SegMap_write_queue.empty():
            i, Mask = SegMap_write_queue.get()
        if not Detection_write_queue.empty():
            i, Positions = Detection_write_queue.get_nowait()
        if not Track_write_queue.empty():
            i, ActiveFilters = Track_write_queue.get_nowait()
            Timer_out.put([i,time.time()])


def DB_write():
    image_dict = {}
    while True:
        try:
            while not Image_write_queue.empty():
                timestamp, i, meta  = Image_write_queue.get()
                # if timestamp is None and i is None and meta is None:
                #     # raise StopIteration
                #     break
                fname = meta['file_name']
                if db.getPath(meta["path"]) is None:
                    path = db.setPath(meta["path"])
                else:
                    path = db.getPath(meta["path"])
                image = db.setImage(filename=fname, path=path, timestamp=timestamp)
                image_dict.update({i: np.copy(image.id)})
                image.save()
                print("Saved image %s!"%i)
            with db.db.atomic() as transaction:
                while not SegMap_write_queue.empty():
                    i, Mask = SegMap_write_queue.get()
                    # if i is None and Mask is None:
                    #     # raise StopIteration
                    #     break
                    if not image_dict.has_key(i):
                        SegMap_write_queue.put([i, Mask])
                        break
                    # db.getImage(id=i)
                    try:
                        db.setMask(image=db.getImage(id=image_dict[i]), data=(PT_Mask_Type.index * (~Mask).astype(np.uint8)))
                    except clickpoints.MaskDimensionMismatch:
                        print(Mask.shape)
                        print(db.getImages()[0].getShape())
                        raise

                    print("Masks set! %s" % i)
                while not Detection_write_queue.empty():
                    i, Positions = Detection_write_queue.get()
                    # if i is None and Positions is None:
                    #     # raise StopIteration
                    #     break
                    if not image_dict.has_key(i):
                        Detection_write_queue.put([i,Positions])
                        break
                    for pos in Positions:
                        while True:
                            if image_dict.has_key(i):
                                break
                        x = pos.PositionX
                        y = pos.PositionY
                        # x_px = x * (VB.height / VB.Max_Dist)
                        # y_px = y * (VB.height / VB.Max_Dist)
                        # x_det, y_det = VB.orth_to_log([y_px, x_px])
                        # x_det *= VB.SubSampling
                        # y_det *= VB.SubSampling
                        detection_marker = db.setMarker(image=db.getImage(id=image_dict[i]),
                                                x=y, y=x,
                                                text="Detection  %.2f \n %s" % (pos.Log_Probability, "\n".join(["%s \t %s"%(k, pos.Data[k]) for k in pos.Data])),
                                                type=PT_Detection_Type)
                        db.setMeasurement(marker=detection_marker, log=pos.Log_Probability, x=pos.PositionX, y=pos.PositionY)
                    print("Detections written! %s"%i)
                while not Track_write_queue.empty():
                    i, X_P = Track_write_queue.get()
                    # if i is None and ActiveFilters is None:
                    #     Timer_out.put([None,None])
                    #     # break
                    #     raise StopIteration
                    if not image_dict.has_key(i):
                        Track_write_queue.put([i, X_P])
                        break
                    for k in X_P:
                        X, Prediction, Measurement, LogProb = X_P[k]
                        if not db.getTrack(id=k+100):
                            track = db.setTrack(type=PT_Track_Type, id=100+k)
                        else:
                            track = db.getTrack(id=100+k)
                        if Measurement is not None:
                            meas = Measurement
                            x = meas.PositionX
                            y = meas.PositionY
                            z = meas.PositionZ
                            # x_px = x * (VB.height / VB.Max_Dist)
                            # y_px = y * (VB.height / VB.Max_Dist)
                            # x_img, y_img = VB.warp_orth([VB.Res * (y_px - VB.width / 2.), VB.Res * (VB.height - x_px)])
                            # prob = ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
                            prob = LogProb#ActiveFilters[k].log_prob(keys=[i])
                            marker = db.setMarker(image=db.getImage(id=image_dict[i]), x=y, y=x,
                                         track=track,
                                         text="Track %s, Prob %.2f" % (k, prob),
                                         type=PT_Track_Type)
                            db.setMeasurement(marker, log=prob, x=y, y=x, z=z)
                        if Prediction is not None:
                            pred_x, pred_y, pred_z = MultiKal.Model.measure(Prediction)
                            # pred_x_px = pred_x *(VB.height / VB.Max_Dist)
                            # pred_y_px = pred_y *(VB.height / VB.Max_Dist)
                            # pred_x_img, pred_y_img = VB.warp_orth([VB.Res * (pred_y_px - VB.width / 2.), VB.Res * (VB.height - pred_x_px)])
                            db.setMarker(image=db.getImage(id=image_dict[i]), x=pred_x, y=pred_y,
                                         text="Prediction %s" % (k),
                                         type=PT_Prediction_Type)
                    Timer_out.put([i, time.time()])
                    print("Tracks written! %s"%i)

                # for track in db.getTracks(type=PT_Track_Type):+
        except peewee.OperationalError:
            pass
        except StopIteration:
        #     print("blab")
            break
    print("Done!Writing!")


loading = Process(target=load, args=(None,))
segmentation = Process(target=segmentate)
detection = Process(target=detect)
tracking = Process(target=track)
writing_DB = Process(target=DB_write)

loading.start()
segmentation.start()
detection.start()
tracking.start()
writing_DB.start()

times1 = {}
times2 = {}
start = time.time()
while True:
    try:
        if not Timer_in.empty():
            i, value = Timer_in.get()
            if i is None and value is None:
                pass
            else:
                times1.update({i: value})

        if not Timer_out.empty():
            i, value = Timer_out.get()
            if i is None and value is None:
                print("DoneTime!")
                # raise StopIteration
            times2.update({i: value})
            for i in times2:
                if times1.has_key(i):
                    pass
                else:
                    print("DAMN!!")
            print("Needed %s s per frame"%((time.time()-start)/len(times2.keys())))
        # print("......")
        # print(loading.is_alive())
        # print(segmentation.is_alive())
        # print(detection.is_alive())
        # print(tracking.is_alive())
        # print(writing_DB.is_alive())
        # if not tracking.is_alive():
        #     raise StopIteration
        # print("......")
        # time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        loading.terminate()
        segmentation.terminate()
        detection.terminate()
        tracking.terminate()
        writing_DB.terminate()
        raise
    except StopIteration:
        # loading.terminate()
        # segmentation.terminate()
        # detection.terminate()
        # tracking.terminate()
        # writing_DB.terminate()
        break
