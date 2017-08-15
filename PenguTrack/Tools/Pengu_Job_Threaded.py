from __future__ import division, print_function

import numpy as np
import peewee

import os
import sys
import time

from qtpy import QtGui, QtCore, QtWidgets
from qimage2ndarray import array2qimage


input_file = "/home/user/Desktop/252/252Horizon.cdb"


def int8(input):
    return np.array(input,ndmin=2,dtype=np.uint8,copy=True)


# seg_cam = GigECam("/home/birdflight/birdflight/src/python/cfg/segmem.xml")
print("Initialized Camera")

# Import PenguTrack
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
# from PenguTrack.Detectors import ViBeSegmentation
# from PenguTrack.Detectors import AlexSegmentation as ViBeSegmentation
# from PenguTrack.Detectors import DumbViBeSegmentation as ViBeSegmentation
from PenguTrack.Detectors import SiAdViBeSegmentation, rgb2gray
# from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector

from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
from skimage.morphology import disk
SELEM = disk(2,dtype=bool)

import scipy.stats as ss

# Load Database
# file_path = "/home/birdflight/Desktop/PT_Test.cdb"
file_path = "/home/user/Desktop/PT_Test_full_n3_r7_A20.cdb"
c=20
# file_path = "/mnt/mmap/PT_Test3.cdb"
# file_path = "/mnt/mmap/PT_Test4.cdb"

global db
db = DataFileExtended(file_path,"w")

db_start = DataFileExtended(input_file)
# images = db_start.getImageIterator(start_frame=2490-30, end_frame=2600)
# images = db_start.getImageIterator(start_frame=1936-210, end_frame=2600)
# images = db_start.getImageIterator(end_frame=52)
# images = db_start.getImageIterator(start_frame=1936-20-90, end_frame=2600)
# images = db_start.getImageIterator(start_frame=1500, end_frame=2600)

images = db_start.getImageIterator()
def getImage():
    try:
        im = images.next()
    except StopIteration:
        print("Done! First!")
        return None, None
    # im.data = rgb2gray(im.data)
    fname = im.filename
    from datetime import datetime
    # d = datetime.strptime(fname[0:15], '%Y%m%d-%H%M%S')
    d = im.timestamp
    time_unix = np.uint32(time.mktime(d.timetuple()))
    time_ms = 0#np.uint32(fname[16:17])*100
    meta = {'time': time_unix,
            'time_ms': time_ms,
            'file_name': fname,
            'path': im.path.path,
            'offset': [im.offset.x, im.offset.y]}
    return rgb2gray(im.data), meta

# Tracking Parameters
q = 3
r = 1

# Initialise PenguTrack
object_size = 3  # Object diameter (smallest)
object_number = 1  # Number of Objects in First Track
object_area = 3
penguin_height = 0.462#0.575
penguin_width = 0.21

# Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
model = VariableSpeed(1, 1, dim=2, timeconst=1.)

X = np.zeros(4).T  # Initial Value for Position
Q = np.diag([q*object_size, q*object_size])  # Prediction uncertainty
R = np.diag([r*object_size, r*object_size])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

# Initialize Filter
MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                       np.diag(R))#, meas_dist=Meas_Dist, state_dist=State_Dist)
# MultiKal.LogProbabilityThreshold = -10000.
MultiKal.LogProbabilityThreshold = -21.
MultiKal.FilterThreshold = 2
MultiKal.MeasurementProbabilityThreshold = 0.
MultiKal.AssignmentProbabilityThreshold = 0.

# Init_Background from Image_Median
# Initialize segmentation with init_image and start updating the first 10 frames.
init_buffer = []
for i in range(10):
    while True:
        img, meta = getImage()
        if img is not None:
            print("Got img from cam")
            init_buffer.append(img)
            print(init_buffer[-1].shape)
            print(init_buffer[-1].dtype)
            break

init = np.array(np.median(init_buffer, axis=0))


# Load horizon-markers
horizont_type = db_start.getMarkerType(name="Horizon")
try:
    horizon_markers = np.array([[m.x, m.y] for m in db_start.getMarkers(type=horizont_type)]).T
except ValueError:
    raise ValueError("No markers with name 'Horizon'!")

# Load penguin-markers
penguin_type = db_start.getMarkerType(name="Penguin_Size")
try:
    penguin_markers = np.array([[m.x1, m.y1, m.x2, m.y2] for m in db_start.getLines(type="Penguin_Size")]).T
except ValueError:
    raise ValueError("No markers with name 'Horizon'!")

VB = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3, 9e-3], penguin_markers, penguin_height, 500, n=3, init_image=init, n_min=3, r=7, phi=1, subsampling=2)#, camera_h=44.)

del init_buffer

for i in range(10):
    while True:
        img, meta = getImage()
        if img is not None:
            print("Got img from cam")
            mask = VB.segmentate(img, do_neighbours=False)
            VB.update(mask, img, do_neighbours=False)
            break

import matplotlib.pyplot as plt

# Initialize Detector
print('Initialized Tracker')
# AD = AreaDetector(object_area, object_number, upper_limit=10, lower_limit=0)
from PenguTrack.Detectors import RegionFilter, RegionPropDetector

rf = RegionFilter("area",200,var=40.**2, lower_limit=c, upper_limit=300)
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
        # try:
        img, meta = getImage()
        # if img is None and meta is None:
        #     print("Done! Second!")
        #     break
        # except StopIteration:
            # break
        if img is not None:
            timestamp = datetime.fromtimestamp(meta["time"])
            timestamp += timedelta(milliseconds=int(meta["time_ms"]))
            offset = meta['offset']
            segmentation_pipe_in.send([i, img, offset])
            Image_write_queue.put([timestamp, i, meta])
            Timer_in.put([i, time.time()])
            print("loaded image %s" % i)
            # print(meta)
            i+=1
        else:
            Image_write_queue.put([None,None,None])
            segmentation_pipe_in.send([None,None])
            Timer_in.put([None,None])
            break
    print("Done!Loading!")


def segmentate():
    LastMap = None
    while True:
        i, img, offset = segmentation_pipe_out.recv()
        if i is None and img is None:
            SegMap_write_queue.put([None, None])
            detection_pipe_in.send([None,None,None])
            break
        print("starting Segmentation %s"%i)
        SegMap = VB.segmentate(img, do_neighbours=False)
        VB.update(SegMap, img, do_neighbours=False)
        SegMap = binary_opening(SegMap)
        SegMap_write_queue.put([i, SegMap])
        detection_pipe_in.send([i, SegMap, VB.horizontal_equalisation(img)])
        print("Segmentated Image %s"%i)
    print("Done Segmenation!")


def detect():
    Map = None
    while True:
        i, SegMap, img = detection_pipe_out.recv()
        if i is None and SegMap is None and img is None:
            Detection_write_queue.put([None,None])
            tracking_pipe_in.send([None,None])
            break
        Positions = AD.detect(SegMap, intensity_image=img)
        for pos in Positions:
            pos.PositionY, pos.PositionX = VB.log_to_orth([pos.PositionY/float(VB.SubSampling)
                                                              , pos.PositionX/float(VB.SubSampling)])
            pos.PositionX *= (VB.Max_Dist / VB.height)
            pos.PositionY *= (VB.Max_Dist / VB.height)
        Detection_write_queue.put([i, Positions])
        tracking_pipe_in.send([i, Positions])
        print("Found %s animals in %s!"%(len(Positions), i))
    print("DoneDetection!")

        # if not detection_pipe_out.poll(1) and not segmentation.is_alive():
        #     break


def track():
    while True:
        try:
            with np.errstate(all="raise"):
                i, Positions = tracking_pipe_out.recv()
                if i is None and Positions is None:
                    Track_write_queue.put([None,None])
                    raise StopIteration
                MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
                if len(Positions) > 2:
                    print("Tracking %s" % i)
                    # Update Filter with new Detections
                    MultiKal.update(z=Positions, i=i)
                    Track_write_queue.put([i, MultiKal.ActiveFilters])
                else:
                    print("empty track at %s"%i)
                    Track_write_queue.put([i, {}])
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
                if timestamp is None and i is None and meta is None:
                    # raise StopIteration
                    break
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
                    if i is None and Mask is None:
                        # raise StopIteration
                        break
                    if not image_dict.has_key(i):
                        SegMap_write_queue.put([i, Mask])
                        break
                    # db.getImage(id=i)
                    db.setMask(image=db.getImage(id=image_dict[i]), data=(PT_Mask_Type.index * (~Mask).astype(np.uint8)))
                    print("Masks set! %s" % i)
                while not Detection_write_queue.empty():
                    i, Positions = Detection_write_queue.get()
                    if i is None and Positions is None:
                        # raise StopIteration
                        break
                    if not image_dict.has_key(i):
                        Detection_write_queue.put([i,Positions])
                        break
                    for pos in Positions:
                        while True:
                            if image_dict.has_key(i):
                                break
                        x = pos.PositionX
                        y = pos.PositionY
                        x_px = x * (VB.height / VB.Max_Dist)
                        y_px = y * (VB.height / VB.Max_Dist)
                        x_det, y_det = VB.orth_to_log([y_px, x_px])
                        x_det *= VB.SubSampling
                        y_det *= VB.SubSampling
                        detection_marker = db.setMarker(image=db.getImage(id=image_dict[i]),
                                                x=x_det, y=y_det,
                                                text="Detection  %.2f \n %s" % (pos.Log_Probability, "\n".join(["%s \t %s"%(k, pos.Data[k]) for k in pos.Data])),
                                                type=PT_Detection_Type)
                        db.setMeasurement(marker=detection_marker, log=pos.Log_Probability, x=pos.PositionX, y=pos.PositionY)
                    print("Detections written! %s"%i)
                while not Track_write_queue.empty():
                    i, ActiveFilters = Track_write_queue.get()
                    if i is None and ActiveFilters is None:
                        Timer_out.put([None,None])
                        # break
                        raise StopIteration
                    if not image_dict.has_key(i):
                        Track_write_queue.put([i, ActiveFilters])
                        break
                    for k in ActiveFilters:
                        if not db.getTrack(id=k+100):
                            track = db.setTrack(type=PT_Track_Type, id=100+k)
                        else:
                            track = db.getTrack(id=100+k)
                        if ActiveFilters[k].Measurements.has_key(i):
                            meas = ActiveFilters[k].Measurements[i]
                            x = meas.PositionX
                            y = meas.PositionY
                            x_px = x * (VB.height / VB.Max_Dist)
                            y_px = y * (VB.height / VB.Max_Dist)
                            x_img, y_img = VB.warp_orth([VB.Res * (y_px - VB.width / 2.), VB.Res * (VB.height - x_px)])
                            # prob = ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
                            prob = ActiveFilters[k].log_prob(keys=[i])
                            db.setMarker(image=db.getImage(id=image_dict[i]), x=x_img, y=y_img,
                                         track=track,
                                         text="Track %s, Prob %.2f" % (k, prob),
                                         type=PT_Track_Type)
                        if ActiveFilters[k].Predicted_X.has_key(i):
                            pred_x, pred_y = MultiKal.Model.measure(ActiveFilters[k].Predicted_X[i])
                            pred_x_px = pred_x *(VB.height / VB.Max_Dist)
                            pred_y_px = pred_y *(VB.height / VB.Max_Dist)
                            pred_x_img, pred_y_img = VB.warp_orth([VB.Res * (pred_y_px - VB.width / 2.), VB.Res * (VB.height - pred_x_px)])
                            db.setMarker(image=db.getImage(id=image_dict[i]), x=pred_x_img, y=pred_y_img,
                                         text="Prediction %s" % (k),
                                         type=PT_Prediction_Type)
                    Timer_out.put([i, time.time()])
                    print("Tracks written! %s"%i)

                for track in db.getTracks(type=PT_Track_Type):
                    if len(track.markers)<3 and not ActiveFilters.has_key(track.id-100):
                        print("Deleting Track %s with length < 3"%(track.id-100))
                        db.deleteTracks(id=track.id)
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
