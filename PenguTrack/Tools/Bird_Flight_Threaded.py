from __future__ import division, print_function

import numpy as np
import peewee

import os
import sys
import time

from qtpy import QtGui, QtCore, QtWidgets
from qimage2ndarray import array2qimage

import imageio_plugin_VIS as vis   # import and register VIS plugin
import imageio
input_file = r'/mnt/131.188.117.98/data2/HGH/Data/2017/name_2017-02-09_15-50-48.db/2017-02-09_15-50-53.vis'

open .vis database with imageIO
reader = imageio.get_reader(input_file)
image_number_iterator = iter(range(700,reader.get_length()))
conv = vis.ConvertTo8Bit()

def getImageVIS():
    path = "/home/birdflight/Desktop/VIS/"
    from datetime import datetime, timedelta
    # from PIL import Image
    # from PenguTrack.Detectors import gray2rgb

    n = image_number_iterator.next()
    img = reader.get_data(n)
    meta = reader.get_meta_data(n)
    d = meta['time_date']
    time_unix = np.uint32(time.mktime(d.timetuple()))
    time_ms = np.uint32(meta['time_ms'])
    timestamp = datetime.fromtimestamp(time_unix)
    timestamp += timedelta(milliseconds=int(time_ms))
    fname = timestamp.strftime("%Y%m%d-%H%M%S") + "-%s" % int(timestamp.microsecond / 100000) + "_Fino4_SpynelS.jpg"

    img8 = conv.convertTo8bit(img, mode='percentile', percentile=[1, 99])
    path, fname = vis.SaveImageHGH(img8,meta,output_basepath=path,filetype="jpg")
    print(path, fname)

    # save_image = Image.fromarray(gray2rgb(img).astype(np.uint8))
    # save_image.save(path+fname,"tif")

    meta_out = {'time': time_unix,
            'time_ms': time_ms,
            'path': path,
            "file_name": fname}
    return img, meta_out


# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "qextendedgraphicsview"))
from include.QExtendedGraphicsView import QExtendedGraphicsView
from include.Tools import GraphicsItemEventFilter
from include.MemMap import MemMap

def int8(input):
    return np.array(input,ndmin=2,dtype=np.uint8,copy=True)

# camera class to retrieve newest image
class GigECam():
    def __init__(self,mmap_xml):
        self.mmap = MemMap(mmap_xml)
        self.counter_last = -1

    def getNewestImage(self):
        # get newest counter
        counters = [ slot.counter for slot in self.mmap.rbf]

        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)

        # return if there is no new one
        if counter_max == self.counter_last:
            # print("not new!")
            return None, None

        image = self.mmap.rbf[counter_max_idx].image

        # check for ROI
        try:
            im_channels = image.shape[2]
        except:
            im_channels = 1
        im_rsize = self.mmap.rbf[counter_max_idx].width * self.mmap.rbf[counter_max_idx].height * im_channels
        if not (im_rsize == len(image.flatten())):
            im_roi = image.flatten()[0: im_rsize ]

            image = im_roi.reshape([self.mmap.rbf[counter_max_idx].height,self.mmap.rbf[counter_max_idx].width,im_channels])

        meta = {'time':self.mmap.rbf[counter_max_idx].time_unix,
                'time_ms':self.mmap.rbf[counter_max_idx].time_ms}

        self.counter_last = counter_max

        return image, meta

cam = GigECam("/home/birdflight/birdflight/src/python/cfg/camera_SpynelS.xml")
# seg_cam = GigECam("/home/birdflight/birdflight/src/python/cfg/segmem.xml")
print("Initialized Camera")

# Import PenguTrack
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import VariableSpeed
# from PenguTrack.Detectors import ViBeSegmentation
# from PenguTrack.Detectors import AlexSegmentation as ViBeSegmentation
from PenguTrack.Detectors import DumbViBeSegmentation as ViBeSegmentation
# from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector

from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
from skimage.morphology import disk
SELEM = disk(2,dtype=bool)

import scipy.stats as ss

# Load Database
# file_path = "/home/birdflight/Desktop/PT_Test.cdb"
file_path = "/mnt/mmap/Starter_Full.cdb"
# file_path = "/mnt/mmap/PT_Test3.cdb"
# file_path = "/mnt/mmap/PT_Test4.cdb"

global db
db = DataFileExtended(file_path,"w")

db_start = DataFileExtended("/home/birdflight/Desktop/Starter.cdb")
# images = db_start.getImageIterator(start_frame=2490-30, end_frame=2600)
# images = db_start.getImageIterator(start_frame=1936-210, end_frame=2600)
images = db_start.getImageIterator(start_frame=700)
# images = db_start.getImageIterator(start_frame=1936-20-90, end_frame=2600)
# images = db_start.getImageIterator(start_frame=1500, end_frame=2600)

# images = db_start.getImageIterator()
def getImage():
    im = images.next()
    fname = im.filename
    from datetime import datetime
    d = datetime.strptime(fname[0:15], '%Y%m%d-%H%M%S')
    time_unix = np.uint32(time.mktime(d.timetuple()))
    time_ms = np.uint32(fname[16:17])*100
    meta = {'time': time_unix,
            'time_ms': time_ms,
            'file_name': fname,
            'path': im.path.path}
    return im.data, meta
cam.getNewestImage = getImage

# Tracking Parameters
q = 200
r = 100

# Initialise PenguTrack
object_size = 3  # Object diameter (smallest)
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
                       np.diag(R))#, meas_dist=Meas_Dist, state_dist=State_Dist)
# MultiKal.LogProbabilityThreshold = -10000.
MultiKal.LogProbabilityThreshold = -50.
MultiKal.FilterThreshold = 2
MultiKal.MeasurementProbabilityThreshold = 0.
MultiKal.AssignmentProbabilityThreshold = 0.

# Init_Background from Image_Median
# Initialize segmentation with init_image and start updating the first 10 frames.
init_buffer = []
for i in range(2):
    while True:
        img, meta = cam.getNewestImage()
        if img is not None:
            print("Got img from cam")
            init_buffer.append(img)
            print(init_buffer[-1].shape)
            print(init_buffer[-1].dtype)
            break
# init = np.array(np.median([init_buffer], axis=0))

NoMask = db_start.getMask(frame=0).data.astype(bool)

# VB = ViBeSegmentation(n=3, init_image=np.array(np.median(init_buffer, axis=0)), n_min=3, r=30 , phi=1) #Starter
# VB = ViBeSegmentation(n=3, init_image=np.array(np.median(init_buffer, axis=0)), n_min=3, r=30 , phi=1) #Starter2
# VB = ViBeSegmentation(n=8, init_image=np.array(np.median(init_buffer, axis=0)), n_min=8, r=50 , phi=1) #Starter3
# VB = ViBeSegmentation(n=3, init_image=np.array(np.median(init_buffer, axis=0)), n_min=3, r=75 , phi=1) #Starter4 - VIS

# VB.Samples[0] = np.amax(init_buffer, axis=0)
# VB.Samples[-1] = np.amin(init_buffer, axis=0)

# import  matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(np.amax(VB.Samples, axis=0))
# plt.figure()
# plt.imshow(np.amin(VB.Samples, axis=0))
# plt.show()
del init_buffer

for i in range(10):
    while True:
        img, meta = cam.getNewestImage()
        if img is not None:
            print("Got img from cam")
            mask = VB.segmentate(img, do_neighbours=False)
            VB.update(mask&(~NoMask), img, do_neighbours=False)
            break

import matplotlib.pyplot as plt

# for i in range(VB.N):
#     plt.figure()
#     plt.imshow(VB.Samples[i])
# plt.show()
#
# plt.figure()
# plt.imshow(np.amax(VB.Samples, axis=0))
# plt.figure()
# plt.imshow(np.amin(VB.Samples, axis=0))
# plt.show()
# Initialize Detector
print('Initialized Tracker')
# AD = AreaDetector(object_area, object_number, upper_limit=10, lower_limit=0)
from PenguTrack.Detectors import RegionFilter, RegionPropDetector
# rf = RegionFilter("area",12,var=9,lower_limit=0, upper_limit=50)
# rf2 = RegionFilter("solidity",0.97,var=0.06,lower_limit=0.7, upper_limit=np.nextafter(1.,2.))
# rf3 = RegionFilter("eccentricity",0.57,var=0.22,lower_limit=np.nextafter(0.,-1.), upper_limit=0.9)
# rf4 = RegionFilter("extent",0.6,var=0.06,lower_limit=0.5, upper_limit=np.nextafter(1.,2.))
# rf5 = RegionFilter("InOutContrast2", 0.18, var=0.04, lower_limit=0.05, upper_limit=np.nextafter(1.,2.))
# rf6 = RegionFilter("moments_hu", np.array([1.6e-1, 0., 0., 0., 0., 0., 0.]),
#                    var= np.array([1e-2, 1., 1., 1., 1., 1., 1.]),
#                    lower_limit=np.array([1.4e-1, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#                    upper_limit=np.array([2.0e-1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
from skimage.measure import regionprops, label
label

rf = RegionFilter("area",17.2,var=11.**2, lower_limit=4., upper_limit=50)
rf2 = RegionFilter("solidity",0.98,var=0.04**2, lower_limit=0.8)
rf3 = RegionFilter("eccentricity",0.51,var=0.31**2, upper_limit=0.95)
# rf4 = RegionFilter("extent",0.66,var=0.07**2, lower_limit=0.5, upper_limit=0.9)
rf5 = RegionFilter("InOutContrast2", 0.89, var=0.13**2, lower_limit=0.9)
if file_path.count("Starter3"):
    rf5 = RegionFilter("InOutContrast2", 0.8, var=0.13**2, lower_limit=0.9) # Starter 3
# rf5 = RegionFilter("InOutContrast2", 0.89, var=0.13**2, lower_limit=0.5)
rf6 = RegionFilter("mean_intensity", 60., var=17.**2, lower_limit=25.)
# rf7 = RegionFilter("max_intensity", 124, var=56, lower_limit=40)
# rf8 = RegionFilter("min_intensity", 21, var=14, lower_limit=0, upper_limit=70)

AD = RegionPropDetector([rf,rf2,rf3,rf5,rf6])


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
        try:
            img, meta = cam.getNewestImage()
        except StopIteration:
            break
        if img is not None:
            timestamp = datetime.fromtimestamp(meta["time"])
            timestamp += timedelta(milliseconds=int(meta["time_ms"]))
            # segmentation_pipe_in.send([i,int8(img)])
            segmentation_pipe_in.send([i, img])
            Image_write_queue.put([timestamp, i, meta])
            Timer_in.put([i, time.time()])
            print("loaded image %s" % i)
            # print(meta)
            i+=1


def segmentate():
    LastMap = None
    while True:
        # i, img = segmentation_queue.get()
        i, img = segmentation_pipe_out.recv()
        print("starting Segmentation %s"%i)
        SegMap, diff = VB.segmentate(img, do_neighbours=False, return_diff=True)
        # import  matplotlib.pyplot as plt
        # plt.imshow(VB.DumbStory)
        # plt.show()
        if LastMap is None:
            LastMap = np.zeros_like(SegMap, dtype=bool)
        SegMap &= ~NoMask
        VB.update(SegMap, img, do_neighbours=False)
        SegMap = binary_dilation(SegMap, selem=disk(5))
        SegMap = binary_erosion(SegMap, selem=disk(4))
        # mask = SegMap & ~ (SegMap & LastMap)
        SegMap_write_queue.put([i, SegMap])
        detection_pipe_in.send([i, SegMap, np.sum(diff, axis=0)])
        # LastMap = SegMap
        print("Segmentated Image %s"%i)


def detect():
    Map = None
    while True:
        # i, SegMap = detection_queue.get()
        i, SegMap, img = detection_pipe_out.recv()
        # from PIL import Image
        # from PenguTrack.Detectors import gray2rgb
        # save_im = Image.fromarray(img.astype(np.uint8))
        # save_im.save("/home/birdflight/Desktop/Diffs/%s.png"%i, "PNG")
        # diff = np.sum(diff, axis=0)
        Positions = AD.detect(SegMap, intensity_image=img)
        # X = np.asarray([[pos.PositionX, pos.PositionY] for pos in Positions])
        # if Map is None:
        #     Map = np.histogram2d(X.T[0], X.T[1])
        # else:
        #     Map *= 0.5
        #     Map += np.histogram2d(X.T[0], X.T[1])
        # import matplotlib.pyplot as plt
        # diff[~SegMap] = 0
        # plt.imshow(diff)
        # plt.figure()
        # plt.imshow(SegMap)
        # plt.show()
        # X = np.asarray([[pos.PositionX, pos.PositionY] for pos in Positions])
        # Positions = [pos for pos in Positions if np.sum(((pos.PositionX-X.T[0])**2+(pos.PositionY-X.T[1])**2)**0.5 < 25) < 2]
        # Positions = [pos for pos in Positions if np.sum(((pos.PositionX-X.T[0])**2+(pos.PositionY-X.T[1])**2)**0.5 < 200) < 2]
        Detection_write_queue.put([i, Positions])
        tracking_pipe_in.send([i, Positions])
        print("Found %s animals in %s!"%(len(Positions), i))

        # if not detection_pipe_out.poll(1) and not segmentation.is_alive():
        #     break


def track():
    while True:
        with np.errstate(all="raise"):
            i, Positions = tracking_pipe_out.recv()
            MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)
            # predictions = [MultiKal.Model.measure(p) for p in MultiKal.Predicted_X.values()]
            # predicted_err = [MultiKal.Model.measure(p) for p in MultiKal.Predicted_X.values()]
            # Positions = []
            if len(Positions) > 2:
                print("Tracking %s" % i)
                # Update Filter with new Detections
                MultiKal.update(z=Positions, i=i)
                Track_write_queue.put([i, MultiKal.ActiveFilters])
            else:
                print("empty track at %s"%i)
                Track_write_queue.put([i, {}])
            print("Got %s Filters in frame %s" % (len(MultiKal.ActiveFilters.keys()), i))


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
                fname = timestamp.strftime("%Y%m%d-%H%M%S")+"-%s"%int(timestamp.microsecond/100000) + "_Fino4_SpynelS.jpg"
                fname = meta['file_name']
                # path = db.setPath(r'/home/birdflight/Data/SPYNELS/20160405/02/')
                # path = db.setPath(db_start.getPaths()[0].path)
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
                    if not image_dict.has_key(i):
                        SegMap_write_queue.put([i, Mask])
                        break
                    # db.getImage(id=i)
                    db.setMask(image=db.getImage(id=image_dict[i]), data=(PT_Mask_Type.index * (~Mask).astype(np.uint8)))
                    print("Masks set! %s" % i)
                while not Detection_write_queue.empty():
                    i, Positions = Detection_write_queue.get()
                    if not image_dict.has_key(i):
                        Detection_write_queue.put([i,Positions])
                        break
                    #
                    for pos in Positions:
                        while True:
                            if image_dict.has_key(i):
                                break
                        detection_marker = db.setMarker(image=db.getImage(id=image_dict[i]),
                                                x=pos.PositionY, y=pos.PositionX,
                                                text="Detection  %.2f \n %s" % (pos.Log_Probability, "\n".join(["%s \t %s"%(k, pos.Data[k]) for k in pos.Data])),
                                                type=PT_Detection_Type)
                        db.setMeasurement(marker=detection_marker, log=pos.Log_Probability, x=pos.PositionX, y=pos.PositionY)
                    print("Detections written! %s"%i)
                while not Track_write_queue.empty():
                    i, ActiveFilters = Track_write_queue.get()
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
                            # prob = ActiveFilters[k].log_prob(keys=[i], compare_bel=False)
                            prob = ActiveFilters[k].log_prob(keys=[i])
                            db.setMarker(image=db.getImage(id=image_dict[i]), x=y, y=x,
                                         track=track,
                                         text="Track %s, Prob %.2f" % (k, prob),
                                         type=PT_Track_Type)
                        if ActiveFilters[k].Predicted_X.has_key(i):
                            pred_x, pred_y = MultiKal.Model.measure(ActiveFilters[k].Predicted_X[i])
                            db.setMarker(image=db.getImage(id=image_dict[i]), x=pred_y, y=pred_x,
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
#

loading = Process(target=load, args=(cam,))
segmentation = Process(target=segmentate)
detection = Process(target=detect)
tracking = Process(target=track)
# writing_SegMaps = Process(target=mask_writing)
# writing_Detections = Process(target=detection_writing)
# writing_Tracks = Process(target=track_writing)
writing_DB = Process(target=DB_write)

loading.start()
segmentation.start()
detection.start()
tracking.start()
# writing_SegMaps.start()
# writing_Detections.start()
# writing_Tracks.start()
writing_DB.start()

times1 = {}
times2 = {}
start = time.time()
while True:
    try:
        # print("SegPipe",segmentation_pipe_out.poll())
        # print("DetPipe",detection_pipe_out.poll())
        # print("TrackPipe",tracking_pipe_out.poll())
        if not Timer_in.empty():
            i, value = Timer_in.get()
            times1.update({i: value})

        if not Timer_out.empty():
            i, value = Timer_out.get()
            times2.update({i: value})
            for i in times2:
                if times1.has_key(i):
                    pass
                    # print("Time for image %s is %s seconds!"%(i, times2[i]-times1[i]))
                else:
                    print("DAMN!!")
            print("Needed %s s per frame"%((time.time()-start)/len(times2.keys())))

    except (KeyboardInterrupt, SystemExit):
        loading.terminate()
        segmentation.terminate()
        detection.terminate()
        tracking.terminate()
        writing_DB.terminate()
        raise

