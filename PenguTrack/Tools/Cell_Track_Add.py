from __future__ import division, print_function
import cv2
import numpy as np
import peewee
import sys

from qtpy import QtGui, QtCore, QtWidgets
from qimage2ndarray import array2qimage
import qtawesome as qta

from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import RandomWalk
# from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector
from PenguTrack.Detectors import SimpleAreaDetector2 as AreaDetector
from PenguTrack.Detectors import TresholdSegmentation, VarianceSegmentation
from PenguTrack.Detectors import Measurement as PT_Measurement

import scipy.stats as ss
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter

import clickpoints

# MaxI_db = clickpoints.DataFile("/home/alex/2017-03-10_Tzellen_microwells_bestdata/30sec/max_Indizes.cdb")
res = 6.45/10

__icon__ = "fa.coffee"

# Connect to database
start_frame, database, port = clickpoints.GetCommandLineArgs()
db = clickpoints.DataFile(database)
com = clickpoints.Commands(port, catch_terminate_signal=True)
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
    try:
        db.db.connect()
    except peewee.OperationalError:
        pass
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
# # get the images
# images = db.getImageIterator(start_frame=start_frame)

class PenguTrackWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(QtWidgets.QWidget, self).__init__(parent)
        self.setWindowTitle("PenguTrack - ClickPoints")
        self.setWindowIcon(qta.icon("fa.coffee"))

        # window layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(20, 10, 20, 10)
        self.setLayout(self.layout)
        self.texts = {}
        self.formats = {}
        self.min_max = {}
        self.functions = {}
        font = QtGui.QFont("", 11)

        # add spinboxes
        self.spinboxes = {}
        self.spin_functions = [self.pt_set_minsize,self.pt_set_maxsize, self.pt_set_number, self.pt_set_distxy_boundary, self.pt_set_distz_boundary]
        self.spin_start = [5, 30, 200, 10, 21]
        self.spin_min_max = [[1, 100], [1, 100], [1, 1000], [1,1000], [1,1000]]
        self.spin_formats = ["    %4d","    %4d", "    %4d","    %4d","    %4d"]
        for i, name in enumerate(["Object Size min(px)","Object Size max(px)", "Object Number (approx.)", "Pre-Stitching Distxy(px)", "Pre-Stitching Distz(px)"]):
            sublayout = QtWidgets.QHBoxLayout()
            sublayout.setContentsMargins(0, 0, 0, 0)
            spin = QtWidgets.QSpinBox(self)
            # spin.format = formats[i]
            spin.setRange(self.spin_min_max[i][0], self.spin_min_max[i][1])
            spin.setValue(self.spin_start[i])
            spin.editingFinished.connect(lambda spin=spin, i=i, name=name: self.spin_functions[i](spin.value(), name))
            spin.valueChanged.connect(lambda value, name=name: self.update_text(value, name))
            self.spinboxes.update({name: spin})
            self.functions.update({name: self.spin_functions[i]})
            sublayout.addWidget(spin)
            text = QtWidgets.QLabel(self)
            text.setText(name + ": " + self.spin_formats[i] % spin.value())
            # text.setBrush(QtGui.QBrush(QtGui.QColor("white")))
            # text.setText(name + ": " + formats[i] % slider.value())
            sublayout.addWidget(text)
            self.formats.update({name: self.spin_formats[i]})
            self.texts.update({name: text})
            self.layout.addLayout(sublayout)

        # Add Detection Button
        self.detect_button = QtWidgets.QPushButton()
        self.detect_button.clicked.connect(self.reload_markers)
        self.detect_button.setCheckable(True)
        self.detect_button.setText("Detect Objects")
        self.layout.addWidget(self.detect_button)


        # Initialise PenguTrack
        self.object_size = self.spin_start[0]  # Object diameter (smallest)
        self.object_number = self.spin_start[1]  # Number of Objects in First Track
        self.object_area = int((self.spin_start[0]/2.)**2*np.pi)

        # add even more sliders. tihihi...
        self.sliders = {}
        self.slider_functions = [self.pt_set_q, self.pt_set_r, self.pt_set_lum_treshold, self.pt_set_var_treshold]
        self.slider_min_max = [[1, 10], [1, 10], [1, 2**12], [-100, 100]]
        self.slider_start = [2, 1, 2**11, -7]
        self.slider_formats = [" %3d", "    %3d", "    %.2f", "    %3d"]
        for i, name in enumerate(["Prediciton Error", "Detection Error", "Luminance Threshold", "Variance Threshold"]):
            sublayout = QtWidgets.QHBoxLayout()
            sublayout.setContentsMargins(0, 0, 0, 0)
            slider = QtWidgets.QSlider(self)
            slider.setMinimum(self.slider_min_max[i][0])
            slider.setMaximum(self.slider_min_max[i][1])
            slider.setValue(self.slider_start[i])
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.sliderReleased.connect(
                lambda slider=slider, i=i, name=name: self.slider_functions[i](slider.value(), name))
            # slider.valueChanged.connect(lambda name=name: self.update_text(value, name))
            self.sliders.update({name: slider})
            self.functions.update({name: self.slider_functions[i]})
            sublayout.addWidget(slider)
            text = QtWidgets.QLabel(self)
            if name != "Luminance Threshold":
                text.setText(name + ": " + self.slider_formats[i] % slider.value())
            else:
                text.setText(name + ": " + self.slider_formats[i] % (slider.value()/self.slider_min_max[i][1]))
            sublayout.addWidget(text)
            self.min_max.update({name: self.slider_min_max[i]})
            self.formats.update({name: self.slider_formats[i]})
            self.texts.update({name: text})
            self.layout.addLayout(sublayout)

        # Add start Button
        self.start_button = QtWidgets.QPushButton()
        self.start_button.clicked.connect(self.track)
        self.start_button.setCheckable(True)
        self.start_button.setText("Start Tracking!")
        self.layout.addWidget(self.start_button)


        # Initialize physical model as 3d variable speed model with 1 Hz frame-rate
        # self.model = VariableSpeed(1, 1, dim=3, timeconst=1)
        self.model = RandomWalk(dim=3)
        self.q = self.slider_start[0]
        self.r = self.slider_start[1]
        X = np.zeros(6).T  # Initial Value for Position
        Q = np.diag([self.q * self.object_size, self.q * self.object_size, self.q * self.object_size])  # Prediction uncertainty
        R = np.diag([self.r * self.object_size, self.r * self.object_size, self.r * self.object_size])  # Measurement uncertainty

        self.FilterType = KalmanFilter
        State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
        Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

        # Initialize Filter
        self.Tracker = MultiFilter(self.FilterType, self.model, np.diag(Q),
                               np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
        self.Tracker.AssignmentProbabilityThreshold=0.
        self.Tracker.MeasurementProbabilityThreshold=0.
        self.Tracker.LogProbabilityThreshold=-30.

        # Init_Background from Image_Median
        N = db.getImages(layer=0).count()
        # init = np.array(np.median([np.asarray(db.getImage(frame=j, layer=2).data, dtype=np.int)
        #                            for j in np.random.randint(0, N, 5)], axis=0), dtype=np.int)

        # Init Segmentation Module with Init_Image
        self.Segmentation = TresholdSegmentation(treshold=float(self.slider_start[2])/self.slider_min_max[2][1])
        self.Segmentation2 = VarianceSegmentation(int(self.slider_start[3]), int(np.ceil(self.object_size/2.)))

        # Init Detection Module
        self.Detector = AreaDetector(self.object_area, self.object_number)
        self.Detector.LowerLimit = int((self.spin_start[0]/2.)**2*np.pi)
        self.Detector.UpperLimit = int((self.spin_start[1]/2.)**2*np.pi)

        # Define ClickPoints Marker

        marker_type = db.getMarkerType(name="PT_Detection_Marker")
        if not marker_type:
            marker_type = db.setMarkerType(name="PT_Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        db.deleteMarkers(type=marker_type)
        self.detection_marker_type = marker_type

        marker_type2 = db.getMarkerType(name="PT_Track_Marker")
        if not marker_type2:
            marker_type2 = db.setMarkerType(name="PT_Track_Marker", color="#00FF00", mode=db.TYPE_Track)
        db.deleteMarkers(type=marker_type2)
        self.track_marker_type = marker_type2

        marker_type3 = db.getMarkerType(name="PT_Prediction_Marker")
        if not marker_type3:
            marker_type3 = db.setMarkerType(name="PT_Prediction_Marker", color="#0000FF")
        db.deleteMarkers(type=marker_type3)
        self.prediction_marker_type = marker_type3

        if not db.getMaskType(name="PT_SegMask"):
            self.mask_type = db.setMaskType(name="PT_SegMask", color="#FF59E3")
        else:
            self.mask_type = db.getMaskType(name="PT_SegMask")
        self.mask_type_id = self.mask_type.id

        # Delete Old Tracks
        db.deleteTracks(type=self.track_marker_type)

        com.ReloadTypes()
        com.ReloadMarker()

        print('Initialized')

        # Get images and template
        self.current_layer = com.CurrentLayer()
        self.image_iterator = db.getImageIterator(start_frame=start_frame, layer=self.current_layer)
        self.current_image = db.getImage(frame=start_frame, layer=self.current_layer)
        self.current_frame = start_frame
        self.image_data = self.current_image.data
        self.markers = db.getMarkers(frame=start_frame)
        self.current_marker = None

        self.reload_mask()
        # start to display first image
        # QtCore.QTimer.singleShot(1, self.displayNext)

    def update_text(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name]%value)

    def pt_set_var_treshold(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        # self.variance_treshold = int(np.ceil(2**(value/10)))
        self.variance_treshold = 2**(value/10)
        print("Setting Variance-Treshold to %s"%self.variance_treshold)
        self.Segmentation2.Treshold = self.variance_treshold
        self.reload_mask()
        # pass

    def pt_set_lum_treshold(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % (self.sliders[name].value()/self.min_max[name][1]))
        self.luminance_treshold = float(value)/self.min_max[name][1]
        self.Segmentation.Treshold = self.luminance_treshold
        self.reload_mask()
        # self.reload_markers()

    def pt_set_distxy_boundary(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
        self.Detector.distxy_boundary = int(value)
        # self.reload_markers()

    def pt_set_distz_boundary(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
        self.Detector.distz_boundary = int(value)
        # self.reload_markers()

    def pt_set_number(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.object_number = int(value)
        self.Detector.ObjectArea = self.object_area
        self.Detector.ObjectNumber = self.object_number
        self.reload_markers()

    def pt_set_area(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
        self.object_area = int(value)
        self.Detector.ObjectArea = self.object_area
        self.Detector.ObjectNumber = self.object_number
        # self.reload_markers()

    def pt_set_minsize(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
        # self.object_size = int(value)
        # self.Segmentation2.Radius = int(np.ceil(value/2.))
        # self.object_area = int((self.object_size/2.)**2*np.pi)
        # self.Detector.ObjectArea = self.object_area
        self.Detector.LowerLimit = int((value/2.)**2*np.pi)
        self.Detector.ObjectNumber = self.object_number

        self.r = int(value)
        X = np.zeros(6).T  # Initial Value for Position
        Q = np.diag([self.q * self.object_size * res,
                     self.q * self.object_size * res,
                     self.q * self.object_size * res])  # Prediction uncertainty
        R = np.diag([self.r * self.object_size * res,
                     self.r * self.object_size * res,
                     self.r * self.object_size * res])  # Measurement uncertainty

        State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
        Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
        self.update_filter_params(np.diag(Q), np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

        self.reload_mask()

    def pt_set_maxsize(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
        # self.object_size = int(value)
        # self.Segmentation2.Radius = int(np.ceil(value/2.))
        # self.object_area = int((self.object_size/2.)**2*np.pi)
        # self.Detector.ObjectArea = self.object_area
        self.Detector.UpperLimit = int((value/2.)**2*np.pi)
        self.Detector.ObjectNumber = self.object_number

        self.r = int(value)
        X = np.zeros(6).T  # Initial Value for Position
        Q = np.diag([self.q * self.object_size * res,
                     self.q * self.object_size * res,
                     self.q * self.object_size * res])  # Prediction uncertainty
        R = np.diag([self.r * self.object_size * res,
                     self.r * self.object_size * res,
                     self.r * self.object_size * res])  # Measurement uncertainty

        State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
        Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
        self.update_filter_params(np.diag(Q), np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

        self.reload_mask()

    # def pt_set_size(self, value, name):
    #     self.texts[name].setText(name + ": " + self.formats[name] % self.spinboxes[name].value())
    #     self.object_size = int(value)
    #     self.Segmentation2.Radius = int(np.ceil(value/2.))
    #     self.object_area = int((self.object_size/2.)**2*np.pi)
    #     self.Detector.ObjectArea = self.object_area
    #     self.Detector.ObjectNumber = self.object_number
    #
    #     self.r = int(value)
    #     X = np.zeros(6).T  # Initial Value for Position
    #     Q = np.diag([self.q * self.object_size * res,
    #                  self.q * self.object_size * res,
    #                  self.q * self.object_size * res])  # Prediction uncertainty
    #     R = np.diag([self.r * self.object_size * res,
    #                  self.r * self.object_size * res,
    #                  self.r * self.object_size * res])  # Measurement uncertainty
    #
    #     State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    #     Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
    #     self.update_filter_params(np.diag(Q), np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
    #
    #     self.reload_mask()

    def pt_set_r(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.r = int(value)
        X = np.zeros(6).T  # Initial Value for Position
        Q = np.diag([self.q * self.object_size * res,
                     self.q * self.object_size * res,
                     self.q * self.object_size * res])  # Prediction uncertainty
        R = np.diag([self.r * self.object_size * res,
                     self.r * self.object_size * res,
                     self.r * self.object_size * res])  # Measurement uncertainty

        State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
        Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
        self.update_filter_params(np.diag(Q), np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

    def pt_set_q(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.q = int(value)
        X = np.zeros(4).T  # Initial Value for Position
        Q = np.diag([self.q * self.object_size * res,
                     self.q * self.object_size * res,
                     self.q * self.object_size * res])  # Prediction uncertainty
        R = np.diag([self.r * self.object_size * res,
                     self.r * self.object_size * res,
                     self.r * self.object_size * res])  # Measurement uncertainty

        State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
        Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
        self.update_filter_params(np.diag(Q), np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)

    def reload_markers(self):
        print("Reloading Markers")
        self.current_frame = com.CurrentImage()
        self.current_image = db.getImage(frame=self.current_frame, layer=self.current_layer)
        self.image_data = self.current_image.data
        db.deleteMarkers(frame=self.current_frame, type=self.detection_marker_type)
        # SegMap = ~db.getMask(frame=self.current_frame, layer=0).data.astype(bool)
        # if np.sum(SegMap) < np.sum(~SegMap):
        #     Index_Image = db.getImage(frame=self.current_frame, layer=3).data
        #     Map = np.zeros_like(Index_Image)
        #     Map[SegMap] = Index_Image[SegMap]
        #     Positions = self.Detector.detect(Map)
        # else:
        #     Positions = self.Detector.detect(SegMap)
        self.Detector.ObjectArea = int((self.Detector.UpperLimit+self.Detector.LowerLimit)/2.)
        print(self.Detector.ObjectArea, self.Detector.ObjectNumber)
        mask = db.getMask(frame=self.current_frame, layer=0).data.astype(bool)

        index_data = db.getImage(frame=self.current_frame, layer=1).data
        Positions = self.Detector.detect(index_data, mask, only_for_detection=True)
        # index_data[mask] = np.zeros_like(index_data)[mask]
        #
        # # index_3d = np.zeros((np.amax(index_data),)+index_data.shape, dtype=np.uint16)
        # # for z in range(np.amax(index_data)):
        # #     index_3d[z] = index_data == (z+1)
        # # from skimage.measure import label
        # # index_3d = label(index_3d, connectivity=3)
        # # index_data2 = np.sum(index_3d, axis=0)
        #
        # j_max = np.amax(index_data)
        # stack = np.zeros((j_max, index_data.shape[0], index_data.shape[1]), dtype=np.bool)
        # for j in range(j_max):
        #     stack[j, index_data == j] = True
        #
        # labels, n = label(stack, structure=np.ones((3, 3, 3)))
        #
        # index_data2 = np.zeros_like(index_data)
        # for l in labels:
        #     index_data2[l > 0] = l[l > 0]
        #
        # Positions = self.Detector.detect(index_data2)
        # # Positions = self.Detector.detect(~db.getMask(frame=self.current_frame, layer=0).data.astype(bool))
        # ####
        # Positions_cor = []
        # for i1, pos1 in enumerate(Positions):
        #     x1 = pos1.PositionX
        #     y1 = pos1.PositionY
        #     z1 = index_data[int(x1), int(y1)] * 10.
        #     inc = 0
        #     for j1, pos2 in enumerate(Positions):
        #         x2 = pos2.PositionX
        #         y2 = pos2.PositionY
        #         z2 = index_data[int(x2), int(y2)] * 10.
        #         dist = np.sqrt((x1 - x2)**2. + (y1 - y2)**2.)
        #         distz = np.abs(z1-z2)
        #         if dist < 10 and dist !=0 and distz < 20:
        #             x3 = (x1+x2)/2.
        #             y3 = (y1+y2)/2.
        #             if [x3,y3] not in Positions_cor:
        #                 Positions_cor.append([x3,y3])
        #                 print ("Replaced")
        #             inc+=1
        #     if inc == 0:
        #         Positions_cor.append([x1,y1])
        # ####
        self.detect_button.setChecked(False)
        if False:#len(Positions) > self.object_number*10:
            pass
            #print("No update! Too many objects detected.")
        else:
            print("Found %s Objects!"%len(Positions))
            # for pos in Positions:
            #     db.setMarker(frame=self.current_frame, layer=0, y=pos.PositionX, x=pos.PositionY, type=self.detection_marker_type)
            ####
            # print("Found %s Objects Cor!" % len(Positions_cor))
            # for pos in Positions_cor:
            for pos in Positions:
                db.setMarker(frame=self.current_frame, layer=0, y=pos[0], x=pos[1], type=self.detection_marker_type)
            ####
            com.ReloadMarker()

    def reload_mask(self):
        print("Reloading Mask")
        self.current_frame = com.CurrentImage()
        self.current_image = db.getImage(frame=self.current_frame, layer=self.current_layer)
        self.image_data = self.current_image.data
        # temp = db.getImage(frame=self.current_frame, layer=2).data.astype(np.float)
        # SegMap1 = self.Segmentation.segmentate(gaussian_filter(temp, 1.5))
        #####
        immin = db.getImage(frame=self.current_frame, layer=0).data
        immin = 1.0 * np.asarray(immin)
        immin -= np.min(immin)
        immin = (immin / np.max(immin))
        immin[immin > 1] = 1
        immax = db.getImage(frame=self.current_frame, layer=2).data
        immax = 1.0 * np.asarray(immax)
        immax -= np.min(immax)
        immax[immax < 0] = 0
        immax = (immax / np.max(immax))
        immax = 1 - immax
        im = immin * immax
        im = im - np.min(im)
        im = im / np.max(im)
        immin_ind = db.getImage(frame=self.current_frame, layer=1).data
        immax_ind = db.getImage(frame=self.current_frame, layer=3).data
        immin_ind = 1.0 * np.asarray(immin_ind)
        immax_ind = 1.0 * np.asarray(immax_ind)
        ind_diff = - immax_ind + immin_ind
        ind_diff = np.abs(ind_diff - 5)
        ind_diff = np.exp(-ind_diff / 2.)
        ind_diff = gaussian_filter(ind_diff, 5)
        imf = 1 - ((1 - im) * (ind_diff))
        print(self.Segmentation.Treshold)
        SegMap1 = self.Segmentation.segmentate(imf)
        # import matplotlib.pyplot as plt
        # plt.imshow(SegMap1)
        # plt.figure()
        # plt.imshow(imf)
        # plt.show()
        ####
        # SegMap1 = self.Segmentation.segmentate(db.getImage(frame=self.current_frame, layer=0).data)
        #import matplotlib.pyplot as plt
        #plt.imshow(db.getImage(frame=self.current_frame, layer=2).data)
        #plt.figure()
        SegMap2 = self.Segmentation2.segmentate(db.getImage(frame=self.current_frame, layer=1).data)
        #plt.imshow(db.getImage(frame=self.current_frame, layer=1).data)
        #plt.show()
        SegMap = ~SegMap1# & SegMap2
        # SegMap = SegMap2 #SegMap1 | SegMap2
        db.setMask(frame=self.current_frame, layer=0, data=((~SegMap).astype(np.uint8)))
        com.ReloadMask()

    def reload(self):
        print("Reloading Mask and Markers!")
        self.reload_mask()
        self.reload_markers()

    def update_filter_params(self, *args, **kwargs):
        self.Tracker.filter_args = args
        self.Tracker.filter_kwargs = kwargs
        for f in self.Tracker.Filters:
            obj = self.Tracker.Filters.pop(f)
            del obj
        for f in self.Tracker.ActiveFilters:
            obj = self.Tracker.Filters.pop(f)
            del obj

        self.Tracker.predict(u=np.zeros((self.model.Control_dim,)).T, i=self.current_frame)

    def track(self, value):
        if value:
            print("Starting Tracking")
            images = db.getImageIterator(start_frame=start_frame, layer=0)
            for image in images:
                i = image.sort_index
                print("Doing Frame %s"%i)
                # i = image.get_id()

                Index_Image = db.getImage(frame=i, layer=1).data
                # Prediction step
                try:
                    self.Tracker.predict(u=np.zeros((self.model.Control_dim,)).T, i=i)
                except KeyError:
                    print(i)
                    print(self.Tracker.X.keys())
                    raise
                #####
                immin = db.getImage(frame=i, layer=0).data
                immin = 1.0 * np.asarray(immin)
                immin -= np.min(immin)
                immin = (immin / np.max(immin))
                immin[immin > 1] = 1
                immax = db.getImage(frame=i, layer=2).data
                immax = 1.0 * np.asarray(immax)
                immax -= np.min(immax)
                immax[immax < 0] = 0
                immax = (immax / np.max(immax))
                immax = 1 - immax
                im = immin * immax
                im = im - np.min(im)
                im = im / np.max(im)
                immin_ind = db.getImage(frame=i, layer=1).data
                immax_ind = db.getImage(frame=i, layer=3).data
                immin_ind = 1.0 * np.asarray(immin_ind)
                immax_ind = 1.0 * np.asarray(immax_ind)
                ind_diff = - immax_ind + immin_ind
                ind_diff = np.abs(ind_diff - 5)
                ind_diff = np.exp(-ind_diff / 2.)
                ind_diff = gaussian_filter(ind_diff, 5)
                imf = 1 - ((1 - im) * (ind_diff))
                SegMap1 = self.Segmentation.segmentate(imf)
                #####
                # Detection step
                # temp = db.getImage(frame=i, layer=2).data.astype(np.float)
                # SegMap1 = self.Segmentation.segmentate(gaussian_filter(temp, 1.5))

                # SegMap1 = self.Segmentation.segmentate(db.getImage(frame=i, layer=0).data)
                SegMap2 = self.Segmentation2.detect(db.getImage(frame=i, layer=1).data) #image.data)
                SegMap = ~SegMap1 #& SegMap2
                db.setMask(frame=i, layer=0, data=((~SegMap).astype(np.uint8)))
                # SegMap = self.Segmentation.detect(image.data)
                # Map = np.zeros_like(Index_Image)
                # Map[SegMap] = Index_Image[SegMap]
                mask = db.getMask(frame=i, layer=0).data.astype(bool)

                # ------
                index_data = db.getImage(frame=i, layer=1).data
                # index_data[mask] = np.zeros_like(index_data)[mask]
                #
                # # index_3d = np.zeros((np.amax(index_data),) + index_data.shape, dtype=np.uint16)
                # # for z in range(np.amax(index_data)):
                # #     index_3d[z] = index_data == (z + 1)
                # # from skimage.measure import label
                # # index_3d = label(index_3d, connectivity=3)
                # # index_data2 = np.sum(index_3d, axis=0)
                # # index_data2 = np.sum(label(index_3d, connectivity=3), axis=0)
                #
                # j_max = np.amax(index_data)
                # stack = np.zeros((j_max, index_data.shape[0], index_data.shape[1]), dtype=np.bool)
                # for j in range(j_max):
                #     stack[j, index_data == j] = True
                #
                # labels, n = label(stack, structure=np.ones((3, 3, 3)))
                #
                # index_data2 = np.zeros_like(index_data)
                # for l in labels:
                #     index_data2[l > 0] = l[l > 0]
                #
                # Positions2D = self.Detector.detect(index_data2)
                # # Positions2D = self.Detector.detect(~db.getMask(frame=i, layer=0).data.astype(bool))
                # ####
                # # Positions2D_cor = []
                # # for i1, pos1 in enumerate(Positions2D):
                # #     Log_Probability1 = pos1.Log_Probability
                # #     Track_Id1 = pos1.Track_Id
                # #     Frame1 = pos1.Frame
                # #     x1 = pos1.PositionX
                # #     y1 = pos1.PositionY
                # #     z1 = index_data[int(x1), int(y1)] * 10.
                # #     PosZ1 = index_data[int(x1), int(y1)]
                # #     inc = 0
                # #     for j1, pos2 in enumerate(Positions2D):
                # #         Log_Probability2 = pos2.Log_Probability
                # #         Log_Probabilitymax = np.max([Log_Probability1,Log_Probability2])
                # #         x2 = pos2.PositionX
                # #         y2 = pos2.PositionY
                # #         z2 = index_data[int(x2), int(y2)] * 10.
                # #         PosZ2 = index_data[int(x2), int(y2)]
                # #         PosZmax = np.max([PosZ1, PosZ2])
                # #         dist = np.sqrt((x1 - x2) ** 2. + (y1 - y2) ** 2.)
                # #         distz = np.abs(z1 - z2)
                # #         if dist < 10 and dist != 0 and distz < 20:
                # #             x3 = (x1 + x2) / 2.
                # #             y3 = (y1 + y2) / 2.
                # #             if [x3, y3, Log_Probabilitymax, PosZmax] not in Positions2D_cor:
                # #                 Positions2D_cor.append([x3, y3, Log_Probabilitymax, PosZmax])
                # #                 print("Replaced")
                # #                 print(x3)
                # #                 print(y3)
                # #                 print (Log_Probabilitymax)
                # #                 print(PosZmax)
                # #                 print('###')
                # #             inc += 1
                # #     if inc == 0:
                # #         Positions2D_cor.append([x1, y1, Log_Probability1, PosZ1])
                # ####
                #
                # Positions3D = []
                # for pos in Positions2D:
                #     posZ = index_data[int(pos.PositionX), int(pos.PositionY)]  # war mal "Index_Image"
                #     Positions3D.append(PT_Measurement(pos.Log_Probability,
                #                                       [pos.PositionX * res, pos.PositionY * res, posZ * 10],
                #                                       frame=pos.Frame,
                #                                       track_id=pos.Track_Id))
                # # ------
                # ####
                # # for pos in Positions2D_cor:
                # #     # posZ = index_data[int(pos[0]), int(pos[1])]
                # #     posZ = pos[3]
                # #     Positions3D.append(PT_Measurement(pos[2],
                # #                                       [pos[0] * res, pos[1] * res, posZ * 10]))
                # ####
                # Positions = Positions3D  # convenience

                Positions = self.Detector.detect(index_data, mask)

                # Setting Mask in ClickPoints
                m = db.setMask(frame=i, layer=0, data=(~SegMap).astype(np.uint8))
                print("Mask save", m)
                n = 1
                if len(Positions) != 0:
                # if np.all(Positions != np.array([])):

                    # Update Filter with new Detections
                    try:
                        self.Tracker.update(z=Positions, i=i)
                    except TypeError:
                        print(self.Tracker.filter_args)
                        print(self.Tracker.filter_kwargs)
                        raise

                    # Get Tracks from Filter (a little dirty)
                    for k in self.Tracker.Filters.keys():
                        x = y = z = np.nan
                        if i in self.Tracker.Filters[k].Measurements.keys():
                            meas = self.Tracker.Filters[k].Measurements[i]
                            x = meas.PositionX
                            y = meas.PositionY
                            z = meas.PositionZ
                            prob = self.Tracker.Filters[k].log_prob(keys=[i], compare_bel=False)
                        elif i in self.Tracker.Filters[k].X.keys():
                            meas = None
                            x, y, z = self.Tracker.Model.measure(self.Tracker.Filters[k].X[i])
                            prob = self.Tracker.Filters[k].log_prob(keys=[i], compare_bel=False)

                        if i in self.Tracker.Filters[k].Measurements.keys():
                            pred_x, pred_y, pred_z = self.Tracker.Model.measure(self.Tracker.Filters[k].Predicted_X[i])
                            prob = self.Tracker.Filters[k].log_prob(keys=[i], compare_bel=False)

                        x_img = y / res
                        y_img = x / res
                        pred_x_img = pred_y / res
                        pred_y_img = pred_x / res
                        try:
                            db.setMarker(frame=i, layer=0, x=x_img, y=y_img, text="Detection %s" % k,
                                         type=self.detection_marker_type)
                        except:
                            pass
                        # Write assigned tracks to ClickPoints DataBase
                        if np.isnan(x) or np.isnan(y):
                            pass
                        else:
                            pred_marker = db.setMarker(frame=i, layer=0, x=pred_x_img, y=pred_y_img,
                                                       text="Track %s" % (1000 + k), type=self.prediction_marker_type)
                            if db.getTrack(k + 1000):
                                track_marker = db.setMarker(frame=i, layer=0, type=self.track_marker_type,
                                                            track=(1000 + k),
                                                            x=x_img, y=y_img,
                                                            text='Track %s, Prob %.2f, Z-Position %s' % ((1000 + k), prob, z))

                                print('Set Track(%s)-Marker at %s, %s' % ((1000 + k), x_img, y_img))
                            else:
                                db.setTrack(self.track_marker_type, id=1000 + k, hidden=False)
                                if k == self.Tracker.CriticalIndex:
                                    db.setMarker(image=i, layer=0, type=self.track_marker_type, x=x_img, y=y_img,
                                                 text='Track %s, Prob %.2f, CRITICAL' % ((1000 + k), prob))
                                track_marker = db.setMarker(image=image, type=self.track_marker_type,
                                                            track=1000 + k,
                                                            x=x_img,
                                                            y=y_img,
                                                            text='Track %s, Prob %.2f, Z-Position %s' % ((1000 + k), prob, z))
                                print('Set new Track %s and Track-Marker at %s, %s' % ((1000 + k), x_img, y_img))

                            # db.setMeasurement(marker=track_marker, log=prob, x=x, y=y, z=z)
                            try:
                                db.db.connect()
                            except peewee.OperationalError:
                                pass
                            meas_entry = Measurement(marker=track_marker, log=prob, x=x, y=y, z=z)
                            meas_entry.save()
                # from time import time
                # start = time()
                # com.JumpToFrame(i)
                # com.ReloadMarker(i)
                # com.ReloadMask()
                # print("Loading Frames costs %s s"%(time()-start))
                print("Got %s Filters" % len(self.Tracker.ActiveFilters.keys()))
                if not self.start_button.isChecked():
                    break


# create qt application
app = QtWidgets.QApplication(sys.argv)

# start window
window = PenguTrackWindow()
window.show()
app.exec_()

