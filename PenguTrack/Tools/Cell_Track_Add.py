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
from PenguTrack.Models import VariableSpeed
from PenguTrack.Detectors import SimpleAreaDetector as AreaDetector
from PenguTrack.Detectors import TresholdSegmentation
from PenguTrack.Detectors import Measurement as PT_Measurement

import scipy.stats as ss

import clickpoints

MaxI_db = clickpoints.DataFile("/home/alex/2017-03-10_Tzellen_microwells_bestdata/30sec/max_Indizes.cdb")
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
        self.functions = {}
        font = QtGui.QFont("", 11)

        # add spinboxes
        self.spinboxes = {}
        self.spin_functions = [self.pt_set_size, self.pt_set_number]
        self.spin_start = [10, 20]
        self.spin_min_max = [[1, 100], [1, 1000]]
        self.spin_formats = ["    %4d", "    %4d"]
        for i, name in enumerate(["Object Size (px)", "Object Number (approx.)"]):
            sublayout = QtWidgets.QHBoxLayout()
            sublayout.setContentsMargins(0, 0, 0, 0)
            spin = QtWidgets.QSpinBox(self)
            # spin.format = formats[i]
            spin.setValue(self.spin_start[i])
            spin.setRange(self.spin_min_max[i][0], self.spin_min_max[i][1])
            spin.editingFinished.connect(lambda spin=spin, i=i: self.spin_functions[i](spin.value(), name))
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

        # Initialise PenguTrack
        self.object_size = self.spin_start[0]  # Object diameter (smallest)
        self.object_number = self.spin_start[1]  # Number of Objects in First Track
        self.object_area = int(self.spin_start[0]**2*np.pi)

        # add even more sliders. tihihi...
        self.sliders = {}
        self.slider_functions = [self.pt_set_q, self.pt_set_r, self.pt_set_lum_treshold, self.pt_set_var_treshold]
        self.slider_min_max = [[1, 10], [1, 10], [1, 2**12], [1, 2**12]]
        self.slider_start = [2, 2, 128, 255]
        self.slider_formats = [" %3d", "    %3d", "    %3d", "    %3d"]
        for i, name in enumerate(["Prediciton Error", "Detection Error", "Luminance Treshold", "Variance Treshold"]):
            sublayout = QtWidgets.QHBoxLayout()
            sublayout.setContentsMargins(0, 0, 0, 0)
            slider = QtWidgets.QSlider(self)
            slider.setMinimum(self.slider_min_max[i][0])
            slider.setMaximum(self.slider_min_max[i][1])
            slider.setValue(self.slider_start[i])
            slider.setOrientation(QtCore.Qt.Horizontal)
            slider.sliderReleased.connect(
                lambda slider=slider, i=i, name=name: self.slider_functions[i](slider.value(), name))
            self.sliders.update({name: slider})
            self.functions.update({name: self.slider_functions[i]})
            sublayout.addWidget(slider)
            text = QtWidgets.QLabel(self)
            text.setText(name + ": " + self.slider_formats[i] % slider.value())
            sublayout.addWidget(text)
            self.formats.update({name: self.slider_formats[i]})
            self.texts.update({name: text})
            self.layout.addLayout(sublayout)

        # Add start Button
        self.start_button = QtWidgets.QPushButton()
        self.start_button.clicked.connect(self.track)
        self.start_button.setCheckable(True)
        self.layout.addWidget(self.start_button)


        # Initialize physical model as 3d variable speed model with 1 Hz frame-rate
        self.model = VariableSpeed(1, 1, dim=3, timeconst=1)
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

        # Init_Background from Image_Median
        N = db.getImages().count()
        init = np.array(np.median([np.asarray(db.getImage(frame=j).data, dtype=np.int)
                                   for j in np.random.randint(0, N, 5)], axis=0), dtype=np.int)

        # Init Segmentation Module with Init_Image
        self.Segmentation = TresholdSegmentation(treshold=int(self.slider_start[3]))

        # Init Detection Module
        self.Detector = AreaDetector(self.object_area, self.object_number)

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

        # Delete Old Tracks
        db.deleteTracks(type=self.track_marker_type)

        com.ReloadTypes()
        com.ReloadMarker()

        print('Initialized')

        # Get images and template
        self.image_iterator = db.getImageIterator(start_frame=start_frame)
        self.current_image = db.getImage(frame=start_frame)
        self.image_data = self.current_image.data
        self.markers = db.getMarkers(frame=start_frame)
        self.current_marker = None

        self.reload_mask()
        # start to display first image
        # QtCore.QTimer.singleShot(1, self.displayNext)

    def pt_set_var_treshold(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        pass

    def pt_set_lum_treshold(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.luminance_treshold = int(value)
        self.Segmentation.Treshold = self.luminance_treshold
        self.reload_mask()
        # self.reload_markers()

    def pt_set_number(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.object_number = int(value)
        self.Detector = AreaDetector(self.object_area, self.object_number)
        self.reload_markers()

    def pt_set_area(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.object_area = int(value)
        self.Detector = AreaDetector(self.object_area, self.object_number)
        self.reload_markers()

    def pt_set_size(self, value, name):
        self.texts[name].setText(name + ": " + self.formats[name] % self.sliders[name].value())
        self.object_size = int(value)
        self.object_area = int(self.object_size**2*np.pi)
        self.Detector = AreaDetector(self.object_area, self.object_number)
        self.reload_markers()
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
        db.deleteMarkers(image=self.current_image, type=self.detection_marker_type)
        Positions = self.Detector.detect(~db.getMask(image=self.current_image).data.astype(bool))
        print(len(Positions))
        for pos in Positions:
            db.setMarker(image=self.current_image, y=pos.PositionX, x=pos.PositionY, type=self.detection_marker_type)
        com.ReloadMarker()

    def reload_mask(self):
        self.setEnabled(False)
        SegMap = self.Segmentation.segmentate(self.image_data)
        db.setMask(image=self.current_image, data=(~SegMap).astype(np.uint8))
        com.ReloadMask()
        self.setEnabled(True)

    def update_filter_params(self, *args, **kwargs):
        self.Tracker.filter_args = args
        self.Tracker.filter_kwargs = kwargs
        for f in self.Tracker.Filters:
            obj = self.Tracker.Filters.pop(f)
            del obj
        for f in self.Tracker.ActiveFilters:
            obj = self.Tracker.Filters.pop(f)
            del obj

        self.Tracker.predict(u=np.zeros((self.model.Control_dim,)).T, i=self.current_image.frame)

    def track(self, value):
        if value:
            images = db.getImageIterator(start_frame=start_frame)
            for image in images:
                # i = image.frame
                i = image.get_id()

                Index_Image = MaxI_db.getImage(frame=i).data
                # Prediction step
                try:
                    self.Tracker.predict(u=np.zeros((self.model.Control_dim,)).T, i=i)
                except KeyError:
                    print(i)
                    print(self.Tracker.X.keys())
                    raise
                # Detection step
                SegMap = self.Segmentation.detect(image.data)
                Positions2D = self.Detector.detect(SegMap)

                Positions3D = []
                for pos in Positions2D:
                    posZ = Index_Image[int(pos.PositionX), int(pos.PositionY)]
                    Positions3D.append(PT_Measurement(pos.Log_Probability,
                                                      [pos.PositionX * res, pos.PositionY * res, posZ * 10],
                                                      frame=pos.Frame,
                                                      track_id=pos.Track_Id))
                Positions = Positions3D  # convenience

                # Setting Mask in ClickPoints
                if not db.getMaskType(name="PT_SegMask"):
                    mask_type = db.setMaskType(name="PT_SegMask", color="#FF59E3")
                m = db.setMask(image=image, data=(~SegMap).astype(np.uint8))
                print("Mask save", m)
                n = 1

                if np.all(Positions != np.array([])):

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
                            db.setMarker(image=image, x=x_img, y=y_img, text="Detection %s" % k,
                                         type=self.detection_marker_type)
                        except:
                            pass
                        # Write assigned tracks to ClickPoints DataBase
                        if np.isnan(x) or np.isnan(y):
                            pass
                        else:
                            pred_marker = db.setMarker(image=image, x=pred_x_img, y=pred_y_img,
                                                       text="Track %s" % (100 + k), type=self.track_marker_type)
                            if db.getTrack(k + 100):
                                track_marker = db.setMarker(image=image, type=self.track_marker_type, track=(100 + k),
                                                            x=x_img, y=y_img,
                                                            text='Track %s, Prob %.2f' % ((100 + k), prob))
                                print('Set Track(%s)-Marker at %s, %s' % ((100 + k), x_img, y_img))
                            else:
                                db.setTrack(self.track_marker_type, id=100 + k)
                                if k == self.Tracker.CriticalIndex:
                                    db.setMarker(image=image, type=self.track_marker_type, x=x_img, y=y_img,
                                                 text='Track %s, Prob %.2f, CRITICAL' % ((100 + k), prob))
                                track_marker = db.setMarker(image=image, type=self.track_marker_type,
                                                            track=100 + k,
                                                            x=x_img,
                                                            y=y_img,
                                                            text='Track %s, Prob %.2f' % ((100 + k), prob))
                                print('Set new Track %s and Track-Marker at %s, %s' % ((100 + k), x_img, y_img))

                            db.setMeasurement(marker=track_marker, log=prob, x=x, y=y, z=z)

                com.ReloadMarker(i+1)
                com.JumpToFrameWait(i+1)
                print("Got %s Filters" % len(self.Tracker.ActiveFilters.keys()))
                if not self.start_button.isChecked():
                    break


# create qt application
app = QtWidgets.QApplication(sys.argv)

# start window
window = PenguTrackWindow()
window.show()
app.exec_()

