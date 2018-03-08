#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PenguTrackDisplay.py

# Copyright (c) 2015-2016, Richard Gerum, Sebastian Richter
#
# This file is part of ClickPoints.
#
# ClickPoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ClickPoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ClickPoints. If not, see <http://www.gnu.org/licenses/>

from __future__ import division, print_function
import clickpoints
from qtpy import QtCore, QtGui, QtWidgets
import numpy as np

from PenguTrack.DataFileExtended import DataFileExtended

# a circle path
path_circle = QtGui.QPainterPath()
path_circle.addEllipse(-.5, -.5, 1, 1)

# a cross path
w = 1.
b = 7
r2 = 10
path1 = QtGui.QPainterPath()
path1.addRect(-r2, -w, b, w * 2)
path1.addRect(r2, -w, -b, w * 2)
path1.addRect(-w, -r2, w * 2, b)
path1.addRect(-w, r2, w * 2, -b)
path1.addEllipse(-r2, -r2, r2 * 2, r2 * 2)
path1.addEllipse(-r2, -r2, r2 * 2, r2 * 2)

class StateMarker(QtWidgets.QGraphicsPathItem):

    def __init__(self, parent, x, e, type="state"):
        # init and store parent
        QtWidgets.QGraphicsPathItem.__init__(self, parent)
        self.parent = parent
        self.type = type
        self.x = x
        self.e = e

        # set path
        self.setBrushes()
        if self.type == "state":
            self.setPath(self.sized_cross(e[0], e[1], np.linalg.norm(e)*0.1))
            self.setZValue(0.9)
        elif self.type == "prediction":
            self.setPath(self.sized_circ(e[0], e[1], 0))
            self.setZValue(0.1)
        elif self.type == "measurement":
            self.setPath(self.sized_cross(e[0], e[1], np.linalg.norm(e)*0.1))
            self.setZValue(0.2)
        self.setPos(x[0], x[1])

    def sized_cross(self, w, b, r):
        # a cross path
        path = QtGui.QPainterPath()
        path.addRect(-r/2., -w/2., r, w)
        path.addRect(-b/2., -r/2., b, r)
        return path

    def sized_circ(self, w, b, r):
        path_circle = QtGui.QPainterPath()
        path_circle.addEllipse(-w/2., -b/2., w, b)
        return path_circle

    def traingel_and_circ(self, w, b, w2,b2):
        path_circle = QtGui.QPainterPath()
        path_circle.addEllipse(-w/2., -b/2., w, b)
        path_circle.moveTo(w2/2.,0.)
        path_circle.lineTo()
        return path_circle

    def setBrushes(self):
        # set the bush to a gray color
        base_color = "808080"
        if self.type == "state":
            base_color = "00ff00"
        elif self.type == "prediction":
            base_color = "0000ff"
        elif self.type == "measurement":
            base_color = "ff0000"
        print(255./max(1.,self.e[0]*self.e[1]*np.pi))
        maximum = np.uint8(255./max(1., self.e[0]*self.e[1]*np.pi))
        maximum = 255
        color = "#" + hex(maximum)[2:] + base_color
        gradient = QtGui.QRadialGradient(0., 0., (self.e[0]+self.e[1])/2.)
        gradient.setColorAt(0., QtGui.QColor(color))
        gradient.setColorAt(np.exp(-1.), QtGui.QColor("#00000000"))
        self.setBrush(gradient)
        # self.setBrush(QtGui.QBrush(QtGui.QColor(color)))

        # set the pen to white with a border of 1
        pen = self.pen()
        # pen.setColor(QtGui.QColor("white"))
        pen.setColor(QtGui.QColor("#00000000"))
        pen.setCosmetic(True)
        # pen.setWidthF(0.)
        self.setPen(pen)

    def setColor(self, color):
        self.setBrush(QtGui.QBrush(QtGui.QColor(color)))

    def delete(self):
        self.scene().removeItem(self)


class Addon(clickpoints.Addon):
    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, database_class=DataFileExtended, **kwargs)

        # try to load the Track data from the PenguTrack/Clickpoints Database
        self.Tracker = self.db.tracker_from_db()[0]

        #initialize point lists
        self.points = []
        # load the current frame
        self.imageLoadedEvent("", self.cp.getCurrentFrame())

    def imageLoadedEvent(self, filename, framenumber):
        # clear the points from the last frame
        self.clearPoints()

        # only proceed if the frame is valid
        if framenumber is None:
            return

        # get the start frame
        if self.db.getOption(key="tracking_show_trailing") == -1:
            start = 0
        else:
            start = max([0, framenumber - self.db.getOption(key="tracking_show_trailing")])
        # get the end frame
        if self.db.getOption(key="tracking_show_leading") == -1:
            end = self.db.getImages().count()
        else:
            end = min([self.db.getImages().count(),
                       framenumber + self.db.getOption(key="tracking_show_leading")])

        for filter in self.Tracker.Filters.values():
            i_x = filter.Model.Measured_Variables.index("PositionY")
            i_y = filter.Model.Measured_Variables.index("PositionX")
            i = framenumber

            if i in filter.Predicted_X:
                P = filter.Model.measure(filter.Predicted_X[i])
                p = [P[i_x], P[i_y]]
                if i in filter.Predicted_X_error and filter.Predicted_X_error[i] is not None:
                    P_err = filter.Model.measure(filter.Predicted_X_error[i])
                    p_err = [P_err[i_x, i_x], P_err[i_y, i_y]]
                else:
                    p_err = [0., 0.]
                point_p = StateMarker(self.cp.window.view.origin, p, p_err, type="prediction")
                # point_p.setColor(QtGui.QColor(0, 0, 255))
                # point_p.setColor(QtGui.QBrush(QtGui.QColor("#88FF8800")))
                self.points.append(point_p)

            if i in filter.Measurements:
                M = filter.Measurements[i]
                m = [M.PositionY, M.PositionX]
                # m = [M[i_x], M[i_y]]
                m_err = np.diag(M.Covariance)
                point_m = StateMarker(self.cp.window.view.origin, m, m_err, type="measurement")
                # point_m.setColor(QtGui.QColor(255, 0, 0))
                self.points.append(point_m)

            if i in filter.X:
                X = filter.Model.measure(filter.X[i])
                x = [X[i_x], X[i_y]]
                if i in filter.X_error and filter.X_error[i] is not None:
                    X_err = filter.Model.measure(filter.X_error[i])
                    err = [X_err[i_x, i_x], X_err[i_y, i_y]]
                else:
                    err = [0., 0.]
                point_x = StateMarker(self.cp.window.view.origin, x, err, type="state")
                # point_x.setColor(QtGui.QColor(0, 255, 0))
                self.points.append(point_x)


            # x_old = None
            # err_old = None
            # for i in range(start, end+1):
            #     if i not in filter.X:
            #         continue
            #     X = filter.Model.measure(filter.X[i])
            #     x = [X[i_x],X[i_y]]
            #
            #     if i in filter.X_error and filter.X_error[i] is not None:
            #         X_err = filter.Model.measure(filter.X_error[i])
            #         err = [X_err[i_x,i_x],X_err[i_y,i_y]]
            #     else:
            #         err = [1.,1.]
            #
            #     point = MyPointItem(self.cp.window.view.origin,x, err)
            #     if x_old is not None:
            #         point.addLine(x_old)
            #
            #     print("bla", i, framenumber)
            #     if i==framenumber:
            #         point.setColor(QtGui.QColor(255, 0, 0))
            #     self.points.append(point)
            #
            #     x_old = np.array(x)
            #     err_old = np.array(err)

    def clearPoints(self):
        # delete all the points
        for point in self.points:
            point.delete()
        # and create a new empty array
        self.points = []

    def delete(self):
        # when the add-on is removed delete all points
        self.clearPoints()
        super(Addon, self).delete()

    def buttonPressedEvent(self):
        # do nothing when the button is pressed
        pass