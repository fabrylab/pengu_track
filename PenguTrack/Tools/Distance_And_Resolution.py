#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import my_plot
sn.set_color_codes()

my_plot.set_style("white_clickpoints")

import clickpoints
from PenguTrack.Detectors import SiAdViBeSegmentation
db_start = clickpoints.DataFile("/home/birdflight/Desktop/252_GT_Detections.cdb")

images = db_start.getImageIterator()
init_buffer = []
for i in range(10):
    while True:
        img = images.next().data
        if img is not None:
            # print("Got img from cam")
            init_buffer.append(img)
            # print(init_buffer[-1].shape)
            # print(init_buffer[-1].dtype)
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
VB = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3,9e-3], penguin_markers, 0.564, 500, init, n=1, n_min=1)
VB2 = SiAdViBeSegmentation(horizon_markers, 14e-3, [17e-3,9e-3], penguin_markers, 0.564, 500, init, n=1, n_min=1,log=False)
image_height = float(VB.height)
image_width = float(VB.width)
y_min = VB.y_min[1]
y_max = VB.y_max[1]
camera_height = VB.camera_h
sensor_width, sensor_height = VB.Sensor_Size
focal_length = VB.F
camera_tilt = VB.Phi
# corrected_log = VB.horizontal_equalisation(images.next().data)
# corrected = VB2.horizontal_equalisation(images.next().data)
db2 = clickpoints.DataFile("/home/birdflight/MasterarbeitAlex/Pictures/DataBase.cdb")
image = db2.getImage(0).data
corrected = db2.getImage(1).data
corrected_log = db2.getImage(2).data
penguin_size_type = db2.getMarkerType(name="Penguin_Size")
penguin_sizes = dict([[i.id, np.array([[m.x1, m.y1, m.x2, m.y2] for m in db2.getLines(type="Penguin_Size", image=i)])] for i in db2.getImages()])

pos_img = np.amin(penguin_sizes[1].T[1::2], axis=0)
size_img=np.linalg.norm(penguin_sizes[1].T[:2]-penguin_sizes[1].T[2:], axis=0)
pos_lin = np.amin(penguin_sizes[2].T[1::2], axis=0)
size_lin=np.linalg.norm(penguin_sizes[2].T[:2]-penguin_sizes[2].T[2:], axis=0)
pos_log = np.amin(penguin_sizes[3].T[1::2], axis=0)
size_log=np.linalg.norm(penguin_sizes[3].T[:2]-penguin_sizes[3].T[2:], axis=0)

# Plotting

d = np.arange(1,1000.)
s2 = 0.525
px2 = 2*np.arctan(s2/2./d)*image_height/(2*np.arctan(sensor_height/2./focal_length))
s3 = 0.3
px3 = 2*np.arctan(s3/2./d)*image_height/(2*np.arctan(sensor_height/2./focal_length))

fig, ax = plt.subplots()
ax.set_ylim(0,20)
ax.set_xlim(0,1000)
ax.set_ylabel("Object Size in Pixel")
ax.set_xlabel("Distance from Object to Camera")
ax.set_title("Object Sizes in Perspective Image")
my_plot.setAxisSizeMM(fig, ax, 147, 90)
ax.plot(d, px3, label=r"$\approx 0.3\mathrm{m}$ Width (theory)")
ax.plot(d, px2, label=r"$\approx 0.5\mathrm{m}$ Height (theory)")
ax.plot(d, np.ones_like(d)*2, label="detection threshold", color="gray")
ax.scatter(camera_height*np.tan(np.arctan((image_height/2.-pos_img)/image_height*sensor_height/focal_length) - camera_tilt + np.pi/2.), size_img, label=r"$\approx 0.5\mathrm{m}$ Height (measured)")
ax.legend(loc="best", prop={"size":12})
my_plot.despine(ax)
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_img.svg")
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_img.png")


s = np.arange(1,500.)
p = 0.3
h = camera_height
px = p/y_max*image_height
p2 = 0.5
px2 = (p2*s/(h-p2))/y_max*image_height/(2*np.arctan(sensor_height/2./focal_length))/2

fig, ax = plt.subplots()
ax.set_ylim(0,60)
ax.set_xlim(0,500)
ax.set_ylabel("Object Size in Pixel")
ax.set_xlabel("Distance from Object to Camera")
ax.set_title("Object Sizes in Orthoprojected Image")
my_plot.setAxisSizeMM(fig, ax, 147, 90)
ax.plot(s, px*np.ones_like(s), label=r"$\approx 0.3\mathrm{m}$ Width (theory)")
ax.plot(s, px2, label=r"$\approx 0.5\mathrm{m}$ Height (theory)")
ax.scatter(500-pos_lin*500./2592., size_lin, label=r"$\approx 0.5\mathrm{m}$Height (measured)")
ax.legend(loc="best", prop={"size":12})
my_plot.despine(ax)
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_lin.svg")
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_lin.png")


s = np.arange(1,500.)
px3 = 0.5/500*2592/(0.25*np.pi)
y_min=62.04
y_max=500.

fig, ax = plt.subplots()
ax.set_ylim(0,30)
ax.set_xlim(0,500)
ax.set_ylabel("Object Size in Pixel")
ax.set_xlabel("Distance from Object to Camera")
ax.set_title("Object Sizes in Log-Orthoprojected Image")
my_plot.setAxisSizeMM(fig, ax, 147, 90)
ax.plot(s, np.ones_like(s)*px3, label=r"$\approx 0.3\mathrm{m}$ Width (theory)")
ax.plot(s, np.ones_like(s)*VB.Penguin_Size, label=r"$\approx 0.5\mathrm{m}$ Height (theory)")
ax.scatter(y_min*(y_max/y_min)**(1.-pos_log/2592.), size_log, label=r"$\approx 0.5\mathrm{m}$ Height (measured)")
ax.legend(loc="upper left", prop={"size":12})
my_plot.despine(ax)
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_log.svg")
plt.savefig("/home/birdflight/Desktop/Adelie_DistanceResolution_log.png")

