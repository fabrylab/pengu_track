#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Detectors.py

# Copyright (c) 2016, Red Hulk Productions
#
# This file is part of PenguTrack
#
# PenguTrack is beer software: you can serve it and/or drink
# it under the terms of the Book of Revelation as published by
# the evangelist John, either version 3 of the Book, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. It may cause
# penguins to explode. It may also cause further harm on coworkers,
# advisors or even lab equipment. See the Book of Revelation for
# more details.
#
# You should have received a copy of the Book of Revelation
# along with PenguTrack. If not, see <http://trumpdonald.org/>

"""
Module containing detector classes to be used with pengu-track filters and models.
"""

from __future__ import division, print_function

import numpy as np
import skimage.feature
import skimage.transform
import skimage.measure
import skimage.color
import skimage.filters as filters
import skimage.morphology

from skimage import transform
from scipy.ndimage.interpolation import map_coordinates
from skimage import img_as_uint

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max


class Measurement(object):
    """
    Base Class for detection results.
    """
    def __init__(self, log_probability, position, frame=None, track_id=None):
        """
        Base Class for detection results.

        probability: float
            Estimated logarithmic probability of measurement.
        position: array_like
            Position of measurement
        frame: int, optional
            Number of Frame, on which the measurement took place.
        track_id: int, optional
            Track, for which this measurement was searched.
        """
        super(Measurement, self).__init__()

        self.Log_Probability = float(log_probability)
        try:
            self.PositionX, self.PositionY, self.PositionZ = np.asarray(position)
        except ValueError:
            try:
                self.PositionX, self.PositionY = np.asarray(position)
            except ValueError:
                self.PositionX = float(position)
        # self.Position = np.asarray(position)
        if track_id is not None:
            self.Track_Id = int(track_id)
        else:
            self.Track_Id = None

        if frame is not None:
            self.Frame = int(frame)
        else:
            self.Frame = None


class Detector(object):
    """
    This Class describes the abstract function of a detector in the pengu-track package.
    It is only meant for subclassing.
    """
    def __init__(self):
        super(Detector, self).__init__()

    def detect(self, image):
        return np.random.rand(2)


class BlobDetector(Detector):
    """
    Detector classifying objects by size and number to be used with pengu-track modules.
    """
    def __init__(self, object_size, object_number, threshold=None):
        """
        Detector classifying objects by size and number.

        Parameters
        ----------
        object_size: non-zero int
            Smallest diameter of an object to be detected.
        object_number: non-zero int
            If threshold in None, this number specifies the number of objects to track
            in the first picture and sets the threshold accordingly.
        threshold: float, optional
            Threshold for binning the image.
        """
        super(BlobDetector, self).__init__()
        self.ObjectSize = int(object_size)
        self.ObjectNumber = int(object_number)
        self.Threshold = threshold

    def detect(self, image, return_regions=False):
        """
        Detection function. Parts the image into blob-regions with size according to object_size.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.
        return_regions: bool, optional
            If True, function will return skimage.measure.regionprops object,
            else a list of the blob centroids and areas.

        Returns
        ----------
        regions: array_like
            List of information about each blob of adequate size.
        """
        image = np.array(image, dtype=int)
        while len(image.shape) > 2:
            image = np.linalg.norm(image, axis=-1)
        if self.Threshold is None:
            filtered = skimage.filters.laplace(skimage.filters.gaussian(image.astype(np.uint8), self.ObjectSize))
            filtermax = filtered.max()
            filtermin = filtered.min()
            n = 255
            for i in range(n):
                threshold = filtermin + (filtermax - filtermin) * (1. - i/(n-1.))
                regions = skimage.measure.regionprops(skimage.measure.label(filtered > threshold))
                if len(regions) > self.ObjectNumber:
                    self.Threshold = threshold
                    print("Set Blob-Threshold to %s" % threshold)
                    break
        regions = skimage.measure.regionprops(
                    skimage.measure.label(
                        skimage.filters.laplace(
                            skimage.filters.gaussian(image.astype(np.uint8), self.ObjectSize)) > self.Threshold
                        ))
        if return_regions:
            return regions
        elif len(regions) > 0:
            return [Measurement(1., [props.centroid[0], props.centroid[1]]) for props in regions]
            # return np.array([[props.centroid[0], props.centroid[1], props.area] for props in regions], ndmin=2)
        else:
            return []


class AreaBlobDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_size, object_number, threshold=None):
        """
        Detector classifying objects by number and size, taking area-separation into account.

        Parameters
        ----------
        object_size: non-zero int
            Smallest diameter of an object to be detected.
        object_number: non-zero int
            If threshold in None, this number specifies the number of objects to track
            in the first picture and sets the threshold accordingly.
        threshold: float, optional
            Threshold for binning the image.
        """
        super(AreaBlobDetector, self).__init__()
        self.ObjectSize = int(object_size)
        self.ObjectNumber = int(object_number)
        self.Threshold = threshold

    def detect(self, image, return_regions=False):
        """
        Detection function. Parts the image into blob-regions with size according to their area.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.
        return_regions: bool, optional
            If True, function will return skimage.measure.regionprops object,
            else a list of the blob centroids and areas.

        Returns
        -------
        regions: array_like
            List of information about each blob of adequate size.
        """
        while len(image.shape) > 2:
            image = np.linalg.norm(image, axis=-1)
        image = np.array(image, dtype=bool)
        regions = skimage.measure.regionprops(skimage.measure.label(image))
        if len(regions) <= 0:
            return np.array([])

        areas = np.asarray([props.area for props in regions])
        if self.Threshold is None:
            testareas = np.array(areas)
            filtered_max = areas.max()
            filtered_min = areas.min()
            n = int(float(filtered_max)/filtered_min)+1
            for i in range(n):
                threshold = filtered_min + (filtered_max - filtered_min) * (1-i/(n-1.))
                if (testareas[testareas > threshold]).shape[0] >= self.ObjectNumber:
                    self.Threshold = threshold
                    print("Set Blob-Threshold to %s and %s" % (np.pi*(self.ObjectSize/2.)**2, threshold))
                    break
            else:
                self.Threshold = filtered_max
                print("Set Blob-Threshold to %s and %s" % (np.pi*(self.ObjectSize/2.)**2, self.Threshold))

        mask = (areas >= self.Threshold)
        out = []

        if return_regions:
            for i, a in enumerate(areas):
                if mask[i]:
                    for j in range(int(a//(areas[mask].mean()))+1):
                        out.append(regions[i])
            return out

        else:
            for i, a in enumerate(areas):
                if mask[i]:
                    for j in range(int(a//(areas[mask].mean()))+1):
                        # out.append(regions[i].centroid)
                        out.append(Measurement(1., regions[i].centroid))
            # return np.array(out, ndmin=2)
            return out


class WatershedDetector(Detector):
    """
    Detector classifying objects by area and number. It uses watershed algorithms to depart bigger areas.
    To be used with pengu-track modules.
    """
    def __init__(self, object_size, object_number, threshold=None):
        """
        Detector classifying objects by number and size, taking area-separation into account.

        Parameters
        ----------
        object_size: non-zero int
            Smallest diameter of an object to be detected.
        object_number: non-zero int
            If threshold in None, this number specifies the number of objects to track
            in the first picture and sets the threshold accordingly.
        threshold: float, optional
            Threshold for binning the image.
        """
        super(WatershedDetector, self).__init__()
        self.ObjectSize = int(object_size)
        self.ObjectNumber = int(object_number)
        self.Threshold = threshold

    def detect(self, image, return_regions=False):
        """
        Detection function. Parts the image into blob-regions with size according to their area.
        Then departs bigger areas into smaller ones with watershed method.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.
        return_regions: bool, optional
            If True, function will return skimage.measure.regionprops object,
            else a list of the blob centroids and areas.

        Returns
        ----------
        list
            List of information about each blob of adequate size.
        """
        while len(image.shape) > 2:
            image = np.linalg.norm(image, axis=-1)
        image = np.array(image, dtype=bool)

        markers = ndi.label(peak_local_max(skimage.filters.laplace(skimage.filters.gaussian(image.astype(np.uint8),
                                           self.ObjectSize)),
                                           min_distance=self.ObjectSize,
                                           labels=image,
                                           indices=False))[0]
        labels = watershed(-image, markers, mask=image)

        regions = skimage.measure.regionprops(labels)

        if return_regions:
            return regions
        elif len(regions) > 0:
            return [Measurement(1., [props.centroid[0], props.centroid[1]]) for props in regions]
            # return np.array([[props.centroid[0], props.centroid[1], props.area] for props in regions], ndmin=2)
        else:
            return []


class Segmentation(object):
    """
    This Class describes the abstract function of a image-segmentation-algorithm in the pengu-track package.
    It is only meant for subclassing.
    """
    def __init__(self):
        super(Segmentation, self).__init__()

    def detect(self, image):
        return np.random.rand(2)


class MoGSegmentation(Segmentation):
    """
    Segmentation method assuming that pixel states depend on various gaussian distributions.
    """
    def __init__(self, n=10, init_image=None):
        """
        Segmentation method assuming that pixel states depend on k gaussian distributions.

        Parameters
        ----------
        n: int
            Number of gaussians for each pixel.

        init_image: array_like, optional
            Image for initialisation of background.

        """
        super(MoGSegmentation, self).__init__()

        # self.K = k
        self.N = n
        self.width = None
        self.height = None
        self.Skale = None

        self.Mu = None
        self.Var = None

        if init_image is not None:
            data = np.array(init_image, ndmin=2)
            selem = np.ones((3, 3))
            if len(data.shape) == 3:
                data_mean = np.asarray([filters.rank.mean(color_channel, selem) for color_channel in data.T]).T
                data_std = np.asarray([filters.rank.mean(color_channel**2, selem)
                                       for color_channel in data.T]).T - data_mean**2
                self.Mu = np.tile(data_mean, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
                self.Var = np.tile(data_std, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            elif len(data.shape) == 2:
                data_mean = filters.rank.mean(data, selem)
                data_std = filters.rank.mean(data**2, selem) - data_mean**2
                self.Mu = np.tile(data_mean, (self.N, 1, 1,))
                self.Var = np.tile(data_std, (self.N, 1, 1,))
            else:
                raise ValueError('False format of data.')

            self.Skale = np.mean(np.linalg.norm(self.Mu, axis=-1).astype(float))

        self.Mu = np.random.normal(self.Mu, np.sqrt(self.Var)+0.5)

    def detect(self, image):
        """
        Segmentation function. This function binearizes the input image by assuming the pixels,
        which do not fit the background gaussians are foreground.

        Parameters
        ----------
        image: array_like
            Image to be segmented.

        Returns
        ----------
        SegMap: array_like, bool
            Segmented Image.
        """
        super(MoGSegmentation, self).detect(image)

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        if self.Mu is None:
            selem = np.ones((3, 3))
            data_mean = filters.rank.mean(data, selem)
            data_std = filters.rank.mean(data**2, selem) - data_mean**2
            if len(data.shape) == 3:
                self.Mu = np.tile(data_mean, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
                self.Var = np.tile(data_std, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            elif len(data.shape) == 2:
                self.Mu = np.tile(data_mean, (self.N, 1, 1,))
                self.Var = np.tile(data_std, (self.N, 1, 1,))
            else:
                raise ValueError('False format of data.')

        dists = np.abs(self.Mu-data)*(1/self.Var**0.5)
        matchs = np.zeros_like(data, dtype=bool)
        matchs[np.unravel_index(np.argmin(dists, axis=0), data.shape)] = True
        # matchs = (dists == np.amin(dists, axis=0))
        outliers = (dists == np.amin(dists, axis=0))

        self.Mu[matchs] = (self.Mu[matchs] * ((self.N-1)/self.N) + data.ravel()*(1./self.N))
        self.Var[matchs] = (self.Var[matchs] * ((self.N-2)/(self.N-1)) + (data.ravel()-self.Mu[matchs]))*(1./(self.N-1))

        self.Mu[outliers] = np.mean(self.Mu, axis=0).ravel()
        self.Var[outliers] = np.mean(self.Var, axis=0).ravel()

        if len(data.shape) == 3:
            return np.sum(np.sum(dists < 4, axis=0), axis=-1) > 2
        elif len(data.shape) == 2:
            return np.sum(dists < 4, axis=0) > 2
        else:
            raise ValueError('False format of data.')


class ViBeSegmentation(Segmentation):
    """
    Segmentation method comparing input images to image-background buffer. Able to learn new background information.
    """
    def __init__(self, n=20, r=15, n_min=1, phi=16, init_image=None):
        """
        Segmentation method comparing input images to image-background buffer. Able to learn new background information.

        Parameters
        ----------
        n: int, optional
            Number of buffer frames.
        r: int, optional
            Distance-Treshold in standard color-room. If a input pixel deviates more than r from a background-buffer
            pixel it is counted as a deviation.
        n_min: int, optional
            Number of minimum deviations to count a pixel as foreground.
        phi: int, optional
            Inverse of update rate. Every phi-est foreground pixel will be updated to the background buffer.

        init_image: array_like, optional
            Image for initialisation of background.

        """
        super(ViBeSegmentation, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Skale = None
        self.Samples = None

        if init_image is not None:
            data = np.array(init_image, ndmin=2)
            if len(data.shape) == 3:
                self.Samples = np.tile(data, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            elif len(data.shape) == 2:
                self.Samples = np.tile(data, (self.N, 1, 1,))

            else:
                raise ValueError('False format of data.')
            self.Skale = np.mean((np.sum(np.mean(self.Samples, axis=0).astype(np.uint32)**2, axis=-1)**0.5))

        self.SegMap = None
        self._Neighbour_Map = {0: [-1, -1],
                               1: [-1, 0],
                               2: [-1, 1],
                               3: [0, -1],
                               4: [0, 1],
                               5: [1, -1],
                               6: [1, 0],
                               7: [1, 1]}
        self.width = None
        self.height = None

    def detect(self, image, do_neighbours=True):
        """
        Segmentation function. This compares the input image to the background model and returns a segmentation map.

        Parameters
        ----------
        image: array_like
            Input Image.
        do_neighbours: bool, optional
            If True neighbouring pixels will be updated accordingly to their foreground vicinity, else this
            time-intensiv calculation will not be done.


        Returns
        ----------
        SegMap: array_like, bool
            The segmented image.
        """
        super(ViBeSegmentation, self).detect(image)

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1])
                                   > self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        elif len(data.shape) == 2:
            self.SegMap = (np.sum((np.abs(self.Samples.astype(np.int32)-data) > self.R), axis=0, dtype=np.uint8)
                           >= self.N_min).astype(bool)
        else:
            raise ValueError('False format of data.')

        image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & self.SegMap

        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(np.uint8)

        n = np.sum(image_mask)
        if n > 0 and do_neighbours:
            x, y = np.array(np.meshgrid(np.arange(self.width), np.arange(self.height))).T[image_mask].T
            rand_x, rand_y = np.asarray(map(self._Neighbour_Map.get, np.random.randint(0, 8, size=n))).T
            rand_x += x
            rand_y += y
            notdoubled = ~(np.asarray([x_ in rand_x[:i] for i, x_ in enumerate(rand_x)]) &
                           np.asarray([y_ in rand_y[:i] for i, y_ in enumerate(rand_y)]))
            notborder = np.asarray(((0 <= rand_y) & (rand_y < self.height)) & ((0 <= rand_x) & (rand_x < self.width)))
            mask = notborder & notdoubled
            x = x[mask]
            y = y[mask]
            rand_x = rand_x[mask]
            rand_y = rand_y[mask]
            neighbours = np.zeros_like(image_mask, dtype=bool)
            neighbours[rand_x, rand_y] = True
            mask1 = np.zeros_like(image_mask, dtype=bool)
            mask1[x, y] = True
            try:
                self.Samples[sample_index][neighbours] = (data[mask1]).astype(np.uint8)
            except ValueError:
                print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)
                raise
        print("Updated %s pixels" % n)
        return self.SegMap

class SiAdViBeSegmentation(Segmentation):
    """
    Segmentation method comparing input images to image-background buffer. Able to learn new background information.
    This Version uses also Size Adjustion, an adapted ortho-projection,
     conserving the original size of an object in a plane.
    """
    def __init__(self, horizonmarkers, f, sensor_size, h, h_p, max_dist,  n=20, r=15, n_min=1, phi=16, init_image=None):
        """
        Segmentation method comparing input images to image-background buffer. Able to learn new background information.

        Parameters
        ----------
        horizonmarkers:
            List of at least 3 points at horizon in the image.
        f: float
            Focal length of the objective.
        sensor_size: array-like
            Sensor-size of the used camera.
        h: float
            Height of the camera over the shown plane.
        h_p: float
            Height of penguin / object in on the plane.
        max_dist: float, optional
            Maximum distance (distance from camera into direction of horizon) to be shown.

        n: int, optional
            Number of buffer frames.
        r: int, optional
            Distance-Treshold in standard color-room. If a input pixel deviates more than r from a background-buffer
            pixel it is counted as a deviation.
        n_min: int, optional
            Number of minimum deviations to count a pixel as foreground.
        phi: int, optional
            Inverse of update rate. Every phi-est foreground pixel will be updated to the background buffer.
        init_image: array_like, optional
            Image for initialisation of background.

        """
        super(SiAdViBeSegmentation, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Skale = None
        self.Samples = None
        self.Horizonmarkers = horizonmarkers
        self.F = f
        self.Sensor_Size = sensor_size
        self.H = h
        self.h_p = h_p
        self.Max_Dist = max_dist

        if init_image is not None:
            data = np.array(init_image, ndmin=2)
            data = self.horizontal_equalisation(data, self.Horizonmarkers, self.F, self.Sensor_Size, self.H, self.h_p,
                                                max_dist=self.Max_Dist)

            if len(data.shape) == 3:
                self.Samples = np.tile(data, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            elif len(data.shape) == 2:
                self.Samples = np.tile(data, (self.N, 1, 1,))

            else:
                raise ValueError('False format of data.')
            self.Skale = np.mean((np.sum(np.mean(self.Samples, axis=0).astype(np.uint32)**2, axis=-1)**0.5))

        self.SegMap = None
        self._Neighbour_Map = {0: [-1, -1],
                               1: [-1, 0],
                               2: [-1, 1],
                               3: [0, -1],
                               4: [0, 1],
                               5: [1, -1],
                               6: [1, 0],
                               7: [1, 1]}
        self.width = None
        self.height = None

    def detect(self, image, do_neighbours=True):
        """
        Segmentation function. This compares the input image to the background model and returns a segmentation map.

        Parameters
        ----------
        image: array_like
            Input Image.
        do_neighbours: bool, optional
            If True neighbouring pixels will be updated accordingly to their foreground vicinity, else this
            time-intensiv calculation will not be done.


        Returns
        ----------
        SegMap: array_like, bool
            The segmented image.
        """
        super(SiAdViBeSegmentation, self).detect(image)

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)

        data = self.horizontal_equalisation(data, self.Horizonmarkers, self.F, self.Sensor_Size, self.H, self.h_p,
                                            max_dist=self.Max_Dist)

        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1])
                                   > self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        elif len(data.shape) == 2:
            self.SegMap = (np.sum((np.abs(self.Samples.astype(np.int32)-data) > self.R), axis=0, dtype=np.uint8)
                           >= self.N_min).astype(bool)
        else:
            raise ValueError('False format of data.')

        image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & self.SegMap

        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(np.uint8)

        n = np.sum(image_mask)
        if n > 0 and do_neighbours:
            x, y = np.array(np.meshgrid(np.arange(self.width), np.arange(self.height))).T[image_mask].T
            rand_x, rand_y = np.asarray(map(self._Neighbour_Map.get, np.random.randint(0, 8, size=n))).T
            rand_x += x
            rand_y += y
            notdoubled = ~(np.asarray([x_ in rand_x[:i] for i, x_ in enumerate(rand_x)]) &
                           np.asarray([y_ in rand_y[:i] for i, y_ in enumerate(rand_y)]))
            notborder = np.asarray(((0 <= rand_y) & (rand_y < self.height)) & ((0 <= rand_x) & (rand_x < self.width)))
            mask = notborder & notdoubled
            x = x[mask]
            y = y[mask]
            rand_x = rand_x[mask]
            rand_y = rand_y[mask]
            neighbours = np.zeros_like(image_mask, dtype=bool)
            neighbours[rand_x, rand_y] = True
            mask1 = np.zeros_like(image_mask, dtype=bool)
            mask1[x, y] = True
            try:
                self.Samples[sample_index][neighbours] = (data[mask1]).astype(np.uint8)
            except ValueError:
                print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)
                raise
        print("Updated %s pixels" % n)
        return self.SegMap

    def horizontal_equalisation(self, image, horizonmarkers, f, sensor_size, h, h_p, markers=[], max_dist=None):
        """
        Parameters:
        ----------
        image: array-like
            Image.
        horizonmarkers:
            List of at least 3 points at horizon in the image.
        f: float
            Focal length of the objective.
        sensor_size: array-like
            Sensor-size of the used camera.
        h: float
        Height of the camera over the shown plane.
        h_p: float
        Height of penguin / object in on the plane.
        markers: array-like, optional
            List of markers in the image, which should also be transformed
        max_dist: float, optional
            Maximum distance (distance from camera into direction of horizon) to be shown.

        Returns:
        ----------
        image: array-like
            Equalised image.
        new_markers: array-like
            Equalised markers.
        """
        if not max_dist:
            max_dist = np.sqrt(2*6371e3*h+h**2)
        x_h, y_h = np.asarray(horizonmarkers)
        # Rotate image
        image = np.asarray(image)#[::-1, ::-1]

        Y, X = image.shape[:2]

        # Rotate other markers
        if np.any(markers):
            markers = np.array(markers)
            markers.T[0] = X - markers.T[0]
            markers.T[1] = Y - markers.T[1]

        # linear fit and rotation to compensate incorrect camera alignment
        m, t = np.polyfit(x_h, y_h, 1)  # linear fit
        image = np.array(image, ndmin=2, dtype=np.uint8)
        image = transform.rotate(image, np.arctan(m) * 180 / np.pi)

        X_s, Y_s = sensor_size
        # correction of the Y-axis section (after image rotation)
        t += m * X / 2
        print(t)

        # Calculate Camera tilt
        print(Y, Y_s, X, X_s, f, t)
        Phi = np.arctan((t * (2. / Y) - 1.) * (Y_s / 2. / f))*-1
        print("Phi at %s" % (Phi * 180 / np.pi))

        # Initialize grid
        xx, yy = np.meshgrid(np.arange(X), np.arange(Y))
        # Grid has same aspect ratio as picture, but width max_dist
        yy = yy * (max_dist/Y)
        xx = (xx-X/2.) * (max_dist/Y)
        # counter-angle to phi is used in further calculation
        phi_ = np.pi/2 - Phi
        # x_s is the crossing-Point of camera mid-beam and grid plane
        x_s = np.asarray([0, np.tan(phi_)*h, -h])
        x_s_norm = np.linalg.norm(x_s)
        # vector of maximal distance to camera (in y)
        y_max = np.asarray([0, max_dist, -h])
        y_max_norm = np.linalg.norm(y_max)
        y_min = np.asarray([0, h*np.tan(phi_-np.arctan(Y_s/f)),-h])
        alpha_y = np.arccos(np.dot(y_max.T, x_s).T/(y_max_norm*x_s_norm)) * -1

        # Define Warp Function
        def warp2(xy):
            xx_, yy_ = xy
            old = np.array(yy_)
            # vectors of every grid point in the plane (-h)
            print(max_dist/Y)
            c = 1./(h/h_p-1.)
            yy_ /= max_dist # 0 bis 1
            yy_ = yy * np.log(max_dist/y_min[1])
            print(np.amin(yy_), np.amax(yy_))

            yy_ = y_min[1]*np.exp(yy_)
            p_s = np.log(1+c)/(np.log(max_dist/y_min[1])/Y)
            print("Penguin size is %s pixels!"%p_s)
            print(np.amin(yy_), np.amax(yy_))

            coord = np.asarray([xx_, yy_, -h*np.ones_like(xx_)])
            coord_norm = np.linalg.norm(coord, axis=0)
            # calculate the angle between camera mid-beam-vector and grid-point-vector
            alpha = np.arccos(np.dot(coord.T, x_s).T/(coord_norm*x_s_norm)) #* np.sign(np.tan(phi_)*h-yy_)
            # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
            theta = np.arccos(np.sum((np.cross(coord.T, x_s) * np.cross(y_max, x_s)).T
                              , axis=0)/(coord_norm*x_s_norm*np.sin(alpha) *
                                         y_max_norm*x_s_norm*np.sin(alpha_y))) * np.sign(xx) #* np.sign(np.tan(phi_)*h-yy_)
            # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
            r = np.tan(alpha)*f
            warp_xx = np.sin(theta)*r*X/X_s + X/2.
            warp_yy = np.cos(theta)*r*Y/Y_s + Y/2.
            return warp_xx, warp_yy

        # warp the grid
        warped_xx, warped_yy = warp2([xx, yy])
        # reshape grid points for image interpolation
        grid = np.asarray([warped_xx.T, warped_yy.T]).T
        grid = grid.T.reshape((2, X * Y))

        if len(image.shape) == 3:
            # split the image in colors and perform interpolation
            return np.asarray([map_coordinates(i, grid[:, ::-1]).reshape((X, Y))[::-1] for i in image.T]).T
        elif len(image.shape) == 2:
            return np.asarray(img_as_uint(map_coordinates(image.T, grid[:, ::-1])).reshape((X, Y))[::-1]).T
        else:
            raise ValueError("The given image is whether RGB nor greyscale!")
