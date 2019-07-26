#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Detectors.py

# Copyright (c) 2016-2017, Alexander Winterl
#
# This file is part of PenguTrack
#
# PenguTrack is free software: you can redistribute and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the Licensem, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PenguTrack. If not, see <http://www.gnu.org/licenses/>.
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
from skimage import img_as_uint
from skimage.measure import regionprops

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import shift

from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import label
from scipy import stats
from scipy.ndimage.filters import gaussian_filter

from skimage.morphology import watershed
from skimage.feature import peak_local_max

from .Parameters import Parameter, ParameterList

import pandas

try:
    from skimage.filters import threshold_niblack
except IOError:
    threshold_niblack = lambda image: image > filters.threshold_otsu(image)
except ImportError:
    threshold_niblack = lambda image: image > filters.threshold_otsu(image)

try:
    import cv2
except ImportError:
    print("No CV2 found. Optical Flow Detectors not usable!")


class dotdict(dict):
    """
    enables dot access on dicts
    """
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        if attr not in self:
            raise AttributeError
        return self.get(attr, None)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def measurements_to_pandasDF(measurement_list):
    entries = ["Log_Probability", "PositionX", "PositionY", "PositionZ", "Covariance", "Frame"]
    extra_keys = set()
    for m in measurement_list:
        entries = [e for e in entries if e in m.keys()]
        extra_keys.update(m.Data.keys())
    entries.extend(extra_keys)
    return pandas.DataFrame([[m.get(e, None) for e in entries] for m in measurement_list],
                            columns=entries)


def measurements_to_array(measurements):
    entries = ["Log_Probability", "PositionX", "PositionY", "PositionZ", "Covariance", "Frame"]
    extra_keys = set()
    for m in measurement_list:
        entries = [e for e in entries if e in m.keys()]
        extra_keys.update(m.Data.keys())
    entries.extend(extra_keys)
    return np.array([[m[e] for e in entries] for m in measurements], dtype=float)


def pandasDF_to_array(DF):
    entries = ["Log_Probability", "PositionX", "PositionY", "PositionZ", "Covariance", "Frame"]
    entries = [e for e in entries if e in DF.columns]
    entries.extend([e for e in DF.columns if e not in entries])
    return np.vstack([DF[e] for e in entries]).T


def pandasDF_to_measurement(DF):
    dims = []
    for k in ["X", "Y", "Z"]:
        if "Position%s"%k in DF.columns:
            dims.append("Position%s"%k)
    keys = set(DF.columns).difference(dims)
    keys = keys.difference(["Log_Probability"])
    if "Log_Probability" in DF.columns:
        return [Measurement(d["Log_Probability"],
                            [d[k] for k in dims],
                            data=dict([[k, d[k]] for k in keys])) for i, d in DF.iterrows()]
    else:
        return [Measurement(1.,
                            [d[k] for k in dims],
                            data=dict([[k, d[k]] for k in keys])) for i, d in DF.iterrows()]


def array_to_pandasDF(array, keys=[], dim=2):
    array = np.asarray(array, dtype=float)
    if len(array.shape) != 2:
        raise ValueError("Array shape does not fit!")

    s = array.shape[1]
    n = array.shape[0]
    if s == dim:
        entries = ["PositionX", "PositionY", "PositionZ"][:dim]
    elif s == dim+1:
        entries = ["Log_Probability","PositionX", "PositionY", "PositionZ"][:1+dim]
    elif s == dim+len(keys):
        entries = ["PositionX", "PositionY", "PositionZ"][:dim]
        entries.extend(keys)
    elif s == dim+len(keys)+1:
        entries = ["Log_Probability", "PositionX", "PositionY", "PositionZ"][:1+dim]
        entries.extend(keys)
    elif s == len(keys):
        entries = keys
    else:
        raise ValueError("Can not interpret input array!")

    if "Log_Probability" not in entries:
        return pandas.DataFrame(np.hstack((np.zeros((n, 1)), array)),
                            columns=entries)
    else:
        return pandas.DataFrame(array, columns=entries)


def array_to_measurement(array, keys=[], dim=2):
    array = np.asarray(array, dtype=float)
    if len(array.shape) != 2:
        raise ValueError("Array shape does not fit!")

    s = array.shape[1]
    n = array.shape[0]
    if s == dim:
        entries = ["PositionX", "PositionY", "PositionZ"][:dim]
    elif s == dim+1:
        entries = ["Log_Probability","PositionX", "PositionY", "PositionZ"][:1+dim]
    elif s == dim+len(keys):
        entries = ["PositionX", "PositionY", "PositionZ"][:dim]
        entries.extend(keys)
    elif s == dim+len(keys)+1:
        entries = ["Log_Probability", "PositionX", "PositionY", "PositionZ"][:1+dim]
        entries.extend(keys)
    elif s == len(keys):
        entries = keys
    else:
        raise ValueError("Can not interpret input array!")
    dim_names = ["PositionX", "PositionY", "PositionZ"][:dim]
    non_dim_entries = [e for e in entries if (e not in dim_names and e!="Log_Probability")]

    if "Log_Probability" in entries:
        return [Measurement(0., [a[entries.index(d)] for d in dim_names],
                            data=dict([[e, a[entries.index(e)]] for e in non_dim_entries])) for a in array]
    else:
        return [Measurement(a[entries.index("Log_Probability")],
                            [a[entries.index(d)] for d in dim_names],
                            data=dict([[e, a[entries.index(e)]] for e in non_dim_entries])) for a in array]


class Measurement(dotdict):
    """
    Base Class for detection results.
    """
    def __init__(self, log_probability, position, cov=None, data=None, frame=None, track_id=None):
        """
        Base Class for detection results.

        probability: float
            Estimated logarithmic probability of measurement.
        position: array_like
            Position of measurement
        cov: array_like
            Covariance matrix of the position or measurement errors
        data: dict or array_like
            Additional data of the measurement
        frame: int, optional
            Number of Frame, on which the measurement took place.
        track_id: int, optional
            Track, for which this measurement was searched.
        """
        super(Measurement, self).__init__()

        self.Log_Probability = float(log_probability)
        try:
            self.PositionX, self.PositionY, self.PositionZ = np.asarray(position)
            self.dim = 3
        except ValueError:
            try:
                self.PositionX, self.PositionY = np.asarray(position)
                self.dim = 2
            except ValueError:
                self.PositionX = float(position)
                self.dim = 1
        if track_id is not None:
            self.Track_Id = int(track_id)
        else:
            self.Track_Id = None

        if cov is not None:
            cov = np.array(cov)
            if len(cov.shape)<1:
                self.Covariance = np.ones(len(position))*float(cov)
            elif len(cov.shape)<2:
                self.Covariance = np.diag(cov[:len(position)])
            else:
                self.Covariance = cov
        else:
            self.Covariance = np.ones(len(position))

        if frame is not None:
            self.Frame = int(frame)
        else:
            self.Frame = None

        self.Data = data
        if type(self.Data) is dict:
            for k in self.Data:
                if not k in self:
                    self.update({k: self.Data[k]})
                else:
                    raise ValueError("Key %s already used for inner argument. Change key!")


    def getPosition(self):
        try:
            return np.array([self.PositionX, self.PositionY, self.PositionZ], dtype=float)
        except AttributeError:
            try:
                return np.array([self.PositionX, self.PositionY], dtype=float)
            except ValueError:
                return np.array([self.PositionX], dtype=float)

    def getVector(self, keys):
        return [self[k] for k in keys]

    def getEntryKeys(self):
        keys = ["Log_Probability"]
        keys.extend(["PositionX","PositionY", "PositionZ"][:self.dim])
        keys.extend(self.Data.keys())
        return keys

    def getEntries(self):
        keys = self.getEntryKeys()
        return np.asarray([self[k] for k in keys], dtype=float)


def detection_parameters(**detection_parameters):
    def real_decorator(function):
        function.detection_parameters = detection_parameters
        return function
    return real_decorator


class Detector(object):
    """
    This Class describes the abstract function of a detector in the pengu-track package.
    It is only meant for subclassing.
    """
    def __init__(self):
        super(Detector, self).__init__()
        self.ParameterList = ParameterList()

    def detect(self, *args, **kwargs):
        return pandas.DataFrame(np.random.rand(2), columns=["PositionX", "PositionY"])
        # return Measurement(1., np.random.rand(2))

    def __getattribute__(self, item):
        if item != "ParameterList":
            if item in self.ParameterList.keys():
                return self.ParameterList[item]
        return super(Detector, self).__getattribute__(item)

    def __setattr__(self, key, value):
        if key != "ParameterList" and key in self.ParameterList.keys():
            self.ParameterList[key] = value
        else:
            return super(Detector, self).__setattr__(key, value)


class BlobDetector(Detector):
    """
    Detector classifying objects by size and number to be used with pengu-track modules.
    """
    def __init__(self, object_size=1, object_number=1, threshold=None):
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
        return_regions: bool, optional
            If True, function will return skimage.measure.regionprops object,
            else a list of the blob centroids and areas.
        """
        super(BlobDetector, self).__init__()
        self.ObjectSize = int(object_size)
        self.ObjectNumber = int(object_number)
        self.Threshold = threshold
        self.ParameterList = ParameterList(Parameter("ObjectSize", object_size, min=0, value_type=int),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Threshold", threshold, value_type=float)
                                           )

    @detection_parameters(image=dict(frame=0, mask=False))
    def detect(self, image):
        """
        Detection function. Parts the image into blob-regions with size according to object_size.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.

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
        out = pandas.DataFrame([[props.centroid[0], props.centroid[1], 1.] for props in regions],
                                columns=["PositionX", "PositionY", "Log_Probability"])
        # out = [Measurement(1., props.centroid) for props in regions]

        return out, None

class SimpleAreaDetector2(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area=1,
                 object_number=1,
                 threshold=None,
                 lower_limit=None,
                 upper_limit=None,
                 distxy_boundary=10,
                 distz_boundary=21):
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
        super(SimpleAreaDetector2, self).__init__()
        self.distxy_boundary = distxy_boundary
        self.distz_boundary = distz_boundary
        self.start_ObjectArea = int(object_area)
        self.sample_size = 1
        self.ObjectArea = int(object_area)
        self.ObjectNumber = object_number
        if lower_limit:
            self.LowerLimit = int(lower_limit)
        else:
            self.LowerLimit = int(0.4*self.ObjectArea)
        if upper_limit:
            self.UpperLimit = int(upper_limit)
        else:
            self.UpperLimit = int(1.6*self.ObjectArea)

        self.Threshold = threshold
        self.Sigma = np.sqrt((self.UpperLimit-self.LowerLimit)/(4*np.log(1./0.95)))

        self.ParameterList = ParameterList(Parameter("ObjectArea", object_area, min=0., value_type=float),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Sigma", self.Sigma, min=0, value_type=float),
                                           Parameter("Threshold", self.Threshold, value_type=float),
                                           Parameter("LowerLimit", self.LowerLimit, value_type=float),
                                           Parameter("UpperLimit", self.UpperLimit, value_type=float),
                                           Parameter("distxy_boundary", self.distxy_boundary, value_type=float),
                                           Parameter("distz_boundary", self.distz_boundary, value_type=float)
                                           )


    @detection_parameters(image=dict(frame=0, mask=False),
                          mask=dict(frame=0, mask=True))
    def detect(self, image, mask):
        """
        Detection function. Parts the image into blob-regions with size according to their area.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.

        Returns
        -------
        regions: array_like
            List of information about each blob of adequate size.
        """
        image[mask] = np.zeros_like(image)[mask]
        j_max = np.amax(image)
        stack = np.zeros((j_max, image.shape[0], image.shape[1]), dtype=np.bool)
        for j in range(j_max):
            stack[j, image == j] = True

        labels, n = label(stack, structure=np.ones((3, 3, 3)))

        index_data2 = np.zeros_like(image)
        for l in labels:
            index_data2[l > 0] = l[l > 0]

        while len(index_data2.shape) > 2:
            index_data2 = np.linalg.norm(index_data2, axis=-1)

        labeled = skimage.measure.label(index_data2, connectivity=2)

        # Filter the regions for area and convexity
        regions_list = [prop for prop in skimage.measure.regionprops(labeled, intensity_image=image)
                       if (self.UpperLimit > prop.area > self.LowerLimit)]
        if len(regions_list) <= 0:
            return np.array([])

        print("Object Area at ", self.ObjectArea, "pm", self.Sigma)

        out = []
        sigma = self.Sigma
        mu = self.ObjectArea
        for prop in regions_list:
            prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                            prop.area - mu) / sigma) ** 2

            intensities = prop.intensity_image[prop.image]
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            out.append([prop.centroid[0], prop.centroid[1], mean_int, prob, std_int])

        out = pandas.DataFrame(out, columns=["PositionX",
                                             "PositionY",
                                             "PositionZ",
                                             "Log_Probability",
                                             "IntensitySTD"])
        # out = [Measurement(o[3], o[:3], data={"IntensitySTD":o[4]}) for o in out]

        Positions2D = out
        Positions2D_cor = []
        for i1, pos1 in enumerate(Positions2D):
            Log_Probability1 = pos1.Log_Probability
            x1 = pos1.PositionX
            y1 = pos1.PositionY
            z1 = pos1.PositionZ
            inc = 0
            for j1, pos2 in enumerate(Positions2D):
                Log_Probability2 = pos2.Log_Probability
                Log_Probability = (Log_Probability1 + Log_Probability2)/2.
                x2 = pos2.PositionX
                y2 = pos2.PositionY
                z2 = pos2.PositionZ

                PosZ = (z1 + z2)/2.
                dist = np.sqrt((x1 - x2) ** 2. + (y1 - y2) ** 2.)
                distz = np.abs(z1 - z2)*10.
                if dist < self.distxy_boundary and dist != 0 and distz < self.distz_boundary:
                    x3 = (x1 + x2) / 2.
                    y3 = (y1 + y2) / 2.
                    if [x3, y3, Log_Probability, PosZ] not in Positions2D_cor:
                        Positions2D_cor.append([x3, y3, Log_Probability, PosZ])
                        print("Replaced")
                    inc += 1
            if inc == 0:
                Positions2D_cor.append([x1, y1, Log_Probability1, z1])

        Positions3D = []
        for pos in Positions2D_cor:
            Positions3D.append([pos[0], pos[1], pos[3], pos[2]])

        Positions3D = pandas.DataFrame(Positions3D, columns=["PositionX", "PositionY", "PositionZ", "Log_Probability"])
        # Positions3D = [Measurement(o[3], o[:3]) for o in out]

        return Positions3D, None


class SimpleAreaDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area=1,
                 object_number=1,
                 threshold=None,
                 lower_limit=None,
                 upper_limit=None):
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
        super(SimpleAreaDetector, self).__init__()
        self.start_ObjectArea = int(object_area)
        self.sample_size = 1
        self.ObjectArea = int(object_area)
        self.ObjectNumber = object_number
        if lower_limit:
            self.LowerLimit = int(lower_limit)
        else:
            self.LowerLimit = int(0.4*self.ObjectArea)
        if upper_limit:
            self.UpperLimit = int(upper_limit)
        else:
            self.UpperLimit = int(1.6*self.ObjectArea)

        self.Threshold = threshold
        self.Sigma = np.sqrt((self.UpperLimit-self.LowerLimit)/(4*np.log(1./0.95))) # self.ObjectArea/2.


        self.ParameterList = ParameterList(Parameter("ObjectArea", object_area, min=0., value_type=float),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Sigma", self.Sigma, min=0, value_type=float),
                                           Parameter("Threshold", self.Threshold, value_type=float),
                                           Parameter("LowerLimit", self.LowerLimit, value_type=float),
                                           Parameter("UpperLimit", self.UpperLimit, value_type=float)
                                           )

    @detection_parameters(image=dict(frame=0, mask=False))
    def detect(self, image):
        """
        Detection function. Parts the image into blob-regions with size according to their area.
        Returns information about the regions.

        Parameters
        ----------
        image: array_like
            Image will be converted to uint8 Greyscale and then binnearized.

        Returns
        -------
        regions: array_like
            List of information about each blob of adequate size.
        """
        while len(image.shape) > 2:
            image = np.linalg.norm(image, axis=-1)

        labeled = skimage.measure.label(image, connectivity=2)

        # Filter the regions for area and convexity
        regions_list = [prop for prop in skimage.measure.regionprops(labeled)
                       if (self.UpperLimit >= prop.area >= self.LowerLimit and prop.solidity > 0.5)]
        if len(regions_list) <= 0:
            return np.array([])

        print("Object Area at ", self.ObjectArea, "pm", self.Sigma)

        out = []
        sigma = self.Sigma
        mu = self.ObjectArea
        for prop in regions_list:
            prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                            prop.area - mu) / sigma) ** 2
            o = list(prop.centroid)
            o.append(prob)
            out.append(o)
        out = pandas.DataFrame(out, columns=["PositionX", "PositionY", "Log_Probability"])
        # out = [Measurement(o[2], o[:2]) for o in out]
        return out, None


def extended_regionprops(label_image, intensity_image=None, cache=True):
    """Measure properties of labeled image regions.

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labeled input image. Labels with value 0 are ignored.
    intensity_image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.

    Returns
    -------
    properties : list of RegionProperties
        Each item describes one labeled region, and can be accessed using the
        attributes listed below.

    Notes
    -----
    The following properties can be accessed as attributes or keys:

    **area** : int
        Number of pixels of region.
    **bbox** : tuple
        Bounding box ``(min_row, min_col, max_row, max_col)``.
        Pixels belonging to the bounding box are in the half-open interval
        ``[min_row; max_row)`` and ``[min_col; max_col)``.
    **bbox_area** : int
        Number of pixels of bounding box.
    **centroid** : array
        Centroid coordinate tuple ``(row, col)``.
    **convex_area** : int
        Number of pixels of convex hull image.
    **convex_image** : (H, J) ndarray
        Binary convex hull image which has the same size as bounding box.
    **coords** : (N, 2) ndarray
        Coordinate list ``(row, col)`` of the region.
    **eccentricity** : float
        Eccentricity of the ellipse that has the same second-moments as the
        region. The eccentricity is the ratio of the focal distance
        (distance between focal points) over the major axis length.
        The value is in the interval [0, 1).
        When it is 0, the ellipse becomes a circle.
    **equivalent_diameter** : float
        The diameter of a circle with the same area as the region.
    **euler_number** : int
        Euler characteristic of region. Computed as number of objects (= 1)
        subtracted by number of holes (8-connectivity).
    **extent** : float
        Ratio of pixels in the region to pixels in the total bounding box.
        Computed as ``area / (rows * cols)``
    **filled_area** : int
        Number of pixels of filled region.
    **filled_image** : (H, J) ndarray
        Binary region image with filled holes which has the same size as
        bounding box.
    **image** : (H, J) ndarray
        Sliced binary region image which has the same size as bounding box.
    **inertia_tensor** : (2, 2) ndarray
        Inertia tensor of the region for the rotation around its mass.
    **inertia_tensor_eigvals** : tuple
        The two eigen values of the inertia tensor in decreasing order.
    **intensity_image** : ndarray
        Image inside region bounding box.
    **label** : int
        The label in the labeled input image.
    **local_centroid** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box.
    **major_axis_length** : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.
    **max_intensity** : float
        Value with the greatest intensity in the region.
    **mean_intensity** : float
        Value with the mean intensity in the region.
    **min_intensity** : float
        Value with the least intensity in the region.
    **minor_axis_length** : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.
    **moments** : (3, 3) ndarray
        Spatial moments up to 3rd order::

            m_ji = sum{ array(x, y) * x^j * y^i }

        where the sum is over the `x`, `y` coordinates of the region.
    **moments_central** : (3, 3) ndarray
        Central moments (translation invariant) up to 3rd order::

            mu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }

        where the sum is over the `x`, `y` coordinates of the region,
        and `x_c` and `y_c` are the coordinates of the region's centroid.
    **moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant).
    **moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) up to 3rd order::

            nu_ji = mu_ji / m_00^[(i+j)/2 + 1]

        where `m_00` is the zeroth spatial moment.
    **orientation** : float
        Angle between the X-axis and the major axis of the ellipse that has
        the same second-moments as the region. Ranging from `-pi/2` to
        `pi/2` in counter-clockwise direction.
    **perimeter** : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.
    **solidity** : float
        Ratio of pixels in the region to pixels of the convex hull image.
    **weighted_centroid** : array
        Centroid coordinate tuple ``(row, col)`` weighted with intensity
        image.
    **weighted_local_centroid** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box, weighted with intensity image.
    **weighted_moments** : (3, 3) ndarray
        Spatial moments of intensity image up to 3rd order::

            wm_ji = sum{ array(x, y) * x^j * y^i }

        where the sum is over the `x`, `y` coordinates of the region.
    **weighted_moments_central** : (3, 3) ndarray
        Central moments (translation invariant) of intensity image up to
        3rd order::

            wmu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }

        where the sum is over the `x`, `y` coordinates of the region,
        and `x_c` and `y_c` are the coordinates of the region's weighted
        centroid.
    **weighted_moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant) of intensity
        image.
    **weighted_moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) of intensity
        image up to 3rd order::

            wnu_ji = wmu_ji / wm_00^[(i+j)/2 + 1]

        where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).

    Each region also supports iteration, so that you can do::

      for prop in region:
          print(prop, region[prop])
    """

    label_image = np.squeeze(label_image)

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integral type.')

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = ExtendedRegionProps(sl, label, label_image, intensity_image,
                                  cache)
        regions.append(props)

    return regions

import skimage.measure._regionprops as REGIONPROPS
from functools import wraps

def _cached(f):
    @wraps(f)
    def wrapper(obj):
        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper

class ExtendedRegionProps(REGIONPROPS._RegionProperties):
    def __init__(self,*args, **kwargs):
        if "coordinates" not in kwargs:
            kwargs.update({"coordinates": None})
        super(ExtendedRegionProps, self).__init__(*args, **kwargs)

    # @_cached
    def _surrounding_image(self):
        slicex, slicey = self._slice
        int_im = np.copy(self._intensity_image[slicex.start-2:slicex.stop+2,
                         slicey.start-2:slicey.stop+2])
        if int_im.shape[0] != slicex.stop - slicex.start+4 or \
                        int_im.shape[1] != slicey.stop -slicey.start +4 :
            int_im = np.copy(self._intensity_image[self._slice])
            image = self.image
        else:
            image = np.zeros_like(int_im, dtype=bool)
            image[2:-2,2:-2] = self.image
        int_im[~image] = 0
        return int_im

    # @_cached
    def _inside_image(self):
        slicex, slicey = self._slice
        int_im = np.copy(self._intensity_image[slicex.start-2:slicex.stop+2,
                         slicey.start-2:slicey.stop+2])
        if int_im.shape[0] != slicex.stop - slicex.start+4 or \
                        int_im.shape[1] != slicey.stop -slicey.start +4 :
            int_im = np.copy(self._intensity_image[self._slice])
            image = self.image
        else:
            image = np.zeros_like(int_im, dtype=bool)
            image[2:-2,2:-2] = self.image
        int_im[image] = 0
        return int_im

    def _full_image(self):
        return np.copy(self._intensity_image[self._slice])

    def _oversize_image(self, o):
        o=int(o)
        if o<1:
            return np.copy(self._intensity_image[self._slice])

        slicex, slicey = self._slice
        int_im = np.copy(self._intensity_image[slicex.start-o:slicex.stop+o,
                         slicey.start-o:slicey.stop+o])
        if int_im.shape[0] != slicex.stop - slicex.start+2*o or \
                        int_im.shape[1] != slicey.stop -slicey.start + 2*o :
            return self._oversize_image(o-1)
        else:
            return int_im

    def _oversize_label(self, o):
        o=int(o)
        if o<1:
            return np.copy(self._label_image[self._slice])

        slicex, slicey = self._slice
        int_im = np.copy(self._label_image[slicex.start-o:slicex.stop+o,
                         slicey.start-o:slicey.stop+o])
        if int_im.shape[0] != slicex.stop - slicex.start+2*o or \
                        int_im.shape[1] != slicey.stop -slicey.start + 2*o :
            return self._oversize_image(o-1)
        else:
            return int_im

    def sur_std(self):
        return np.std(self._surrounding_image())

    def sur_mu(self):
        return np.mean(self._surrounding_image())

    def in_max(self):
        return np.nanmax(self._inside_image())

    def in_min(self):
        return np.nanmin(self._inside_image())

    def sur_max(self):
        return np.nanmax(self._surrounding_image())

    def sur_min(self):
        return np.nanmin(self._surrounding_image())

    def in_std(self):
        return np.std(self._inside_image())

    def in_mu(self):
        return np.mean(self._inside_image())

    def in_out_contrast(self):
        i_m = self.InsideMean
        o_m = self.SurroundingMean
        return np.abs(i_m-o_m).astype(float)/(o_m+i_m)

    def in_out_contrast2(self):
        int_im = self._oversize_image(1)
        i_max = np.amax(int_im).astype(float)
        i_min = np.amin(int_im).astype(float)
        return (i_max-i_min)/(i_max+i_min)

    def in_out_contrast3(self):
        ov_int_im = self._oversize_image(2).astype(float)
        ov_lab_im = self._oversize_label(2).astype(bool)
        mu = np.mean(ov_int_im[~ov_lab_im])
        sd = np.std(ov_int_im[~ov_lab_im])
        return np.mean((np.abs(ov_int_im-mu)/sd)[ov_lab_im])


    def __getattribute__(self, item):
        try:
            return super(ExtendedRegionProps, self).__getattribute__(item)
        except AttributeError:
            return  self.__getattr__(item)

    def __getattr__(self, item):
        if item == "InOutContrast":
            return self.in_out_contrast()
        elif item == "InOutContrast2":
            return self.in_out_contrast2()
        elif item == "InOutContrast3":
            return self.in_out_contrast3()
        elif item == "SurroundingStd":
            return self.sur_std()
        elif item == "SurroundingMean":
            return self.sur_mu()
        elif item == "InsideStd":
            return self.in_std()
        elif item == "InsideMean":
            return self.in_mu()
        elif item == "SurroundingMin":
            return self.sur_min()
        elif item == "SurroundingMax":
            return self.sur_max()
        elif item == "InsideMin":
            return self.in_min()
        elif item == "InsideMax":
            return self.in_max()
        else:
            raise AttributeError("'ExtendedRegionProps' object has no attribute '%s'"%item)

class RegionFilter(object):
    def __init__(self, prop, value, var=None, lower_limit=None, upper_limit=None, dist=None):
        super(RegionFilter, self).__init__()
        self.prop = str(prop)
        self.val_parameter = None
        self.var_parameter = None

        self.value = np.asarray(value, dtype=float)
        if self.value.ndim == 1:
            self.dim = self.value.shape[0]
        elif self.value.ndim == 0:
            self.dim = 1
        else:
            raise ValueError("Maximal Values of Vektor Shape allowed!")

        if var is not None:
            var = np.asarray(var, dtype=float)
            if len(var.shape) == 2:
                assert var.shape[0]==self.dim and var.shape[1]==self.dim
                self.var = var
            elif len(var.shape)==1:
                assert len(var) == self.dim
                self.var = np.diag(var)
            elif var.shape == ():
                self.var = np.diag(np.ones(self.dim))*var
            else:
                raise ValueError("False shape for variance parameter!")
        else:
            self.var = np.diag(np.ones(self.dim))

        if lower_limit is None:
            self.lower_limit = np.ones(self.dim)*-np.inf
        elif len(np.asarray(lower_limit).shape) == 1:
            self.lower_limit = np.asarray(lower_limit, dtype=float)
        elif np.asarray(lower_limit).shape == ():
            self.lower_limit = np.ones(self.dim)*lower_limit
        else:
            raise ValueError("False shape for lower limit parameter!")

        if upper_limit is None:
            self.upper_limit = np.ones(self.dim)*np.inf
        elif len(np.asarray(upper_limit).shape) == 1:
            self.upper_limit = np.asarray(upper_limit, dtype=float)
        elif np.asarray(upper_limit).shape == ():
            self.upper_limit = np.ones(self.dim)*upper_limit
        else:
            raise ValueError("False shape for upper limit parameter!")

        self._in_var = np.linalg.inv(self.var)
        self._N = (np.linalg.det(self.var)*(2*np.pi)**self.dim)**-0.5
        self._logN = np.log(self._N)

        self.set_dist()

        self.val_parameter = Parameter(self.prop, value, min=self.lower_limit, max=self.upper_limit, value_type=float)
        self.var_parameter = Parameter(self.prop+"_variance", self.var, min=0, value_type=float)

    def filter(self, regions):
        return [self.logprob(region.__getattribute__(self.prop)) for region in regions]

    def logprob(self, test_value):
        # self.set_dist()
        if np.all(self.lower_limit < test_value) and np.all(test_value  < self.upper_limit):
            return self._logN * np.dot(np.dot((test_value - self.value), self._in_var), (test_value - self.value))
            # return float(self.dist.logpdf(test_value-self.value))
        else:
            return -np.inf

    def set_dist(self):
        if self.dim == 1:
            self.dist = stats.norm(loc=np.zeros_like(self.value), scale=np.sqrt(self.var))
        else:
            self.dist = stats.multivariate_normal(mean=np.zeros_like(self.value), cov = self.var)

    @classmethod
    def from_dict(cls, dictionary):
        prop = dictionary.get("prop")
        value = dictionary.get("value")
        var = dictionary.get("var", None)
        lower_limit = dictionary.get("lower_limit", None)
        upper_limit = dictionary.get("upper_limit", None)
        dist = dictionary.get("dist", None)
        return cls(prop, value, var=var, lower_limit=lower_limit, upper_limit=upper_limit, dist=dist)


    def __getattribute__(self, item):
        if item == "value":
            if self.val_parameter is None:
                return super(RegionFilter, self).__getattribute__(item)
            else:
                return self.val_parameter.value
        elif item == "var":
            if self.var_parameter is None:
                return super(RegionFilter, self).__getattribute__(item)
            else:
                return self.var_parameter.value
        else:
            return super(RegionFilter, self).__getattribute__(item)


class RegionPropDetector(Detector):
    def __init__(self, RegionFilters):
        super(RegionPropDetector, self).__init__()
        self.Filters = []
        param_list = []
        for filter in RegionFilters:
            if type(filter)==RegionFilter:
                self.Filters.append(filter)
            elif type(filter)==dict:
                self.Filters.append(RegionFilter.from_dict(filter))
            else:
                raise ValueError("Filter must be of type RegionFilter or dictionary, not %s"%type(filter))
            param_list.append(filter.val_parameter)
            param_list.append(filter.var_parameter)
        self.ParameterList = ParameterList(*param_list)

    @detection_parameters(image=dict(frame=0, mask=False), mask=dict(frame=0, mask=True))
    def detect(self, image, mask):
        intensity_image = rgb2gray(image)
        regions = extended_regionprops(skimage.measure.label(mask), intensity_image=intensity_image)
        filter_dict = dict([[filter.prop, filter] for filter in self.Filters])

        cols = ["PositionX", "PositionY", "Log_Probability"]
        cols.extend(sorted(filter_dict.keys()))

        out = []
        for region in regions:
            o = []
            o.extend(region.centroid)
            prob = 0.
            o.append(float(prob))
            for key in cols[3:]:
                filter = filter_dict[key]
                # cols.update(filter.prop)
                # if np.isneginf(o[cols.index("Log_Probability")]):
                #     break
                o[cols.index("Log_Probability")] += filter.filter([region])[0]
                o.append(region.__getattribute__(filter.prop))
            out.append(o)

        out = pandas.DataFrame(out, columns=cols)
        # out = Measurement(o[cols.index("Log_Probability")],
        #                   [o[cols.index("PositionX")], o[cols.index("PositionY")]],
        #                   data=dict([[k, out[k]] for k in filter_dict.keys()]))
        return out, None


class AreaDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area=1, object_number=1, threshold=None, lower_limit=None, upper_limit=None):
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
        super(AreaDetector, self).__init__()
        self.start_ObjectArea = int(object_area)
        self.sample_size = 1
        self.ObjectArea = int(object_area)
        self.ObjectNumber = object_number
        if lower_limit:
            self.LowerLimit = int(lower_limit)
        else:
            self.LowerLimit = 0.4*self.ObjectArea
        if upper_limit:
            self.UpperLimit = int(upper_limit)
        else:
            self.UpperLimit = 1.6*self.ObjectArea

        self.Threshold = threshold
        self.Areas = []
        self.Sigma = None



        self.ParameterList = ParameterList(Parameter("ObjectArea", object_area, min=0., value_type=float),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Sigma", self.Sigma, min=0, value_type=float),
                                           Parameter("Threshold", self.Threshold, value_type=float),
                                           Parameter("LowerLimit", self.LowerLimit, value_type=float),
                                           Parameter("UpperLimit", self.UpperLimit, value_type=float)
                                           )

    @detection_parameters(image=dict(frame=0, mask=False))
    def detect(self, image):
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

        labeled = skimage.measure.label(image, connectivity=2)

        if len(self.Areas)<1e5 or self.Sigma is None:
            self.sample_size +=1
            self.Areas.extend([prop.area for prop in skimage.measure.regionprops(labeled)])
        else:
            new_areas = [prop.area for prop in skimage.measure.regionprops(labeled)]
            self.Areas = self.Areas[len(new_areas):]
            self.Areas.extend(new_areas)
        if True and len(self.Areas) > 0:
            self.ObjectArea = self.start_ObjectArea
            from scipy.optimize import curve_fit
            hist, bins = np.histogram(self.Areas, bins=max(self.Areas) - min(self.Areas))
            def gauss(x, sigma, mu):
                return self.sample_size*0.15*self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5 * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            def curve(x, sigma, mu):
                return gauss(x, sigma, mu) + hist[0] * x ** -2.2

            params, cov = curve_fit(curve, bins[1:], hist, (self.ObjectArea/6., self.ObjectArea), bounds=([1,0.5*self.ObjectArea], [self.ObjectArea, 2*self.ObjectArea]))
            self.Sigma, self.ObjectArea = params
            print("FOUND SIGMA OF %s"%self.Sigma)
        regions_list = [prop for prop in skimage.measure.regionprops(labeled) if prop.area > self.LowerLimit]

        if len(regions_list) <= 0:
            return np.array([])


        print("Object Area at ",self.ObjectArea, "pm", self.Sigma)

        out = []
        regions = {}
        [regions.update({prop.label: prop}) for prop in regions_list if prop.area > self.LowerLimit]

        N_max = np.floor(self.ObjectArea/self.Sigma)

        for prop in regions.values():
            n = np.ceil(prop.area/self.ObjectArea)
            if self.LowerLimit < prop.area < self.UpperLimit:
                out.extend(self.measure(prop, 1))
            elif n < N_max:
                pass
        return pandas.DataFrame(out, columns=["PositionX", "PositionY", "Log_Probability"])
        # return [Measurement(o[3], o[:3]) for o in out]

    def measure(self, prop, n):
        out = []
        if n == 1:
            sigma = self.Sigma
            mu = self.ObjectArea
            prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                        prop.area - mu) / sigma) ** 2
            o = []
            o.extend(prop.centroid)
            o.append(prob)
            out.append(o)
        elif n > 1:
            if min(prop.image.shape) < 2:
                bb = np.asarray(prop.bbox)
                sigma = 2 * self.Sigma
                mu = 2 * self.ObjectArea
                prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                            prop.area - mu) / sigma) ** 2
                for i in range(n+1):
                    o = []
                    o.extend(bb[:2]+i/float(n+1)*(bb[2:]-bb[:2]))
                    o.append(prob)
                    out.append(o)
            else:
                distance = ndi.distance_transform_edt(prop.image)
                local_maxi = peak_local_max(distance, indices=False,
                                            labels=prop.image, num_peaks=n)
                markers = ndi.label(local_maxi)[0]
                labels = watershed(-distance, markers, mask=prop.image)
                new_reg = skimage.measure.regionprops(labels)
                new_reg = [new for new in new_reg if new.label <= n]

                sigma = 2 * self.Sigma
                mu = 2 * self.ObjectArea
                prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                            prop.area - mu) / sigma) ** 2
                for new in new_reg:
                    o = []
                    o.extend(np.asarray(prop.bbox)[:2] + new.centroid)
                    o.append(prob)
                    out.append(o)
        return out


class AreaBlobDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_size=1, object_number=1, threshold=None):
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

        self.ParameterList = ParameterList(Parameter("ObjectSize", object_size, min=0., value_type=int),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Threshold", self.Threshold, value_type=float)
                                           )

    @detection_parameters(image=dict(frame=0, mask=False))
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


        for i, a in enumerate(areas):
            if mask[i]:
                for j in range(int(a//(areas[mask].mean()))+1):
                    # out.append(regions[i].centroid)
                    o = []
                    o.extend(regions[i].centroid)
                    o.append(1)
                    out.append(o)
        return pandas.DataFrame(out, columns=["PositionX", "PositionY", "Log_Probability"]), None
        # return [Measurement(o[3], o[:3]) for o in out], None


class WatershedDetector(Detector):
    """
    Detector classifying objects by area and number. It uses watershed algorithms to depart bigger areas.
    To be used with pengu-track modules.
    """
    def __init__(self, object_size=1, object_number=1, threshold=None):
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

        self.ParameterList = ParameterList(Parameter("ObjectSize", object_size, min=0., value_type=int),
                                           Parameter("ObjectNumber", object_number, min=0, value_type=int),
                                           Parameter("Threshold", self.Threshold, value_type=float)
                                           )

    @detection_parameters(image=dict(frame=0, mask=False))
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
            return pandas.DataFrame([[1., props.centroid[0], props.centroid[1]] for props in regions],
                                    columns=["Log_Probability", "PositionX", "PositionY"]), None
            # return [Measurement(1., props.centroid) for props in regions], None
        else:
            return [], None


def enhance_contrast(image):
    # kontrastspreizung: Helligkeitsunterschiede werden ausgeglichen und Bild mit maximalen Kontrast dargestellt
    image = image / np.percentile(image, 99.9)
    image = image - np.percentile(image, 0.1)

    image[image < 0] = 0  # alle Pixel mit Wert <0 werden auf 0 gesetzt
    image[image > 1] = 1
    return image


class TinaCellDetector(Detector):

    def __init__(self, disk_size=0, minimal_area=57, maximal_area=350, threshold=0.2):
        super(TinaCellDetector, self).__init__()
        self.DiskSize = disk_size
        self.MinimalArea = minimal_area
        self.MaximalArea = maximal_area
        self.Threshold = threshold

        self.ParameterList = ParameterList(Parameter("DiskSize", disk_size, min=0, max=20, value_type=int),
                                           Parameter("MinimalArea", minimal_area, min=0, max =100, value_type=int),
                                           Parameter("MaximalArea", maximal_area, min=0, max=1000, value_type=int),
                                           Parameter("Threshold", threshold, min=0., max=1., value_type=float)
                                           )

    @detection_parameters(minProj=dict(frame=0, mask=False, layer="MinProj"),
                          minIndices=dict(frame=0, mask=False, layer="MinIndices"),
                          maxProj=dict(frame=0, mask=False, layer="MaxProj"),
                          maxIndices=dict(frame=0, mask=False, layer="MaxIndices"))
    def detect(self, minProj, minIndices, maxProj, maxIndices):
        """

        :param minProj:
        :param minIndices:
        :param maxProj:
        :param maxIndices:
        :return:
        """
        print(minProj.sum(), minIndices.sum(), maxProj.sum(), maxIndices.sum())

        minind_contrast = enhance_contrast(minIndices)
        maxproj_contrast =  enhance_contrast(maxProj)
        minproj_contrast = enhance_contrast(minProj)

        minproj_ben = minproj_contrast - np.min(minproj_contrast)
        minproj_ben = minproj_ben / np.max(minproj_ben)
        maxproj_ben = maxproj_contrast - np.min(maxproj_contrast)
        maxproj_ben = maxproj_ben / np.max(maxproj_ben)

        import scipy
        from skimage import filters
        maxproj_gausfilt = scipy.ndimage.filters.gaussian_filter(maxproj_ben, 2)
        thres_max = self.Threshold # threshold_otsu(maxproj_ben)

        #minproj_gausfilt = scipy.ndimage.filters.gaussian_filter(minproj_ben, 0.2)
        #thres_min = threshold_otsu(minproj_ben)

        mask = (maxproj_gausfilt > thres_max)
        #mask = (maxproj_gausfilt>thres_max)*(minproj_gausfilt>thres_min)

        struct1 = skimage.morphology.disk(self.DiskSize)
        mask_erosion = skimage.morphology.binary_erosion(mask).astype(mask.dtype)  # rauschbedingte Pixel werden geloescht
        mask_dilation = skimage.morphology.binary_dilation(mask_erosion).astype(mask_erosion.dtype)

        mask_dilation2 = skimage.morphology.binary_dilation(mask_dilation,struct1).astype(mask_dilation.dtype)
        # mask_erosion = skimage.morphology.binary_erosion(mask_dilation2).astype(mask_dilation2.dtype)  # rauschbedingte Pixel werden geloescht

        mask_ind = mask_dilation2*maxIndices

        labels, n = label(mask_ind)
        #labels, n = label(mask_dilation2)
        regions = regionprops(labels)
        posx = np.array([e.centroid[0] for e in regions])
        posy = np.array([e.centroid[1] for e in regions])

        # threshold depending on cellsize
        area = np.array([e.area for e in regions])
        area_thres_min = self.MinimalArea
        area_thres_max = self.MaximalArea
        cells = [r for r in regions if (r.area <= area_thres_max) and (r.area >= area_thres_min)]
        cellx = np.array([e.centroid[0] for e in cells])
        celly = np.array([e.centroid[1] for e in cells])

        cell_program = np.vstack((cellx, celly))
        cell_program = cell_program.transpose()

        posarea = ['%d' %e.area for e in regions]

        labelposforz = [cells[e].label for e in range(len(cells))]
        zmaxind = np.array([np.nanmean(maxIndices[labels==labelposforz[e]]) for e in range(len(labelposforz))])
        zminind = np.array([np.nanmean(minIndices[labels==labelposforz[e]]) for e in range(len(labelposforz))])
        meanind = (zmaxind+zminind)/2
        zpos = []
        for e in meanind:
            (values, counts) = np.unique(e, return_counts=True) #to get most frequent number (=zpos) in  meanind
            ind = np.argmax(counts)
            z = values[ind]
            z = np.round(z)
            zpos.append(z)

        Positions3D = np.c_[cellx,celly,zpos]
        Positions3D = pandas.DataFrame(Positions3D, columns=["PositionY", "PositionX", "PositionZ"])

        # return posx,posy, cell_program, Positions3D, cellx, celly, mask_dilation2,posarea
        return Positions3D, mask_dilation2



class Segmentation(object):
    """
    This Class describes the abstract function of a image-segmentation-algorithm in the pengu-track package.
    It is only meant for subclassing.
    """
    def __init__(self):
        super(Segmentation, self).__init__()

    def detect(self, image, *args, **kwargs):
        out = self.segmentate(image, *args, **kwargs)
        self.update(out, image, *args, **kwargs)
        return out

    def update(self, mask, image, *args, **kwargs):
        pass

    def segmentate(self, image, *args, **kwargs):
        pass


class ThresholdSegmentation(Segmentation):

    def __init__(self, treshold, reskale=True):
        super(ThresholdSegmentation, self).__init__()

        self.width = None
        self.height = None
        self.SegMap = None
        self.Treshold = float(treshold)
        self.Skale = None
        self.reskale = reskale

    def segmentate(self, image, *args, **kwargs):
        data = np.array(image, ndmin=2)
        print(data.shape)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]

        if self.reskale:
            if len(data.shape)==3:
                # this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))
                this_skale = np.mean(rgb2gray(data))
            elif len(data.shape)==2:
                this_skale = np.mean(data)
        else:
            this_skale = 1.

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale
        dt = data.dtype
        data = (data.astype(float)*(self.Skale/this_skale)).astype(dt)

        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            self.SegMap = (rgb2gray(data)>self.Treshold).astype(bool)
            # self.SegMap = (np.sum(data**2, axis=-1)**0.5/data.shape[-1]**0.5 > self.Treshold).astype(bool)
        elif len(data.shape) == 2:
            print(np.amin(data), np.amax(data))
            self.SegMap = (data > self.Treshold).astype(bool)
        else:
            raise ValueError('False format of data.')
        return self.SegMap

    def update(self, mask, image, *args, **kwargs):
        pass

class VarianceSegmentation(Segmentation):

    def __init__(self, treshold, r):
        super(VarianceSegmentation, self).__init__()

        self.width = None
        self.height = None
        self.SegMap = None
        self.Treshold = int(treshold)
        self.Skale = None
        self.Radius = int(r)
        self.selem = skimage.morphology.disk(self.Radius)

    def segmentate(self, image, *args, **kwargs):
        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)

        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            self.SegMap = (self.local_std(data) < self.Treshold).astype(bool)
        elif len(data.shape) == 2:
            std = self.local_std(data)
            self.SegMap = (self.local_std(data) < self.Treshold).astype(bool)
        else:
            raise ValueError('False format of data.')
        return self.SegMap

    def update(self, mask, image, *args, **kwargs):
        pass

    def local_std(self, img):
        shifts = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]
        stack = [img]
        for s in shifts:
            stack.append(shift(img, s, order=0, mode='reflect'))
        stack = np.array(stack)
        return np.std(stack, axis=0)

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

        self.dists = None

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

    def segmentate(self, image, *args, **kwargs):
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

        self.dists = np.abs(self.Mu-data)*(1/self.Var**0.5)

        if len(data.shape) == 3:
            return np.sum(np.sum(self.dists < 4, axis=0), axis=-1) > 2
        elif len(data.shape) == 2:
            return np.sum(self.dists < 4, axis=0) > 2
        else:
            raise ValueError('False format of data.')

    def update(self, mask, image, *args, **kwargs):

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        dists = self.dists
        matchs = np.zeros_like(data, dtype=bool)
        matchs[np.unravel_index(np.argmin(dists, axis=0), data.shape)] = True
        # matchs = (dists == np.amin(dists, axis=0))
        outliers = (dists == np.amin(dists, axis=0))

        self.Mu[matchs] = (self.Mu[matchs] * ((self.N-1)/self.N) + data.ravel()*(1./self.N))
        self.Var[matchs] = (self.Var[matchs] * ((self.N-2)/(self.N-1)) + (data.ravel()-self.Mu[matchs]))*(1./(self.N-1))

        self.Mu[outliers] = np.mean(self.Mu, axis=0).ravel()
        self.Var[outliers] = np.mean(self.Var, axis=0).ravel()

    def detect(self, image, *args, **kwargs):
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

class MoGSegmentation2(Segmentation):
    """
    Segmentation method comparing input images to image-background buffer. Able to learn new background information.
    """

    def __init__(self, n=20, r=15, init_image=None):
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
        super(MoGSegmentation2, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        # self.Phi = int(phi)
        self.Skale = None
        self.Mu = None
        self.Sig = None
        self.Max = None
        self.NN = None

        if init_image is not None:
            data = np.array(init_image, ndmin=2)
            self.__dt__ = smallest_dtype(data)
            print(self.__dt__)
            if len(data.shape) == 3:
                self.Mu = np.tile(data.astype(self.__dt__), self.N
                                       ).reshape(data.shape + (self.N,)
                                                 ).transpose((3, 0, 1, 2))
                self.Sig = np.ones_like(self.Mu, dtype=self.__dt__)*self.__dt__(self.R)
                self.N = np.ones_like(self.Mu, dtype=np.uint16)
                self.Skale = np.mean(rgb2gray(data))
            elif len(data.shape) == 2:
                self.Mu = np.tile(data.astype(self.__dt__), (self.N, 1, 1,))
                self.Sig = np.ones_like(self.Mu, dtype=self.__dt__) * self.__dt__(self.R)
                self.N = np.ones_like(self.Mu, dtype=np.uint16)
                self.Skale = np.mean(data)
            else:
                raise ValueError('False format of data.')
            self.Max = np.zeros_like(self.Mu, dtype=self.__dt__)
            # self.Max = np.tile(np.arange(N), data.shape + (1,)).T
        else:
            self.__dt__ = None

        self.SegMap = None
        self.width = None
        self.height = None

    def detect(self, image, do_neighbours=True, *args, **kwargs):
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
        return super(MoGSegmentation2, self).detect(image, *args, **kwargs)

    def segmentate(self, image, do_neighbours=True, mask=None, *args, **kwargs):
        super(MoGSegmentation2, self).segmentate(image)
        data = np.array(image, ndmin=2)
        self.Mask = mask

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float) * (self.Skale / this_skale)).astype(self.__dt__)
        if self.Mu is None:
            if len(data.shape) == 3:
                self.Mu = np.tile(data, self.N).reshape((self.N,) + data.shape)
            elif len(data.shape)==2:
                self.Mu = np.tile(data.T, self.N).T.reshape((self.N,) + data.shape[::-1])

        if self.Sig is None:
                self.Sig = np.ones_like(self.Mu) * self.R
        if self.N is None:
                self.N = np.ones_like(self.Mu)

        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if self.Max is None:
            self.Max = np.zeros_like(self.Mu, dtype=bool)

        if len(data.shape) == 3:
            diff = self.Mu.astype(next_dtype(-1 * data))
            diff = np.abs(rgb2gray(diff - data))
            self.SegMap = np.all((diff > self.Sig), axis=0)
            self.Max = (np.tile(np.arange(self.N), data.shape[::-1] + (1, )).T == np.argmin(np.abs(diff-self.Sig), axis=0))
        elif len(data.shape) == 2:
            diff = np.asarray(
                [np.amax([sample, data], axis=0) - np.amin([sample, data], axis=0) for sample in self.Mu])
            self.SegMap = np.all((diff > self.Sig), axis=0)
            self.Max = (np.tile(np.arange(self.N), data.shape[::-1] + (1, )).T == np.argmin(np.abs(diff-self.Sig), axis=0))
        else:
            raise ValueError('False format of data.')
        self.N += self.SegMap
        self.N[~self.SegMap] = 1
        if self.Mask is not None and np.all(self.Mask.shape == self.SegMap.shape):
            self.SegMap &= ~self.Mask
        return self.SegMap

    def update(self, mask, image, do_neighbours=True, *args, **kwargs):

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float) * (self.Skale / this_skale)).astype(self.__dt__)

        for i in range(self.N):
            self.Mu[i][self.Max[i]] += (data[self.Max[i]]).astype(self.__dt__)/self.N[i]
            self.Sig[i][self.Max[i]] = (self.Sig[i][self.Max[i]]**2+(((data[self.Max[i]]).astype(self.__dt__)-self.Mu[i][self.Max[i]])**2)/(self.N[i]))**2

        n = np.sum(image_mask)
        print("Updated %s pixels" % n)


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
            self.__dt__ = smallest_dtype(data)
            print(self.__dt__)
            if len(data.shape) == 3:
                self.Samples = np.tile(data.astype(self.__dt__), self.N
                                       ).reshape(data.shape+(self.N,)
                                                                      ).transpose((3, 0, 1, 2))
                self.Skale = np.mean(rgb2gray(data))
            elif len(data.shape) == 2:
                self.Samples = np.tile(data.astype(self.__dt__), (self.N, 1, 1,))
                self.Skale = np.mean(data)
            else:
                raise ValueError('False format of data.')
        else:
            self.__dt__ = None

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

    def detect(self, image, do_neighbours=True, *args, **kwargs):
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
        return super(ViBeSegmentation, self).detect(image, do_neighbours=True, *args, **kwargs)

    def segmentate(self, image, do_neighbours=True, mask=None, return_diff=False, *args, **kwargs):
        super(ViBeSegmentation, self).segmentate(image)
        data = np.array(image, ndmin=2)
        self.Mask = mask

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            # self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1])
            #                        > self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
            diff = self.Samples.astype(next_dtype(-1*data))
            diff = np.abs(rgb2gray(diff-data))
            self.SegMap = (np.sum((diff
                                   > self.R).astype(np.uint8), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        elif len(data.shape) == 2:
            diff = np.asarray([np.amax([sample, data], axis=0)-np.amin([sample,data], axis=0) for sample in self.Samples])
            self.SegMap = (np.sum((diff > self.R).astype(np.uint8), axis=0, dtype=np.uint8)
                           >= self.N_min).astype(bool)
        else:
            raise ValueError('False format of data.')
        if self.Mask is not None and np.all(self.Mask.shape == self.SegMap.shape):
            self.SegMap &= ~self.Mask

        if return_diff:
            return self.SegMap, diff
        return self.SegMap

    def update(self, mask, image, do_neighbours=True, *args, **kwargs):

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)
        image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & mask

        sample_index = np.random.randint(0, self.N)

        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(self.__dt__)

        do_neighbours = False
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


class MeanViBeSegmentation(Segmentation):
    def __init__(self, sensitivity=1, n=20, m=1, init_image=None, *args, **kwargs):
        super(MeanViBeSegmentation, self).__init__(*args, **kwargs)
        self.sensitivity = float(sensitivity)
        self.N = int(n)
        self.M = int(m)
        self.Samples = None
        self.Mu = None
        self.Sig = None
        self.SegMap = None
        self._counter = 0
        self.__dt__ = float
        if init_image is not None:
            data = np.array(init_image, ndmin=2)
            # self.__dt__ = smallest_dtype(data)
            # print(self.__dt__)
            if len(data.shape) == 3:
                self.Samples = np.tile(data.astype(self.__dt__), self.N)\
                    .reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            elif len(data.shape) == 2:
                self.Samples = np.tile(data.astype(self.__dt__), (self.N, 1, 1,))
            else:
                raise ValueError('False format of data.')
        # else:
        #     self.__dt__ = None

    def segmentate(self, image, *args, **kwargs):
        data = np.array(image, ndmin=2)
            
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)

        # mu = np.mean(self.Samples.astype(next_dtype(-1*data))[::self.M], axis=0)
        # sig = np.std(self.Samples.astype(next_dtype(-1*data))[::self.M], axis=0)
        if self.Mu is None:
            self.Mu = np.mean(self.Samples[::self.M], axis=0)
        if self.Sig is None:
            self.Sig = np.std(self.Samples[::self.M], axis=0)
        mu = self.Mu
        sig = self.Sig
        if len(data.shape) == 3:
            data = rgb2gray(data)
            self.SegMap = np.abs(mu-data)>sig*self.sensitivity
        elif len(data.shape) == 2:
            self.SegMap = np.abs(mu-data)>sig*self.sensitivity
        else:
            raise ValueError('False format of data.')

            
        return self.SegMap

    def update(self, mask, image, *args, **kwargs):
        data = np.array(image, ndmin=2, dtype=self.__dt__)
        # self.Samples[:-1] = self.Samples[1:]

        x_0 = data
        x_N = self.Samples[self._counter]

        if self.Mu is None:
            self.Mu = np.mean(self.Samples[::self.M], axis=0)

        mu_1 = ((self.N*self.Mu) - x_N)/(self.N-1)
        self.Mu = (((self.N - 1) * mu_1 + x_0)/self.N)

        if self.Sig is None:
            self.Sig = np.std(self.Samples[::self.M], axis=0)

        sig_1 = np.sqrt(((self.N*self.Sig**2)-(x_N-mu_1)*(x_N-self.Mu))/(self.N-1))
        self.Sig = np.sqrt(((self.N-1)*sig_1**2 + (x_N-mu_1)*(x_N-self.Mu))/self.N)

        self.Samples[self._counter] = data

        self._counter = (self._counter+1)%self.N


class DumbViBeSegmentation(ViBeSegmentation):
    def __init__(self, *args, **kwargs):
        super(DumbViBeSegmentation, self).__init__(*args, **kwargs)
        self.SampleIndex = 0

    def segmentate(self, image, do_neighbours=True, mask=None, return_diff=False, *args, **kwargs):
        super(ViBeSegmentation, self).segmentate(image)
        data = np.array(image, ndmin=2)
        self.Mask = mask

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            diff = self.Samples.astype(next_dtype(-1*data))
            diff = np.abs(rgb2gray(diff-data))
            self.SegMap = np.all(diff > self.R, axis=0).astype(bool)
        elif len(data.shape) == 2:
            diff = np.asarray([np.amax([sample, data],axis=0)-np.amin([sample,data], axis=0) for sample in self.Samples])
            self.SegMap = np.all(diff>self.R, axis=0).astype(bool)
        else:
            raise ValueError('False format of data.')
        if self.Mask is not None and np.all(self.Mask.shape == self.SegMap.shape):
            self.SegMap &= ~self.Mask

        if return_diff:
            return self.SegMap, diff
        return self.SegMap

    def update(self, mask, image, do_neighbours=True, *args, **kwargs):

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)

        self.SampleIndex = (self.SampleIndex+1)%self.N

        self.Samples[self.SampleIndex] = data.astype(self.__dt__)

        print("Updated %s pixels" % np.sum(self.SegMap))


class AlexSegmentation(ViBeSegmentation):
    """
    Segmentation method comparing input images to image-background buffer. Able to learn new background information.
    """

    def __init__(self, *args, **kwargs):
        super(AlexSegmentation, self).__init__(*args, **kwargs)

    def segmentate(self, image, do_neighbours=True, mask=None, *args, **kwargs):
        super(ViBeSegmentation, self).segmentate(image)
        data = np.array(image, ndmin=2)
        self.Mask = mask

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        if len(data.shape) == 3:
            diff = self.Samples.astype(next_dtype(-1*data))
            diff = np.abs(rgb2gray(diff-data))
            self.SegMap = (np.sum((diff
                                   > self.R).astype(np.uint8), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        elif len(data.shape) == 2:
            diff = np.asarray([np.amax([sample, data],axis=0)-np.amin([sample, data], axis=0) for sample in self.Samples])
            self.SegMap = (np.sum((diff > self.R).astype(np.uint8), axis=0, dtype=np.uint8)
                           >= self.N_min).astype(bool)
        else:
            raise ValueError('False format of data.')
        if self.Mask is not None and np.all(self.Mask.shape == self.SegMap.shape):
            self.SegMap &= ~self.Mask
        return self.SegMap, diff

    def update(self, mask, image, do_neighbours=True):
        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        if len(data.shape) == 3:
            this_skale = np.mean(rgb2gray(data))
        elif len(data.shape) == 2:
            this_skale = np.mean(data)
        else:
            raise ValueError('False format of data.')

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        if self.__dt__ is None:
            self.__dt__ = smallest_dtype(data)

        data = (data.astype(float)*(self.Skale/this_skale)).astype(self.__dt__)
        image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & mask

        sample_index = np.random.randint(0, self.N)

        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(self.__dt__)

        print("Updated %s pixels" % np.sum(self.SegMap))

class BlobSegmentation(Segmentation):
    """
    Segmentation method detecting blobs.
    """
    def __init__(self, max_size, min_size=1, init_image=None):
        """
    Segmentation method detecting blobs.

        Parameters
        ----------
        min_size: int, optional
            Number of buffer frames.
        min_size: int, optional
            Minimum diameter of blobs.

        init_image: array_like, optional
            Image for initialisation of background.

        """
        super(BlobSegmentation, self).__init__()
        self.Min = min_size
        self.Max = max_size
        self.Skale = None

        if init_image is not None:
            self.Skale = np.mean((np.sum(np.mean(self.Samples, axis=0).astype(np.uint32)**2, axis=-1)**0.5))

        self.SegMap = None
        self.width = None
        self.height = None

    def detect(self, image, do_neighbours=True, *args, **kwargs):
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
        super(BlobSegmentation, self).detect(image)

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        image = np.array(data, dtype=int)
        while len(image.shape) > 2:
            image = np.linalg.norm(image, axis=-1)

        if self.SegMap is None:
            self.SegMap = np.ones((self.width, self.height), dtype=bool)

        self.SegMap = img_as_uint(-1*
                      skimage.filters.laplace(
                      skimage.filters.gaussian(image.astype(np.uint8), self.Min)))
        self.SegMap = (self.SegMap/256).astype(np.uint8)
        self.SegMap = (self.SegMap) >0 & (self.SegMap <6)

        return self.SegMap


class FlowDetector(Segmentation):
    def __init__(self, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                 *args, **kwargs):
        super(FlowDetector, self).__init__()
        self.Flow = flow
        self.PyrScale = pyr_scale
        self.Levels = levels
        self.WinSize = winsize
        self.Iterations = iterations
        self.PolyN = poly_n
        self.PolySigma = poly_sigma
        self.Flags = flags
        self.Prev = None

    def segmentate(self, image, *args, **kwargs):
        if self.Prev is None:
            raise AttributeError("First a starting image has to be added by FlowDetector.update")
        return cv2.calcOpticalFlowFarneback(self.Prev, image,
                                 self.Flow,
                                 self.PyrScale,
                                 self.Levels,
                                 self.WinSize,
                                 self.Iterations,
                                 self.PolyN,
                                 self.PolySigma,
                                 self.Flags)

    def update(self, mask, image, *args, **kwargs):
        if self.Prev is None:
            self.Prev = image
        else:
            self.Prev[mask] = image[mask]

    def detect(self, image, *args, **kwargs):
        out = self.segmentate(image, *args, **kwargs)
        mask = kwargs.get("mask", np.ones_like(image, dtype=bool))
        self.update(mask, image, *args, **kwargs)
        return out


class EmperorDetector(FlowDetector):
    def __init__(self, initial_image, **kwargs):
        super(EmperorDetector, self).__init__(**kwargs)
        self.Area = kwargs.get("area", 10)
        self.LowerLimitArea = kwargs.get("lower_limit_area", 1)
        self.UpperLimitArea = kwargs.get("upper_limit_area", 50)
        rf1 = RegionFilter("area", self.Area, lower_limit=self.LowerLimitArea, upper_limit=self.UpperLimitArea)

        self.LuminanceThreshold = kwargs.get("luminance_threshold", 1.3)
        self.RegionDetector = RegionPropDetector([rf1])

        image = np.array(initial_image)
        if len(image.shape) == 3:
            image_int = rgb2gray(image)
        elif len(image.shape) == 2:
            image_int = image
        else:
            raise ValueError("Initial Image must be a RGB or Gray image!")
        self.update(np.ones_like(image_int, dtype=bool), image_int)

    def detect(self, image0, image1, *args, **kwargs):
        image0 = np.array(image0)
        image1 = np.array(image1)
        if len(image0.shape) == 3:
            image0_int = rgb2gray(image0)
        elif len(image0.shape) == 2:
            image0_int = image0
        else:
            raise ValueError("Input Image must be a RGB or Gray image!")
        if len(image1.shape) == 3:
            image1_int = rgb2gray(image1)
        elif len(image1.shape) == 2:
            image1_int = image1
        else:
            raise ValueError("Input Image must be a RGB or Gray image!")
        flow = super(EmperorDetector, self).detect(image1_int, *args, **kwargs)
        if not flow.shape == image1.shape[:2]+(2,):
            raise ValueError("Input Flow must be of shape [M,N,2]! (M,N being the image dimensions)")
        win_size = self.WinSize
        flowX = flow.T[0].T
        flowY = flow.T[1].T
        amax = np.amax(np.abs(flow))
        flowX = filters.gaussian(flowX / amax, sigma=win_size) * amax
        flowY = filters.gaussian(flowY / amax, sigma=win_size) * amax

        interpolatorX = RegularGridInterpolator((np.arange(flowX.shape[0]), np.arange(flowX.shape[1])), flowX)
        interpolatorY = RegularGridInterpolator((np.arange(flowY.shape[0]), np.arange(flowY.shape[1])), flowY)

        window_size = int(np.round(np.round(np.sqrt(self.Area))/np.pi)*2 + 1)*3

        measurements = self.RegionDetector.detect(
            image0_int > self.LuminanceThreshold*threshold_niblack(image0_int, window_size=window_size)
            , image0_int)

        indices, points_x, points_y = np.array([[i, m.PositionX, m.PositionY] for i, m in enumerate(measurements)],
                                               dtype=object).T
        points = np.array([points_x, points_y]).T
        vel_x = interpolatorX(points)
        vel_y = interpolatorY(points)
        for i in indices:
            measurements[i]["VelocityX"] = vel_y[i]
            measurements[i]["VelocityY"] = vel_x[i]

        return measurements

def rgb2gray(rgb):
    rgb = np.asarray(rgb)
    if len(rgb.shape)>2 and rgb.shape[2]==3:
        dt = rgb.dtype
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(dt)
    elif len(rgb.shape)==2:
        return rgb
    else:
        raise ValueError("Array is not in image shape (N,M,3)!")

def gray2rgb(gray):
    dt = gray.dtype
    return np.tensordot(gray, [1,  1,  1], axes=0).astype(dt)

def next_dtype(array):
    dt = array.dtype

    if dt == np.int and np.amin(array) >= 0:
        a_max = np.amax(array)
        if a_max < 2**8:
            return np.uint16
        elif a_max < 2**16:
            return np.uint32
        elif a_max < 2**32:
            return np.uint64
        elif a_max < 2**64:
            return np.uint128
        else:
            return np.float
    elif dt == np.int:
        a_max = max(-np.amin(array), np.amax(array))
        if a_max < 2**7:
            return np.int16
        elif a_max < 2**15:
            return np.int32
        elif a_max < 2**31:
            return np.uint64
        elif a_max < 2**63:
            return np.uint128
    if dt == np.uint8:
        return np.uint16
    elif dt == np.uint16:
        return np.uint32
    elif dt == np.uint32:
        return np.uint64
    elif dt == np.uint64:
        return np.uint128
    elif dt == np.int8:
        return np.int16
    elif dt == np.int16:
        return np.int32
    elif dt == np.int32:
        return np.int64
    elif dt == np.int64:
        return np.int128
    elif type(dt) == np.dtype:
        return np.float
    else:
        raise ValueError("Input is not a numpy array")

def smallest_dtype(array):
    dt = array.dtype

    if str(dt).count("int") and np.amin(array) >= 0:
        a_max = np.amax(array)
        if a_max < 2**8:
            return np.uint8
        elif a_max < 2**16:
            return np.uint16
        elif a_max < 2**32:
            return np.uint32
        elif a_max < 2**64:
            return np.uint64
        elif a_max < 2**128:
            return np.uint128
        else:
            return np.float
    elif str(dt).count("int"):
        a_max = max(-np.amin(array), np.amax(array))
        if a_max < 2**7:
            return np.int8
        elif a_max < 2**15:
            return np.int16
        elif a_max < 2**31:
            return np.uint32
        elif a_max < 2**63:
            return np.uint64
        elif a_max < 2**127:
            return np.uint128
    elif type(dt) == np.dtype:
        return np.float
    else:
        raise ValueError("Input is not a numpy array")

####### LEGACY ######

class NKCellDetector(Detector):
    def __init__(self):
        super(NKCellDetector, self).__init__()

    def enhance(self, image, percentile):
        image -= np.percentile(image, percentile)
        image /= np.percentile(image, 100. - percentile)
        image[image < 0.] = 0.
        image[image > 1.] = 1.
        return image


    @detection_parameters(minProj=dict(frame=0, layer="MinProj", mask=False),
                          minIndices=dict(frame=0, layer="MinIndices", mask=False),
                          minProjPrvs=dict(frame=-1, layer="MinProj", mask=False),
                          maxIndices=dict(frame=0, layer="MaxIndices", mask=False))
    def detect(self, minProj, minIndices, minProjPrvs, maxIndices):
        data = gaussian_filter(minProj.astype(float)-minProjPrvs.astype(float), 5)
        data[data > 0] = 0.
        data = self.enhance(data, 1.)
        data = 1. - data
        mask = data > 0.5
        labeled, num_objects = label(mask)

        regions = regionprops(labeled, intensity_image=data)
        regions = [r for r in regions if r.area > 100]

        out = []
        mu = np.mean([prop.area for prop in regions])
        for prop in regions:
            sigma = np.sqrt(prop.area)  # assuming Poisson distributed areas
            prob = np.log(len(regions) / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((prop.area - mu) / sigma) ** 2
            intensities = prop.intensity_image[prop.image]
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            # out.append(Measurement(prob, [prop.weighted_centroid[0], prop.weighted_centroid[1], mean_int], data=std_int))
            out.append([prop.weighted_centroid[1], prop.weighted_centroid[0], mean_int, prob, std_int])
        out = pandas.DataFrame(out, columns=["PositionX", "PositionY", "PositionZ", "Log_Probability", "IntensitySTD"])
        # out = [Measurement(o[3], o[:3], data={"IntensitySTD": o[4]}) for o in out]

        return out, mask

class NKCellDetector2(Detector):
    def __init__(self):
        super(NKCellDetector2, self).__init__()

    def enhance(self, image, percentile):
        image = image.astype(np.float)
        image -= np.percentile(image, percentile)
        image /= np.percentile(image, 100. - percentile)
        image[image < 0.] = 0.
        image[image > 1.] = 1.
        return image


    @detection_parameters(minProj=dict(frame=0, layer="MinimumProjection", mask=False),
                          minProjPrvs=dict(frame=-1, layer="MinimumProjection", mask=False))
    def detect(self, minProj, minProjPrvs):

        minProjPrvs = self.enhance(minProjPrvs, 0.1)
        minProj = self.enhance(minProj, 0.1)

        data = gaussian_filter(minProj-minProjPrvs, 5)
        data[data > 0] = 0.
        data = -1. * data
        mask = data > 0.1
        labeled, num_objects = label(mask)

        regions = regionprops(labeled, intensity_image=data)
        regions = [r for r in regions if r.area > 100]

        out = []
        mu = np.mean([prop.area for prop in regions])
        for prop in regions:
            sigma = np.sqrt(prop.area)  # assuming Poisson distributed areas
            prob = np.log(len(regions) / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((prop.area - mu) / sigma) ** 2
            intensities = prop.intensity_image[prop.image]
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            # out.append(Measurement(prob, [prop.weighted_centroid[0], prop.weighted_centroid[1], mean_int], data=std_int))
            out.append([prop.weighted_centroid[1], prop.weighted_centroid[0], mean_int, prob, std_int])
        out = pandas.DataFrame(out, columns=["PositionX", "PositionY", "PositionZ", "Log_Probability", "IntensitySTD"])
        # out = [Measurement(o[3], o[:3], data={"IntensitySTD":o[4]}) for o in out]

        return out, mask

class TCellDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self):
        super(TCellDetector, self).__init__()

    def enhance(self, image, percentile):
        image = image.astype(np.float)

        bgd = threshold_niblack(image, 51)
        image = image / bgd

        image -= np.percentile(image, percentile)
        image /= np.percentile(image, 100. - percentile)
        image[image < 0.] = 0.
        image[image > 1.] = 1.
        return image


    @detection_parameters(minProj=dict(frame=0, layer="MinimumProjection", mask=False),
                          minIndices=dict(frame=0, layer="MinimumIndices", mask=False))
    def detect(self, minProj, minIndices):
        minProj2 = self.enhance(minProj, 0.1)
        thres = filters.threshold_otsu(minProj2)
        mask = minProj2 < thres

        mask = skimage.morphology.binary_erosion(mask)
        # mask = remove_small_objects(mask, 24)

        maskedMinIndices = minIndices.copy() + 1
        maskedMinIndices = maskedMinIndices.astype('uint8')
        # maskedMinIndices = maskedMinIndices.astype(np.uint8)
        # maskedMinIndices = minIndices.data[:] + 1
        # maskedMinIndices = np.round(gaussian_filter(maskedMinIndices, 1)).astype(np.int)
        # maskedMinIndices = np.round(cv2.bilateralFilter(maskedMinIndices, -1, 3, 5)).astype(np.int)
        # maskedMinIndices = np.round(cv2.bilateralFilter(maskedMinIndices, -1, 3, 5)).astype(np.int)
        maskedMinIndices = np.round(cv2.bilateralFilter(maskedMinIndices, -1, 3, 5)).astype(np.int)
        # maskedMinIndices = np.round(cv2.bilateralFilter(maskedMinIndices, -1, 3, 5)).astype(np.int)
        maskedMinIndices[~mask] = 0
        j_max = np.amax(maskedMinIndices)
        stack = np.zeros((j_max, minProj.shape[0], minProj.shape[1]), dtype=np.bool)
        for j in range(j_max):
            stack[j, maskedMinIndices == j] = True

        labels3D, n = label(stack, structure=np.ones((3, 3, 3)))
        labels2D = np.sum(labels3D, axis=0) - 1
        # labels2D = remove_small_objects(labels2D, 24)

        regions = regionprops(labels2D, maskedMinIndices)
        areas = np.array([r.area for r in regions])
        area_thres = filters.threshold_otsu(areas) * 0.7
        if area_thres > 60:
            area_thres = 38.0
        if area_thres < 10:
            area_thres = 10.0
        regions = [r for r in regions if r.area >= area_thres]
        #centroids = np.array([r.centroid for r in regions]).T

        out = []
        mu = np.mean([prop.area for prop in regions])
        for prop in regions:
            sigma = np.sqrt(prop.area)  # assuming Poisson distributed areas
            prob = np.log(len(regions) / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                            prop.area - mu) / sigma) ** 2
            intensities = prop.intensity_image[prop.image]
            mean_int = np.mean(intensities)
            std_int = np.std(intensities)
            # out.append(Measurement(prob, [prop.centroid[0], prop.centroid[1], mean_int], data=std_int))
            out.append([prop.weighted_centroid[1], prop.weighted_centroid[0], mean_int, prob, std_int])
        out = pandas.DataFrame(out, columns=["PositionX", "PositionY", "PositionZ", "Log_Probability", "IntensitySTD"])
        # out = [Measurement(o[3], o[:3], data={"IntensitySTD": o[4]}) for o in out]

        #res = 6.45 / 10
        #out.PositionX *= res
        #out.PositionY *= res
        #out.PositionZ *= 10

        return out, mask
