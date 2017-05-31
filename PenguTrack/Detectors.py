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
from scipy.ndimage.interpolation import shift
from skimage import img_as_uint

from scipy import ndimage as ndi
from scipy.special import lambertw
from scipy.ndimage.measurements import label

from skimage.morphology import watershed
from skimage.feature import peak_local_max


# import theano
# import theano.tensor as T

class Measurement(object):
    """
    Base Class for detection results.
    """
    def __init__(self, log_probability, position, data=None, frame=None, track_id=None):
        """
        Base Class for detection results.

        probability: float
            Estimated logarithmic probability of measurement.
        position: array_like
            Position of measurement
        data: array_like
            Additional data of the measurement, i.e. measurement errors
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

        self.Data = data


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

class SimpleAreaDetector2(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area, object_number, threshold=None, lower_limit=None, upper_limit=None, distxy_boundary = 10, distz_boundary = 21, pre_stitching = True):
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
        self.pre_stitching = pre_stitching
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
        self.Sigma = np.sqrt((self.UpperLimit-self.LowerLimit)/(4*np.log(1./0.95))) # self.ObjectArea/2.

    def detect(self, image, mask, return_regions=False, get_all=False, return_labeled=False, only_for_detection=False):
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
        if get_all:
            regions_list = [prop for prop in skimage.measure.regionprops(labeled)]
        else:
            regions_list = [prop for prop in skimage.measure.regionprops(labeled, intensity_image=image)
                       if (self.UpperLimit > prop.area > self.LowerLimit and prop.solidity > 0.5)]
        if len(regions_list) <= 0:
            return np.array([])

        print("Object Area at ",self.ObjectArea, "pm", self.Sigma)

        out = []
        sigma = self.Sigma
        mu = self.ObjectArea
        for prop in regions_list:
            prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                            prop.area - mu) / sigma) ** 2
            if return_regions:
                out.append(prop, prob)
            else:
                # prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                #                                                                         prop.area - mu) / sigma) ** 2
                # print(prob)
                # out.append(Measurement(prob, prop.centroid))
                intensities = prop.intensity_image[prop.image]
                mean_int = np.mean(intensities)
                std_int = np.std(intensities)
                out.append(Measurement(prob, [prop.centroid[0], prop.centroid[1], mean_int], data=std_int))
        if return_regions:
            if return_labeled:
                return out, labeled
            else:
                return out
        Positions2D = out
        # Positions2D = self.Detector.detect(~db.getMask(frame=i, layer=0).data.astype(bool))
        Positions2D_cor = []
        for i1, pos1 in enumerate(Positions2D):
            Log_Probability1 = pos1.Log_Probability
            Track_Id1 = pos1.Track_Id
            Frame1 = pos1.Frame
            x1 = pos1.PositionX
            y1 = pos1.PositionY
            z1 = pos1.PositionZ
            # z1 = index_data[int(x1), int(y1)] * 10.
            # PosZ1 = index_data[int(x1), int(y1)]
            inc = 0
            for j1, pos2 in enumerate(Positions2D):
                Log_Probability2 = pos2.Log_Probability
                Log_Probability = (Log_Probability1 + Log_Probability2)/2.
                # Log_Probabilitymax = np.max([Log_Probability1,Log_Probability2])
                x2 = pos2.PositionX
                y2 = pos2.PositionY
                z2 = pos2.PositionZ
                # z2 = index_data[int(x2), int(y2)] * 10.
                # PosZ2 = index_data[int(x2), int(y2)]
                PosZ = (z1 + z2)/2.
                dist = np.sqrt((x1 - x2) ** 2. + (y1 - y2) ** 2.)
                # distz = np.abs(z1 - z2)
                distz = np.abs(z1 - z2)*10.
                if dist < self.distxy_boundary and dist != 0 and distz < self.distz_boundary:
                    x3 = (x1 + x2) / 2.
                    y3 = (y1 + y2) / 2.
                    if [x3, y3, Log_Probability, PosZ] not in Positions2D_cor:
                        Positions2D_cor.append([x3, y3, Log_Probability, PosZ])
                        print("Replaced")
                        # print(x3)
                        # print(y3)
                        # print (Log_Probabilitymax)
                        # print(PosZmax)
                        # print('###')
                    inc += 1
            if inc == 0:
                Positions2D_cor.append([x1, y1, Log_Probability1, z1])
        if only_for_detection:
            return Positions2D_cor
        Positions3D = []
        res = 6.45 / 10
        if self.pre_stitching:
            for pos in Positions2D_cor:
                # posZ = index_data[int(pos[0]), int(pos[1])]
                posZ = pos[3]
                Positions3D.append(Measurement(pos[2], [pos[0] * res, pos[1] * res, posZ * 10]))
            if return_labeled:
                return Positions3D, labeled
            else:
                return Positions3D
        # for pos in Positions2D:
        #     posZ = pos.mean_int  # war mal "Index_Image"
        #     Positions3D.append(Measurement(pos.Log_Probability,
        #                                           [pos.PositionX * res, pos.PositionY * res, posZ * 10],
        #                                           frame=pos.Frame,
        #                                           track_id=pos.Track_Id))
        # if return_labeled:
        #     return Positions3D, labeled
        # else:
        #     return Positions3D

class SimpleAreaDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area, object_number, threshold=None, lower_limit=None, upper_limit=None):
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

    def detect(self, image, return_regions=False, get_all=False, return_labeled=False):
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

        labeled = skimage.measure.label(image, connectivity=2)

        # Filter the regions for area and convexity
        if get_all:
            regions_list = [prop for prop in skimage.measure.regionprops(labeled)]
        else:
            regions_list = [prop for prop in skimage.measure.regionprops(labeled)
                       if (self.UpperLimit > prop.area > self.LowerLimit and prop.solidity > 0.5)]
        if len(regions_list) <= 0:
            return np.array([])

        print("Object Area at ",self.ObjectArea, "pm", self.Sigma)

        out = []
        sigma = self.Sigma
        mu = self.ObjectArea
        for prop in regions_list:
            prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                            prop.area - mu) / sigma) ** 2
            if return_regions:
                out.append(prop, prob)
            else:
                # prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                #                                                                         prop.area - mu) / sigma) ** 2
                # print(prob)
                out.append(Measurement(prob, prop.centroid))
        if return_labeled:
            return out, labeled
        else:
            return out


class AreaDetector(Detector):
    """
    Detector classifying objects by area and number to be used with pengu-track modules.
    """
    def __init__(self, object_area, object_number, threshold=None, lower_limit=None, upper_limit=None):
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
        # bad_ids = [prop.label for prop in skimage.measure.regionprops(labeled) if prop.area < self.ObjectArea]
        # for id in bad_ids:
        #     labeled[labeled == id] = 0
        # regions_list = skimage.measure.regionprops(labeled)
        regions_list = [prop for prop in skimage.measure.regionprops(labeled) if prop.area > self.LowerLimit]

        if len(regions_list) <= 0:
            return np.array([])

        # import matplotlib.pyplot as plt
        # print(self.ObjectArea, len(self.Areas))
        print("Object Area at ",self.ObjectArea, "pm", self.Sigma)
        #if len(self.Areas)>1e3:
        #    bins = int(np.amax(self.Areas)+1)
        #    plt.hist(self.Areas, bins=bins)
        #    hist, bin_edges = np.histogram(self.Areas, bins=bins)
            # plt.figure()
            # hfft = np.fft.fft(hist)
            # plt.plot(hfft)
        #    plt.show()

        out = []
        regions = {}
        [regions.update({prop.label: prop}) for prop in regions_list if prop.area > self.LowerLimit]

        N_max = np.floor(self.ObjectArea/self.Sigma)

        for prop in regions.values():
            n = np.ceil(prop.area/self.ObjectArea)
            if self.LowerLimit < prop.area < self.UpperLimit:
                # if return_regions:
                #     out.append(prop)
                # else:
                #     sigma = self.Sigma
                #     mu = self.ObjectArea
                #     prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5*((prop.area - mu)/sigma) ** 2
                #     out.append(Measurement(prob, prop.centroid))
                out.extend(self.measure(prop, 1, return_regions))
            elif n < N_max:
                pass
                # out.extend(self.measure(prop, n, return_regions))
        return out

    def measure(self, prop, n, return_regions):
        out = []
        if n ==1:
            if return_regions:
                out.append(prop)
            else:
                sigma = self.Sigma
                mu = self.ObjectArea
                prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                            prop.area - mu) / sigma) ** 2
                out.append(Measurement(prob, prop.centroid))
        elif n>1:
            if min(prop.image.shape) < 2:
                if return_regions:
                    out.extend([prop, prop])
                else:
                    bb = np.asarray(prop.bbox)
                    sigma = 2 * self.Sigma
                    mu = 2 * self.ObjectArea
                    prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                                prop.area - mu) / sigma) ** 2
                    out.extend([Measurement(prob, bb[:2]+i/float(n+1)*(bb[2:]-bb[:2])) for i in range(n+1)])
            else:
                distance = ndi.distance_transform_edt(prop.image)
                local_maxi = peak_local_max(distance, indices=False,
                                            labels=prop.image, num_peaks=n)
                markers = ndi.label(local_maxi)[0]
                labels = watershed(-distance, markers, mask=prop.image)
                new_reg = skimage.measure.regionprops(labels)
                new_reg = [new for new in new_reg if new.label <= n]
                if return_regions:
                    out.extend(new_reg)
                else:
                    sigma = 2 * self.Sigma
                    mu = 2 * self.ObjectArea
                    prob = np.log(self.ObjectNumber / (2 * np.pi * sigma ** 2) ** 0.5) - 0.5 * ((
                                                                                                prop.area - mu) / sigma) ** 2
                    out.extend([Measurement(prob, np.asarray(prop.bbox)[:2] + new.centroid) for new in new_reg])
        return out


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
        out = self.segmentate(image)
        self.update(out, image)
        return out

    def update(self,mask, image):
        pass

    def segmentate(self, image):
        pass

class TresholdSegmentation(Segmentation):

    def __init__(self, treshold):
        super(TresholdSegmentation, self).__init__()

        self.width = None
        self.height = None
        self.SegMap = None
        self.Treshold = int(treshold)
        self.Skale = None

    def segmentate(self, image):
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
            self.SegMap = (np.sum(data**2, axis=-1)**0.5/data.shape[-1]**0.5 > self.Treshold).astype(bool)
        elif len(data.shape) == 2:
            print(np.amin(data), np.amax(data))
            self.SegMap = (data > self.Treshold).astype(bool)
        else:
            raise ValueError('False format of data.')
        return self.SegMap

    def update(self,mask, image):
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

    def segmentate(self, image):
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


        # data_mean = filters.rank.mean(data, self.selem)
        # data_std = filters.rank.mean(data**2, self.selem) - data_mean**2
        if len(data.shape) == 3:
            # self.SegMap = (np.sum(data_std**2, axis=-1)**0.5/data.shape[-1]**0.5 > self.Treshold**2).astype(bool)
            self.SegMap = (self.local_std(data) < self.Treshold).astype(bool)
        elif len(data.shape) == 2:
            # print(np.amin(data), np.amax(data))
            std = self.local_std(data)
            self.SegMap = (self.local_std(data) < self.Treshold).astype(bool)
        else:
            raise ValueError('False format of data.')
        return self.SegMap

    def update(self,mask, image):
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

    def segmentate(self, image):
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

    def update(self, mask, image):

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
        return super(ViBeSegmentation, self).detect(image)

        # data = np.array(image, ndmin=2)
        #
        # if self.width is None or self.height is None:
        #     self.width, self.height = data.shape[:2]
        # this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))
        #
        # if this_skale == 0:
        #     this_skale = self.Skale
        # if self.Skale is None:
        #     self.Skale = this_skale
        #
        # data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        # if self.Samples is None:
        #     self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        # if self.SegMap is None:
        #     self.SegMap = np.ones((self.width, self.height), dtype=bool)
        #
        # if len(data.shape) == 3:
        #     self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1])
        #                            > self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        # elif len(data.shape) == 2:
        #     self.SegMap = (np.sum((np.abs(self.Samples.astype(np.int32)-data) > self.R), axis=0, dtype=np.uint8)
        #                    >= self.N_min).astype(bool)
        # else:
        #     raise ValueError('False format of data.')
        #
        # image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & self.SegMap
        #
        # sample_index = np.random.randint(0, self.N)
        # self.Samples[sample_index][image_mask] = (data[image_mask]).astype(np.uint8)
        #
        # n = np.sum(image_mask)
        # if n > 0 and do_neighbours:
        #     x, y = np.array(np.meshgrid(np.arange(self.width), np.arange(self.height))).T[image_mask].T
        #     rand_x, rand_y = np.asarray(map(self._Neighbour_Map.get, np.random.randint(0, 8, size=n))).T
        #     rand_x += x
        #     rand_y += y
        #     notdoubled = ~(np.asarray([x_ in rand_x[:i] for i, x_ in enumerate(rand_x)]) &
        #                    np.asarray([y_ in rand_y[:i] for i, y_ in enumerate(rand_y)]))
        #     notborder = np.asarray(((0 <= rand_y) & (rand_y < self.height)) & ((0 <= rand_x) & (rand_x < self.width)))
        #     mask = notborder & notdoubled
        #     x = x[mask]
        #     y = y[mask]
        #     rand_x = rand_x[mask]
        #     rand_y = rand_y[mask]
        #     neighbours = np.zeros_like(image_mask, dtype=bool)
        #     neighbours[rand_x, rand_y] = True
        #     mask1 = np.zeros_like(image_mask, dtype=bool)
        #     mask1[x, y] = True
        #     try:
        #         self.Samples[sample_index][neighbours] = (data[mask1]).astype(np.uint8)
        #     except ValueError:
        #         print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)
        #         raise
        # print("Updated %s pixels" % n)
        # out = self.segmentate(image, do_neighbours=do_neighbours)
        # self.update(out, image, do_neighbours=do_neighbours)
        # return out

    def segmentate(self, image, do_neighbours=True):
        super(ViBeSegmentation, self).segmentate(image)
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
        return self.SegMap

    def update(self, mask, image, do_neighbours=True):

        data = np.array(image, ndmin=2)

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]
        this_skale = np.mean((np.sum(data.astype(np.uint32)**2, axis=-1)**0.5))

        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)

        image_mask = (np.random.rand(self.width, self.height)*self.Phi < 1) & mask

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
        # else:
        #     raise ValueError('False format of data.')

        return self.SegMap


class SiAdViBeSegmentation(Segmentation):
    """
    Segmentation method comparing input images to image-background buffer. Able to learn new background information.
    This Version uses also Size Adjustion, an adapted ortho-projection,
     conserving the original size of an object in a plane.
    """
    function = None

    def __init__(self, horizonmarkers, f, sensor_size, pengu_markers, h_p, max_dist, init_image,  n=20, r=15, n_min=1, phi=16, camera_h = None, log=True):
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
        self.Pengu_Markers = pengu_markers
        self.F = f
        self.Sensor_Size = sensor_size
        self.h_p = h_p
        self.w_p = 0.21
        self.Max_Dist = max_dist


        data = np.array(init_image, ndmin=2)

        self.SegMap = None
        self._Neighbour_Map = {0: [-1, -1],
                               1: [-1, 0],
                               2: [-1, 1],
                               3: [0, -1],
                               4: [0, 1],
                               5: [1, -1],
                               6: [1, 0],
                               7: [1, 1]}
        self.width = data.shape[1]
        self.height = data.shape[0]

        x, y = self.Horizonmarkers
        m, t = np.polyfit(x, y, 1)  # linear fit for camera role
        angle_m = np.arctan(m)
        x_0 = self.width / 2
        y_0 = x_0 * m + t

        x_p1, y_p1, x_p2, y_p2 = self.Pengu_Markers
        x_p1 = (x_p1 - x_0) * np.cos(angle_m) + x_0 - (y_p1 - y_0) * np.sin(angle_m)
        y_p1 = (x_p1 - x_0) * np.sin(angle_m) + (y_p1 - y_0) * np.cos(angle_m) + y_0

        x_p2 = (x_p2 - x_0) * np.cos(angle_m) + x_0 - (y_p2 - y_0) * np.sin(angle_m)
        y_p2 = (x_p2 - x_0) * np.sin(angle_m) + (y_p2 - y_0) * np.cos(angle_m) + y_0

        # calc phi (tilt of the camera)
        self.Phi = self.calc_phi([x, y], self.Sensor_Size, [self.height, self.width], f)
        # pp = 20 * np.pi / 180.

        # get the lower ones of the markers
        args = np.argmax([y_p1, y_p2], axis=0)
        y11 = np.asarray([[y_p1[i], y_p2[i]][a] for i, a in enumerate(args)])
        x11 = np.asarray([[x_p1[i], x_p2[i]][a] for i, a in enumerate(args)])

        # calc height above plane
        tt = self.calc_theta([x11, y11], self.Sensor_Size, [self.height, self.width], f)
        gg = self.calc_gamma([x_p1, y_p1], [x_p2, y_p2], self.Sensor_Size, [self.height, self.width], f)
        hh = self.calc_height(tt, gg, self.Phi, self.h_p)

        if camera_h is None:
            self.camera_h = np.mean(hh)
            print("Height", np.mean(hh), np.std(hh) / len(hh) ** 0.5)
        else:
            self.camera_h = float(camera_h)

        # Initialize grid
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # Grid has same aspect ratio as picture

        # counter-angle to phi is used in further calculation
        phi_ = np.pi/2 - self.Phi


        # calculate Maximal analysed distance
        self.c = 1. / (self.camera_h / self.h_p - 1.)
        self.y_min = np.asarray([0, self.camera_h*np.tan(phi_-np.arctan(self.Sensor_Size[1]/2./f)), -self.camera_h])
        # y_min = self.camera_h*np.tan(phi_-np.arctan(self.Sensor_Size[1]/2./f))
        # print("y_min at ",y_min)
        # CC = self.h_p/np.log(1+self.c)
        # self.Max_Dist = - CC * lambertw(-self.y_min[1]/CC)
        # self.Max_Dist = [y_min*np.exp(-1*lambertw(-y_min*(2j*np.pi*n*np.log(1+self.c)/self.h_p), k=-1)) for n in range(-10,10)]
        # print(np.nanargmin([m.imag for m in self.Max_Dist]))
        # self.Max_Dist = self.Max_Dist[np.nanargmax([abs(m) for m in self.Max_Dist])]
        # max_dist = self.Max_Dist
        # print(self.Max_Dist)
        # self.Max_Dist = 1250#abs(self.Max_Dist)
        # max_dist = 1250#abs(max_dist)
        # raise NotImplementedError

        # calculate Penguin-Projection Size
        self.Penguin_Size = np.log(1 + self.c) * self.height / np.log(self.Max_Dist / self.y_min[1])

        # warp the grid
        self.Res = max_dist/self.height
        yy = yy * (max_dist/self.height)
        xx = (xx-self.width/2.) * (max_dist/self.height)
        # x_s is the crossing-Point of camera mid-beam and grid plane
        self.x_s = np.asarray([0, np.tan(phi_)*self.camera_h, -self.camera_h])
        self.x_s_norm = np.linalg.norm(self.x_s)
        # vector of maximal distance to camera (in y)
        self.y_max = np.asarray([0, max_dist, -self.camera_h])
        self.y_max_norm = np.linalg.norm(self.y_max)
        self.alpha_y = np.arccos(np.dot(self.y_max.T, self.x_s).T/(self.y_max_norm*self.x_s_norm)) * -1
        if log:
            warped_xx, warped_yy = self.warp_log([xx, yy])
        else:
            warped_xx, warped_yy = self.warp_orth([xx, yy])
        # reshape grid points for image interpolation
        grid = np.asarray([warped_xx.T, warped_yy.T]).T
        self.grid = grid.T.reshape((2, self.width * self.height))


        # Initialize with warped image
        # data = self.horizontal_equalisation(data)

        if len(data.shape) == 3:
            self.Samples = np.tile(data, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
        elif len(data.shape) == 2:
            self.Samples = np.tile(data, (self.N, 1, 1,))

        else:
            raise ValueError('False format of data.')
        self.Skale = np.mean((np.sum(np.mean(self.Samples, axis=0).astype(np.uint32)**2, axis=-1)**0.5))

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

        # super(SiAdViBeSegmentation, self).detect(image)

        h_eq_SegMap = self.segmentate(image, do_neighbours=do_neighbours)
        self.update(self.SegMap, image, do_neighbours=do_neighbours)
        return h_eq_SegMap

    def segmentate(self, image, do_neighbours=True):
        super(SiAdViBeSegmentation, self).segmentate(image)

        data = image

        if self.width is None or self.height is None:
            self.width, self.height = data.shape[:2]

        this_skale = np.mean((np.sum(data.astype(np.uint16) ** 2, axis=-1) ** 0.5))


        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float) * (self.Skale / this_skale)).astype(np.int32)

        # data = self.horizontal_equalisation(data)

        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((self.height, self.width), dtype=bool)

        if len(data.shape) == 3:
            self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1])
                                   > self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)
        elif len(data.shape) == 2:
            # if self.function is None:
            #     samples = T.tensor3("samples", dtype="int16")
            #     t_data = T.matrix("data", dtype="int16")
            #     R = T.scalar(dtype="int16")
            #     N_min = T.scalar(dtype="int16")
            #     result = (T.sum((abs(samples - t_data) > R), axis=0) >= N_min).astype("bool")
            #     self.function = theano.compile.function([samples, t_data, R, N_min], result, allow_input_downcast=True)
            # SegMap = self.function(self.Samples, data, self.R, self.N_min)
            self.SegMap = (np.sum((np.abs(self.Samples.astype(np.int32)-data) > self.R), axis=0, dtype=np.uint8)
                           >= self.N_min).astype(bool)

        else:
            raise ValueError('False format of data.')
        return self.horizontal_equalisation(self.SegMap).astype(bool)

    def update(self,mask, image, do_neighbours=True):
        super(SiAdViBeSegmentation, self).update(mask, image)

        data = image
        image_mask = (np.random.rand(self.height, self.width)*self.Phi < 1) & self.SegMap
        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(np.uint8)

        if do_neighbours:
            n = np.sum(image_mask)
        else:
            n=0
        if n > 0 and do_neighbours:
            x, y = np.array(np.meshgrid(np.arange(self.height), np.arange(self.width))).T[image_mask].T
            rand_x, rand_y = np.asarray(map(self._Neighbour_Map.get, np.random.randint(0, 8, size=n))).T
            rand_x += x
            rand_y += y
            notdoubled = ~(np.asarray([x_ in rand_x[:i] for i, x_ in enumerate(rand_x)]) &
                           np.asarray([y_ in rand_y[:i] for i, y_ in enumerate(rand_y)]))
            notborder = np.asarray(((0 <= rand_y) & (rand_y < self.width)) & ((0 <= rand_x) & (rand_x < self.height)))
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
        print("Updated %s pixels" % np.sum(image_mask))
        # return self.horizontal_equalisation(self.SegMap).astype(bool)

    def stepA0(self, data):
        return (self.Samples.astype(np.int32)-data)**2
    def stepA(self, data):
        return np.sum(data, axis=-1)**0.5/np.sqrt(data.shape[-1])
    def stepB(self, data):
        return np.sum((data > self.R), axis=0, dtype=np.uint8)
    def stepC(self, data):
        return (data>= self.N_min).astype(bool)

    def horizontal_equalisation(self, image):
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
        h: float, optional
        Height of the camera over the shown plane.
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

        if len(image.shape) == 3:
            # split the image in colors and perform interpolation
            return np.asarray([map_coordinates(i, self.grid[:, ::-1], order=0).reshape((self.width, self.height))[::-1] for i in image.T]).T
        elif len(image.shape) == 2:
            return np.asarray(img_as_uint(map_coordinates(image.T, self.grid[:, ::-1], order=0)).reshape((self.width, self.height))[::-1]).T
        else:
            raise ValueError("The given image is whether RGB nor greyscale!")

    # Define Warp Function
    def warp_log(self, xy):
        xx_, yy_ = xy

        #xx_ /= (self.Max_Dist/self.height)/(self.h_p/self.Penguin_Size)
        # vectors of every grid point in the plane (-h)
        yy_ /= self.Max_Dist  # linear 0 to 1
        yy_ *= np.log(self.Max_Dist / self.y_min[1])  # linear 0 to log(max/min)
        yy_ = self.y_min[1] * np.exp(yy_)  # exponential from y_min to y_max

        # initialize 3d positions
        coord = np.asarray([xx_, yy_, -self.camera_h * np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, self.x_s).T / (coord_norm * self.x_s_norm))  # * np.sign(np.tan(phi_)*h-yy_)
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        theta = np.sum((np.cross(coord.T, self.x_s) * np.cross(self.y_max, self.x_s)).T
                                 , axis=0) / (coord_norm * self.x_s_norm * np.sin(alpha) *
                                              self.y_max_norm * self.x_s_norm * np.sin(self.alpha_y))
        try:
            theta[theta>1.] = 1.
            theta[theta<-1.] = -1.
        except TypeError:
            if theta > 1.:
                theta = 1.
            elif theta < -1:
                theta = -1.
            else:
                pass
        theta = np.arccos(theta) * np.sign(xx_)
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha) * self.F
        warp_xx = np.sin(theta) * r * self.width / self.Sensor_Size[0] + self.width / 2.
        warp_yy = np.cos(theta) * r * self.height / self.Sensor_Size[1] + self.height / 2.
        return warp_xx, warp_yy

    # Define Warp Function
    def warp_orth(self, xy):
        # if True:
        #     return xy
        xx_, yy_ = xy
        # vectors of every grid point in the plane (-h)
        coord = np.asarray([xx_, yy_, -self.camera_h*np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, self.x_s).T/(coord_norm*self.x_s_norm))
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        theta = np.sum((np.cross(coord.T, self.x_s) * np.cross(self.y_max, self.x_s)).T
                                 , axis=0) / (coord_norm * self.x_s_norm * np.sin(alpha) *
                                              self.y_max_norm * self.x_s_norm * np.sin(self.alpha_y))
        try:
            theta[theta>1.] = 1.
            theta[theta<-1.] = -1.
        except TypeError:
            if theta > 1.:
                theta = 1.
            elif theta < -1:
                theta = -1.
            else:
                pass
        theta = np.arccos(theta) * np.sign(xx_)
        # print(np.nanmin(theta), np.nanmax(theta))
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha)*self.F
        warp_xx = np.sin(theta)*r*self.width/self.Sensor_Size[0] + self.width/2.
        warp_yy = np.cos(theta)*r*self.height/self.Sensor_Size[1] + self.height/2.
        return warp_xx, warp_yy

    def back_warp_orth(self, xy):
        if True:
            return xy
        warp_xx, warp_yy = xy
        warp_xx = (warp_xx-self.width/2.)*self.Sensor_Size[0]/self.width
        warp_yy = (warp_yy-self.height/2.)*self.Sensor_Size[1]/self.height
        # Calculate angles in Camera-Coordinates
        theta = np.arctan2(warp_yy,
                           warp_xx)
        # theta = np.pi/2.-theta
        r = (warp_xx**2+warp_yy**2)**0.5

        # theta = np.arccos((-1) * (warp_xx-self.width/2.)*self.Sensor_Size[0]/self.width/r)

        # print(np.amin(r), np.amax(r))
        alpha = np.arctan(r/self.F)
        lam = -self.camera_h/(np.sin(theta)*np.sin(np.pi/2.-self.Phi)*np.tan(alpha)-np.cos(np.pi/2.-self.Phi))
        # print(np.amin(lam), np.amax(lam))
        # xx_ = -lam * (np.tan(alpha)*np.sin(np.pi/2.-self.Phi)*np.cos(theta)-np.sin(np.pi/2.-self.Phi))
        # yy_ = lam*np.tan(alpha)*np.sin(theta)
        xx_ = - np.tan(alpha) * lam * np.cos(theta)
        yy_ = lam * np.cos(np.pi/2. - self.Phi) * np.tan(alpha) * np.sin(theta) - lam * np.sin(np.pi/2 - self.Phi)
        return xx_, -yy_

    def calc_phi(self, horizonmarkers, sensor_size, image_size, f):
        x_h, y_h = np.asarray(horizonmarkers)

        y, x = image_size

        # linear fit and rotation to compensate incorrect camera alignment
        m, t = np.polyfit(x_h, y_h, 1)  # linear fit

        x_s, y_s = sensor_size

        # correction of the Y-axis section (after image rotation)
        t += m * x / 2

        print("Role at %s"%(np.arctan(m)*180/np.pi))

        # Calculate Camera tilt
        return np.arctan((t / y - 1/2.) * (y_s / f))*-1

    def calc_theta(self, theta_markers, sensor_size, image_size, f):
        x_t1, y_t1 = theta_markers
        y, x = image_size
        x_s, y_s = sensor_size
        # y_t1 = (y_t1-y/2.)*y_s/y
        # r =  (((x_t1-x/2.)*(x_s/x))**2+((y_t1-y/2.)*(y_s/y))**2)**0.5
        return np.arctan2((y_t1-y/2.)*(y_s/y), f)

    def calc_gamma(self, markers1, markers2, sensor_size, image_size, f):
        x1, y1 = markers1
        x2, y2 = markers2
        y, x = image_size
        x_s, y_s = sensor_size
        x1 = (x1-x/2.)*(x_s/x)
        y1 = (y1-y/2.)*(y_s/y)
        x2 = (x2-x/2.)*(x_s/x)
        y2 = (y2-y/2.)*(y_s/y)
        return np.arccos((x1*x2+y1*y2+f**2)/((x1**2+y1**2+f**2)*(x2**2+y2**2+f**2))**0.5)

    def calc_height(self, the, gam, p, h_t):
        alpha = np.pi/2.-p-the
        return h_t*np.abs(np.cos(alpha)*np.sin(np.pi-alpha-gam)/np.sin(gam))

    def log_to_orth(self, xy):
        # self.height / self.Sensor_Size[1] + self.height / 2.
        xx_, yy_ = xy
        xx_ -= self.width/2.
        #xx_ /= (self.Max_Dist/self.height)/(self.h_p/self.Penguin_Size)
        xx_ += self.width/2.
        yy_ = self.height - self.y_min[1]*np.exp((self.height-yy_)/self.height*np.log(self.Max_Dist/self.y_min[1]))/self.Res
        return xx_, yy_

    def orth_to_log(self, xy):
        # self.height / self.Sensor_Size[1] + self.height / 2.
        xx_, yy_ = xy
        xx_ -= self.width/2.
        #xx_ /= (self.h_p/self.Penguin_Size)/(self.Max_Dist/self.height)
        xx_ += self.width/2.
        yy_ = self.height - np.log((self.height - yy_)*(self.Res/self.y_min[1]))*(self.height/np.log(self.Max_Dist/self.y_min[1]))
        return xx_, yy_


def rgb2gray(rgb):
    # rgb = np.asarray(rgb)
    dt = rgb.dtype
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(dt)
