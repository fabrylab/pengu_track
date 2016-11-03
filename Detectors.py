import numpy as np
import skimage.feature
import skimage.transform
import skimage.measure
import skimage.filters as filters
import skimage.morphology

class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

    def detect(self, image):
        return np.random.rand(2)

class ViBeDetector(Detector):
    def __init__(self, n=20, r=15, n_min=1, phi=16, init_image=None, object_size=None, object_number=None):
        super(ViBeDetector, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)


        if object_size is None:
            object_size = 10
        self.ObjectSize = int(object_size)
        if object_number is None:
            object_number = 10
        self.ObjectNumber = int(object_number)

        self.Treshold = None

        self.Skale = None
        self.Samples = None
        if init_image is not None:
            data = np.array(init_image, ndmin=3)
            self.Samples = np.tile(data, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
            self.Skale = np.mean(np.linalg.norm(self.Samples, axis=-1).astype(float))
        self.SegMap = None
        self._Neighbour_Map = {0: [-1, -1],
                               1: [-1, 0],
                               2: [-1, 1],
                               3: [0, -1],
                               4: [0, 1],
                               5: [1, -1],
                               6: [1, 0],
                               7: [1, 1]}

    def detect(self, image):
        super(ViBeDetector, self).detect(image)
        width, height = image.shape[:2] #image.getShape()
        data = np.array(image, ndmin=3)
        this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))
        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*(self.Skale/this_skale)).astype(np.int32)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.ones((width, height), dtype=bool)

        self.SegMap = (np.sum((np.sum((self.Samples.astype(np.int32)-data)**2, axis=-1)**0.5/np.sqrt(data.shape[-1]) >
                               self.R), axis=0, dtype=np.uint8) >= self.N_min).astype(bool)

        # selem = np.ones((self.ObjectSize, self.ObjectSize))
        # self.SegMap = skimage.morphology.binary_dilation(
        #                     skimage.morphology.binary_erosion(
        #                         self.SegMap, selem=selem), selem=selem)

        image_mask = (np.random.rand(width, height)*self.Phi < 1) & self.SegMap

        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask] = (data[image_mask]).astype(np.uint8)

        n = np.sum(image_mask)
        print("Updated %s pixels"%n)
        if n > 0:
            x, y = np.array(np.meshgrid(np.arange(width), np.arange(height))).T[image_mask].T
            randX, randY = np.asarray(map(self._Neighbour_Map.get, np.random.randint(0, 8, size=n))).T
            randX += x
            randY += y
            notdoubled = ~(np.asarray([x_ in randX[:i] for i, x_ in enumerate(randX)]) &
                           np.asarray([y_ in randY[:i] for i, y_ in enumerate(randY)]))
            notborder = np.asarray(((0 <= randY) & (randY < height)) & ((0 <= randX) & (randX < width)))
            mask = notborder & notdoubled
            x = x[mask]
            y = y[mask]
            randX = randX[mask]
            randY = randY[mask]
            neighbours = np.zeros_like(image_mask, dtype=bool)
            neighbours[randX, randY] = True
            mask1 = np.zeros_like(image_mask, dtype=bool)
            mask1[x, y] = True
            try:
                self.Samples[sample_index][neighbours] = (data[mask1]).astype(np.uint8)
            except (ValueError, e):
                print(e)
                print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)
                raise

        if self.Treshold is None:
            filtered = skimage.filters.laplace(skimage.filters.gaussian(self.SegMap.astype(np.uint8), self.ObjectSize))
            print(filtered.min(), filtered.max(), filtered.mean(), filtered.std(), self.SegMap.dtype)
            filtermax = filtered.max()
            filtermin = filtered.min()
            n = 1000
            for i in range(n):
                treshold = filtermin + (filtermax - filtermin) * (1. - i/(n-1.))
                regions = skimage.measure.regionprops(skimage.measure.label(filtered > treshold))
                if len(regions) > self.ObjectNumber:
                    self.Treshold = treshold
                    print("Set Treshold to %s"%treshold)
                    break
        regions = skimage.measure.regionprops(
                    skimage.measure.label(
                        skimage.filters.laplace(
                            skimage.filters.gaussian(self.SegMap.astype(np.uint8), self.ObjectSize)) > self.Treshold
                        ))
        if len(regions) > 0:
            detections = np.array([[props.centroid[0], props.centroid[1], props.area] for props in regions], ndmin=2)

        return detections


# class AdvancedViBeDetector(Detector):
#     def __init__(self, n=20, r=15, n_min=1, phi=16, init_image=None):
#         super(AdvancedViBeDetector, self).__init__()
#         self.N = int(n)
#         self.R = np.uint16(r)
#         self.N_min = int(n_min)
#         self.Phi = int(phi)
#         self.Skale = None
#         self.Samples = None
#         self.Blobs = None
#         if init_image is not None:
#             data = np.array(init_image, ndmin=3)
#             std = data.std()
#             print(np.random.normal(data, scale=std).shape)
#             self.Samples = np.tile(data, self.N).reshape(data.shape+(self.N,)).transpose((3, 0, 1, 2))
#             self.Skale = np.mean(np.linalg.norm(self.Samples, axis=-1).astype(float))
#         self.SegMap = None
#         self._Neighbour_Map = {0: [-1, -1],
#                                1: [-1, 0],
#                                2: [-1, 1],
#                                3: [0, -1],
#                                4: [0, 1],
#                                5: [1, -1],
#                                6: [1, 0],
#                                7: [1, 1]}
#
#     def detect(self, image, blob_size=1., blob_treshold=0):
#         super(AdvancedViBeDetector, self).detect(image)
#         width, height = image.getShape()
#         data = np.array(image.data, ndmin=3)
#         this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))
#         if this_skale == 0:
#             this_skale = self.Skale
#         if self.Skale is None:
#             self.Skale = this_skale
#
#         data = (data.astype(float)*self.Skale/this_skale).astype(np.uint8)
#         if self.Samples is None:
#             self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
#         if self.SegMap is None:
#             self.SegMap = np.zeros((width, height), dtype=bool)
#
#         self.SegMap = (np.sum((np.linalg.norm(self.Samples-data, axis=-1).astype(np.uint32) > (self.R)**2), axis=0, dtype=np.uint32) >= self.N_min).astype(bool) ^ True
#
#         filtered_seg = np.abs(filters.laplace(filters.gaussian(self.SegMap, blob_size)))
#         filtered_seg = (filtered_seg-filtered_seg.min())/(filtered_seg.max()-filtered_seg.min())
#         if self.Blobs is None:
#             self.Blobs = (filtered_seg > 0.2)
#         filtered_dat = filters.laplace(filters.gaussian(np.linalg.norm(data, axis=-1).astype(np.uint32), blob_size))
#         filtered_dat = (filtered_dat-filtered_dat.min())/(filtered_dat.max()-filtered_dat.min())
#         self.Blobs = True^(filtered_seg > 0.3)#|(self.Blobs & (filtered_dat > 0.6)))^True#(filtered_seg > 0.8) | (self.Blobs & (filtered_dat > 0.8))
#
#         image_mask = (np.random.rand(width, height)*self.Phi < 1) & (self.SegMap & self.Blobs)
#
#         sample_index = np.random.randint(0, self.N)
#         self.Samples[sample_index][image_mask ^ True] = data[image_mask ^ True]
#
#         n = np.sum(image_mask)
#         print("Updated %s pixels"%n)
#         if n > 0:
#             x, y = np.array(np.meshgrid(np.arange(width), np.arange(height))).T[image_mask].T
#             randX, randY = np.array([self._Neighbour_Map[k] for k in np.random.randint(0, 8, size=n)]).T
#             neighbours = np.zeros_like(image_mask, dtype=bool)
#             for i in range(n):
#                 k = min(max(randX[i]+x[i], 0), width-1)
#                 l = min(max(randY[i]+y[i], 0), height-1)
#                 if neighbours[k, l]:
#                     image_mask[x[i], y[i]] = False
#                 neighbours[k, l] = True
#             j = np.random.randint(0, self.N)
#             try:
#                 self.Samples[sample_index][neighbours] = np.array(data[image_mask])
#             except ValueError, e:
#                 print(e)
#                 print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)
#         regions = skimage.measure.regionprops(
#                     skimage.measure.label(
#                         skimage.filters.laplace(
#                             skimage.filters.gaussian(self.SegMap.astype(np.uint8), self.ObjectSize)) > 7e-6
#                         ))
#         detections = np.array([])
#         if len(regions) > 0:
#             detections = np.array([[props.centroid[0], props.centroid[1], props.area] for props in regions], ndmin=2)
#
#         return detections