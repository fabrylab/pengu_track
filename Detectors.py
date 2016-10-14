import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage.filters as filters
from time import time

class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

    def detect(self, image):
        return np.random.rand(2)

class ViBeDetector(Detector):
    def __init__(self, n=20, r=15, n_min=1, phi=16, init_image=None):
        super(ViBeDetector, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Skale = None
        self.Samples = None
        if init_image is not None:
            data = np.array(init_image, ndmin=3)
            std = data.std()
            print(np.random.normal(data, scale=std).shape)
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
        width, height = image.getShape()
        data = np.array(image.data, ndmin=3)
        this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))
        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*self.Skale/this_skale).astype(np.uint8)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.zeros((width, height), dtype=bool)

        self.SegMap = (np.sum((np.linalg.norm(self.Samples-data, axis=-1).astype(np.uint32) > (self.R)**2), axis=0, dtype=np.uint32) >= self.N_min).astype(bool) ^ True
        image_mask = (np.random.rand(width, height)*self.Phi < 1) & (self.SegMap)

        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask] = data[image_mask]

        n = np.sum(image_mask)
        print("Updated %s pixels"%n)
        if n > 0:
            x, y = np.array(np.meshgrid(np.arange(width), np.arange(height))).T[image_mask].T
            randX, randY = np.array([self._Neighbour_Map[k] for k in np.random.randint(0, 8, size=n)]).T
            neighbours = np.zeros_like(image_mask, dtype=bool)
            for i in range(n):
                k = min(max(randX[i]+x[i], 0), width-1)
                l = min(max(randY[i]+y[i], 0), height-1)
                if neighbours[k, l]:
                    image_mask[x[i], y[i]] = False
                neighbours[k, l] = True
            j = np.random.randint(0, self.N)
            try:
                self.Samples[sample_index][neighbours] = np.array(data[image_mask])
            except ValueError, e:
                print(e)
                print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)


class AdvancedViBeDetector(Detector):
    def __init__(self, n=20, r=15, n_min=1, phi=16, init_image=None):
        super(AdvancedViBeDetector, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Skale = None
        self.Samples = None
        self.Blobs = None
        if init_image is not None:
            data = np.array(init_image, ndmin=3)
            std = data.std()
            print(np.random.normal(data, scale=std).shape)
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

    def detect(self, image, blob_size=1., blob_treshold=0):
        super(AdvancedViBeDetector, self).detect(image)
        width, height = image.getShape()
        data = np.array(image.data, ndmin=3)
        this_skale = np.mean(np.linalg.norm(data, axis=-1).astype(float))
        if this_skale == 0:
            this_skale = self.Skale
        if self.Skale is None:
            self.Skale = this_skale

        data = (data.astype(float)*self.Skale/this_skale).astype(np.uint8)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
        if self.SegMap is None:
            self.SegMap = np.zeros((width, height), dtype=bool)

        self.SegMap = (np.sum((np.linalg.norm(self.Samples-data, axis=-1).astype(np.uint32) > (self.R)**2), axis=0, dtype=np.uint32) >= self.N_min).astype(bool) ^ True

        filtered_seg = np.abs(filters.laplace(filters.gaussian(self.SegMap, blob_size)))
        filtered_seg = (filtered_seg-filtered_seg.min())/(filtered_seg.max()-filtered_seg.min())
        if self.Blobs is None:
            self.Blobs = (filtered_seg > 0.2)
        filtered_dat = filters.laplace(filters.gaussian(np.linalg.norm(data, axis=-1).astype(np.uint32), blob_size))
        filtered_dat = (filtered_dat-filtered_dat.min())/(filtered_dat.max()-filtered_dat.min())
        self.Blobs = True^(filtered_seg > 0.3)#|(self.Blobs & (filtered_dat > 0.6)))^True#(filtered_seg > 0.8) | (self.Blobs & (filtered_dat > 0.8))

        image_mask = (np.random.rand(width, height)*self.Phi < 1) & (self.SegMap & self.Blobs)

        sample_index = np.random.randint(0, self.N)
        self.Samples[sample_index][image_mask ^ True] = data[image_mask ^ True]

        n = np.sum(image_mask)
        print("Updated %s pixels"%n)
        if n > 0:
            x, y = np.array(np.meshgrid(np.arange(width), np.arange(height))).T[image_mask].T
            randX, randY = np.array([self._Neighbour_Map[k] for k in np.random.randint(0, 8, size=n)]).T
            neighbours = np.zeros_like(image_mask, dtype=bool)
            for i in range(n):
                k = min(max(randX[i]+x[i], 0), width-1)
                l = min(max(randY[i]+y[i], 0), height-1)
                if neighbours[k, l]:
                    image_mask[x[i], y[i]] = False
                neighbours[k, l] = True
            j = np.random.randint(0, self.N)
            try:
                self.Samples[sample_index][neighbours] = np.array(data[image_mask])
            except ValueError, e:
                print(e)
                print(np.sum(neighbours), np.sum(image_mask), x.shape, y.shape)


if __name__ == '__main__':
    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()

    import clickpoints
    import skimage.feature
    import skimage.transform
    from Filters import AdvancedKalmanFilter
    from Filters import MultiFilter
    from Models import VariableSpeed
    import scipy.stats as ss

    model = VariableSpeed(2)
    ucty = 10.26#optimal['x']
    xy_uncty = ucty
    vxvy_uncty = ucty
    meas_uncty = 10
    X = np.zeros(4).T
    P = np.diag([ucty, ucty, ucty, ucty])
    # Q = np.diag([0., vxvy_uncty, 0, vxvy_uncty])  # Prediction uncertainty
    Q = np.diag([vxvy_uncty, vxvy_uncty])  # Prediction uncertainty
    R = np.diag([meas_uncty, meas_uncty])  # Measurement uncertainty

    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)
    MultiKal = MultiFilter(AdvancedKalmanFilter, model, [X], np.array([vxvy_uncty, vxvy_uncty]),
                           np.array([meas_uncty, meas_uncty]), meas_dist=Meas_Dist, state_dist=State_Dist)


    db = clickpoints.DataFile("click2.cdb")
    images = db.getImageIterator(start_frame=30, end_frame=40)

    init = np.array(np.median([np.array(i.data, dtype=np.uint8) for i in images], axis=0), dtype=np.uint8)

    VB = ViBeDetector(init_image=init, phi=64)
    print('Initialized')
    ## plt.imshow(Image.fromarray(next(images).data))
    # plt.ion()
    ## s = 0
    marker_type = db.setMarkerType(name="ViBe_Marker", color="#FF0000")
    db.deleteMarkers(type=marker_type)
    marker_type2 = db.setMarkerType(name="ViBe_Kalman_Marker", color="#00FF00", mode=db.TYPE_Track)
    db.deleteMarkers(type=marker_type2)

    db.deleteTracks()
    images = db.getImageIterator()
    for image in images:
        i = image.get_id()
        MultiKal.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        VB.detect(image)
        db.setMask(image=image, data=(VB.SegMap ^ True).astype(np.uint8))
        print("Mask save")
        n = 8
        blobs = skimage.feature.blob_log(skimage.transform.downscale_local_mean(VB.SegMap, (n, n)),
                                         min_sigma=3./(n**0.5))
        try:
            db.setMarkers(image=image, x=blobs.T[1]*n, y=blobs.T[0]*n, type=marker_type)
            print("Markers Saved (%s)" % blobs.shape[0])
            print([MultiKal.Filters[k].X.values() for k in MultiKal.Filters.keys()])
            MultiKal.update(z=blobs.T[:2].T, i=i)
            for k in MultiKal.Filters.keys():
                x, y = model.measure(MultiKal.Filters[k].X[i])
                if x != np.nan and y != np.nan:
                    try:
                        db.setMarker(image=image, track=k, x=x,y=y)
                    except:
                        db.setTrack(type=marker_type2, id=k)
                        db.setMarker(image=image, track=k, x=x, y=y)


        except IndexError:
            print("No Markers")
            pass

        db.setTrack()
        db.setMarkers()
        # plt.imshow(skimage.transform.downscale_local_mean(MAP, (n, n)))
        # plt.imshow(VB.Blobs&VB.SegMap)
        # plt.savefig('./adelie_data/Tracked/Image%04d.png'%s)
        # s += 1
        # plt.pause(0.05)

    print('done with ViBe')


    #
    # import skimage.filters
    # import skimage.feature
    # import skimage.transform
    # import skimage.color
    # plt.ion()
    # images = db.getImageIterator(start_frame=40)
    # p1 = plt.plot([], [], 'wo')
    # for image in images:
    #     MAP = (db.getMask(image=image).data.astype(bool) ^ True)#.astype(np.uint8)
    #     PIC = np.linalg.norm(image.data, axis=-1) < 30*3**0.5
    #     n = 8
    #     blobs = skimage.feature.blob_log(skimage.transform.downscale_local_mean(MAP,(n,n)), min_sigma=5./(n**0.5))
    #     plt.imshow(skimage.transform.downscale_local_mean(MAP,(n,n)))
    #     # blobs = skimage.feature.blob_log(skimage.transform.downscale_local_mean(PIC,(n,n)), min_sigma=0.1/(n**0.5), max_sigma=0.5/(n**0.5))
    #     # plt.imshow(skimage.transform.downscale_local_mean(PIC,(n,n)))
    #     try:
    #         p1[0].set_data(blobs.T[1], blobs.T[0])
    #     except IndexError:
    #         pass
    #     plt.pause(0.05)
