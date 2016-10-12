import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time import time

class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

    def detect(self, image):
        return np.random.rand(2)

class ViBeDetector(Detector):
    def __init__(self, n=20, r=50, n_min=2, phi=32):
        super(ViBeDetector, self).__init__()
        self.N = int(n)
        self.R = np.uint16(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Samples = None
        self.SegMap = None
        # self.Dist = None
        self._Neighbour_Map ={0: [-1, -1],
                              1: [-1, 0],
                              2: [-1, 1],
                              3: [0, -1],
                              4: [0, 1],
                              5: [1, -1],
                              6: [1, 0],
                              7: [1, 1]}
        # self._Neighbour_Map = {0: [None, -2, None, -2],
        #                        1: [None, -2, 1, -1],
        #                        2: [None, -2, 2, None],
        #                        3: [1, -1, None, -2],
        #                        4: [1, -1, 2, None],
        #                        5: [2, None, None, -2],
        #                        6: [2, None, 1, -1],
        #                        7: [2, None, 2, None]}

    def detect(self, image):
        super(ViBeDetector, self).detect(image)
        width, height = image.getShape()
        data = np.array(image.data, dtype=np.uint16)
        if self.Samples is None:
            self.Samples = np.tile(data, self.N).reshape((self.N,)+data.shape)
            self.SegMap = np.zeros((width, height), dtype=bool)

        self.SegMap = (np.sum((np.sum((self.Samples-data)**2, axis=-1, dtype=np.uint16) < self.R**2), axis=0, dtype=np.uint16) > self.N_min)
        current_sample_mask = (np.meshgrid(np.arange(width), np.arange(height), np.arange(self.N))[2].T ==
                               np.random.randint(0, self.N, size=width*height).reshape((width, height)))
        image_mask = (np.random.randint(0, self.Phi, size=width*height).reshape((width, height)) == 0) & self.SegMap
        current_sample_mask = current_sample_mask & image_mask
        self.Samples[current_sample_mask] = data[image_mask]

        n = np.sum(image_mask)
        if n > 0:
            x, y = np.array(np.meshgrid(np.arange(height), np.arange(width))).T[image_mask.T].T
            randX, randY = np.array([self._Neighbour_Map[k] for k in np.random.randint(0, 8, size=n)]).T
            x = np.array([min(max(randX[i]+x_, 0), width-1) for i, x_ in enumerate(x)])
            y = np.array([min(max(randY[i]+y_, 0), height-1) for i, y_ in enumerate(y)])
            neighbours = np.array(image_mask)
            neighbours[:] = Falset
            for i in range(n):
                neighbours[x[i], y[i]] = True
            j = np.random.randint(0, self.N)
            try:
                self.Samples[j][neighbours] = data[image_mask]
            except ValueError, e:
                print(e)
                print(neighbours)
                print(np.sum(neighbours), self.Samples[j].shape, neighbours.shape, data.shape, image_mask.shape)
                #DIRTY TRICK
                self.Samples[j][neighbours] = data[image_mask][:np.sum(neighbours)]
                # print("could not update neighbours")

    def _get_Rand_Ng(self, shape=(1) ):#
        size = np.prod(shape)
        n = np.random.randint(1, 4, size)
        n *= (1-np.random.randint(0, 2, size)*2)
        n += 4
        n.reshape(shape)
        return (n % 3)-1, np.floor(n/3)-1

if __name__ == '__main__':
    import cProfile, pstats, StringIO

    pr = cProfile.Profile()
    pr.enable()

    import clickpoints
    db = clickpoints.DataFile("click1.cdb")
    VB = ViBeDetector()
    images = db.getImageIterator()
    # plt.imshow(Image.fromarray(next(images).data))
    # plt.ion()
    s = 0
    for image in images:
        VB.detect(image)
        print(VB.SegMap.shape, image.data.shape)
        db.setMask(image=image, data=255-VB.SegMap.astype(np.uint8))
        print("Mask save")
        # plt.imshow(Image.fromarray(np.array(VB.SegMap, dtype=float)*255))
        # plt.savefig('./adelie_data/Tracked/Image%04d.png'%s)
        s += 1
        # plt.pause(0.05)

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

    # def stack(array):
    #     stacked = np.zeros_like(array, dtype=int)
    #     stacked[1:-1, 1:-1] += array[2:, 2:]
    #     stacked[1:-1, 1:-1] += array[2:, 1:-1]
    #     stacked[1:-1, 1:-1] += array[2:, :-2]
    #     stacked[1:-1, 1:-1] += array[1:-1, 2:]
    #     stacked[1:-1, 1:-1] += array[1:-1, 1:-1]
    #     stacked[1:-1, 1:-1] += array[1:-1, :-2]
    #     stacked[1:-1, 1:-1] += array[:-2, 2:]
    #     stacked[1:-1, 1:-1] += array[:-2, 1:-1]
    #     stacked[1:-1, 1:-1] += array[:-2, :-2]
    #     return (stacked == 9) | ((stacked ==0) & (array>0))
    #     # return np.array(stacked/9., dtype=int)
    #
    # def stack(array):
    #     array = array.as_array(dtype=bool)
    #     return (array[2:, 2:] & array[2:, 1:-1] & array[2:, :-2] & array[1:-1, 2:] & array[1:-1, 1:-1] &
    #                            array[1:-1, :-2] & array[:-2, 2:] & array[:-2, 1:-1] & array[:-2, :-2]
    #                            ) | (((array[2:, 2:] | array[2:, 1:-1] | array[2:, :-2] | array[1:-1, 2:] |
    #                                 array[1:-1, :-2] | array[:-2, 2:] | array[:-2, 1:-1] | array[:-2, :-2]) ^ True
    #                                  ) & array[1:-1, 1:-1])
    #
    #
    # def blobs(array):
    #     stacked = np.array(array, dtype=int)
    #     while True:
    #         new_stack = stack(stacked)
    #         if np.all(new_stack == 0):
    #             break
    #         else:
    #             stacked = new_stack
    #     return stacked
