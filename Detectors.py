import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Detector(object):
    def __init__(self):
        super(Detector, self).__init__()

    def detect(self, image):
        return np.random.rand(2)

class ViBeDetector(Detector):
    def __init__(self, n=20, r=20, n_min=2, phi=16):
        super(ViBeDetector, self).__init__()
        self.N = int(n)
        self.R = float(r)
        self.N_min = int(n_min)
        self.Phi = int(phi)
        self.Samples = None
        self.SegMap = None
        self.Dist = None

    def detect(self, image):
        super(ViBeDetector, self).detect(image)
        width, height = image.getShape()
        data = image.data
        if self.Samples is None:
            self.Samples = np.zeros((self.N,)+data.shape)
            self.SegMap = np.zeros((width, height))
            self.Dist = np.zeros((width, height, self.N))
        for i in range(self.N):
            self.Dist[:, :, i] = (np.linalg.norm(data-self.Samples[i], axis=-1) < self.R)
        num_mask = (np.sum(self.Dist, axis=-1) > self.N_min)
        print(np.sum(num_mask))
        self.SegMap = num_mask

        current_sample_mask = (np.meshgrid(np.arange(width), np.arange(height), np.arange(self.N))[2].T ==
                               np.random.randint(0, self.N, size=width*height).reshape((width, height)))
        image_mask = (np.random.randint(0, self.Phi, size=width*height).reshape((width, height)) == 0) & num_mask
        current_sample_mask = current_sample_mask & image_mask
        self.Samples[current_sample_mask] = data[image_mask]

        current_sample_mask = (np.meshgrid(np.arange(width), np.arange(height), np.arange(self.N))[2].T ==
                               np.random.randint(0, self.N, size=width*height).reshape((width, height)))
        image_mask = (np.random.randint(0, self.Phi, size=width*height).reshape((width, height)) == 0) & num_mask
        current_sample_mask = current_sample_mask & image_mask
        current_sample_mask = np.pad(current_sample_mask, 1, mode='wrap')[:-2, :-2]
        self.Samples[current_sample_mask] = data[image_mask]




        # for x in range(width):
        #     for y in range(height):
        #         count = 0
        #         index = 0
        #         dist = 0
        #         while count < self.N_min and index < self.N:
        #             dist = np.linalg.norm(data[x, y] - self.samples[x, y].T[index])
        #             if dist < self.R:
        #                 count += 1
        #             index += 1
        #         if count >= self.N_min:
        #             self.segMap[x, y] = background
        #             rand = np.random.randint(0,  self.Phi-1)
        #             if rand == 0:
        #                 rand = np.random.randint(0, self.N-1)
        #                 self.samples[x, y].T[rand] = data[x, y]
        #             rand = np.random.randint(0, self.Phi-1)
        #             if rand == 0:
        #                 x_NG, y_NG = self._get_Rand_Ng()
        #                 rand = np.random.randint(0, self.N-1)
        #                 self.samples[x_NG, y_NG].T[rand] = data[x, y]
        #         else:
        #             self.segMap[x, y] = foreground
    def _get_Rand_Ng(self, shape=(1) ):
        size = np.prod(shape)
        n = np.random.randint(1, 4, size)
        n *= (1-np.random.randint(0, 2, size)*2)
        n += 4
        n.reshape(shape)
        return (n % 3)-1, np.floor(n/3)-1

if __name__ == '__main__':
    import clickpoints
    db = clickpoints.DataFile("click1.cdb")
    VB = ViBeDetector()
    images = db.getImageIterator()
    plt.imshow(Image.fromarray(next(images).data))
    plt.ion()
    for image in images:
        VB.detect(image)
        plt.imshow(Image.fromarray(np.array(VB.SegMap, dtype=float)*255))
        plt.pause(0.05)

