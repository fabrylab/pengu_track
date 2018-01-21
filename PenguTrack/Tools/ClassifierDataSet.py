import numpy as np
import pickle

class dotdict(dict):
    """
    enables .access on dicts
    """
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ClassfierDataSet(list):
    def __init__(self, img_size, *args, **kwargs):
        self.Image_Size=int(img_size)
        super(ClassfierDataSet, self).__init__(*args, **kwargs)

    def load_data(self, image, x, y, timestamp, filename, tag="none", label=0):
        image = np.array(image, ndmin=2, dtype=np.int8)
        x = float(x)
        y = float(y)
        label = int(label)
        self.append(dotdict({"image": self.__sample__(image, x, y),
                             "label": label,
                             "meta": self.__meta__(image, tag, x, y, timestamp, filename)}))

    def __sample__(self, img, x, y):
        h = img.shape[0]
        w = img.shape[1]
        # size = np.ceil(self.Image_Size/(2**0.5)).astype(int)
        size = self.Image_Size
        x_min = max(0, int(x-size/2.))
        x_max = min(w, int(x+size/2.))
        y_min = max(0, int(y-size/2.))
        y_max = min(h, int(y+size/2.))
        sl = np.zeros([self.Image_Size, self.Image_Size])
        sl[:y_max-y_min, :x_max-x_min] = img[y_min:y_max, x_min:x_max]
        return sl

    def __meta__(self, img, tag, x, y, timestamp, filename):
        h = img.shape[0]
        w = img.shape[1]
        return dotdict({"timestamp": timestamp,
                        "filename": filename,
                        "x": x,
                        "y": y,
                        "w": self.Image_Size,
                        "h": self.Image_Size,
                        "transformations": [],
                        "tag": tag})

    def save(self, file):
        pickle.dump([s for s in self], open(file, "wb"), protocol=2)

    def load(self, file):
        self.extend(pickle.load(open(file, "rb")))

    def __meta_cp__(self, img, tag, x, y):
        h = img.data.shape[0]
        w = img.data.shape[1]
        return dotdict({"timestamp":img.timestamp,
                "filename": img.filename,
                "x":x,
                "y":y,
                "w":self.Image_Size,
                "h":self.Image_Size,
                "transformations": [],
                "tag":tag})

    def load_from_clickpoints(self, file, marker_type, label=0, tag=None, n=None):
        import clickpoints
        db = clickpoints.DataFile(file)
        if tag is None:
            tag = file.split("/")[-1].split("\\")[-1]
        marker_type = db.getMarkerType(marker_type)
        if n is None:
            self.extend([dotdict({"image":self.__sample__(m.image.data,m.x,m.y),
                          "label":label,
                          "meta":self.__meta__(m.image,tag,m.x,m.y)}) for m in db.getMarkers(type=marker_type)])
        else:
            i = int(db.getMarkers(type=marker_type).count()/n)
            try:
                self.extend([dotdict({"image":self.__sample__(m.image.data,m.x,m.y),
                              "label":label,
                              "meta":self.__meta__(m.image,tag,m.x,m.y)}) for m in db.getMarkers(type=marker_type)[:n*i:i]])
            except ValueError:
                self.extend([dotdict({"image":self.__sample__(m.image,m.x,m.y),
                              "label":label,
                              "meta":self.__meta__(m.image.data,tag,m.x,m.y)}) for m in db.getMarkers(type=marker_type)])
                raise Warning("Not enough data found in Database. n=%s , found %s "%(n,db.getMarkers(type=marker_type).count()))
# class ClassfierDataSet(list):
#     def __init__(self, img_size, *args, **kwargs):
#         self.Image_Size=int(img_size)
#         super(ClassfierDataSet, self).__init__(*args, **kwargs)
#
#     def save(self, file):
#         pickle.dump([s for s in self], open(file, "wb"), protocol=2)
#
#     def load(self, file):
#         self.extend(pickle.load(open(file, "rb")))

from scipy.ndimage.interpolation import rotate, shift
from scipy.ndimage import zoom
class AugmentedDataSet(ClassfierDataSet):
    def __init__(self, *args, **kwargs):
        super(AugmentedDataSet, self).__init__(*args, **kwargs)

    def rotate(self, rotations):
        """

        :param rotations: array_like, list of angles in degrees.
        :return: DataSet updated with rotated pictures.
        """
        for r in rotations:
            self.extend([dotdict({"image":rotate(s.image, r, reshape=False),
                          "label":s.label,
                          "meta":self.__update_meta__(s.meta, "rotated by %s degree"%r)}) for s in self])
        return self

    def crop(self, crop_x, crop_y):
        """

        :param crop_x:
        :param crop_y:
        :return:
        """

        for cx in crop_x:
            cx1 = np.floor(cx/2)
            cx2 = np.ceil(cx/2)
            for cy in crop_y:
                cy1 = np.floor(cy/2)
                cy2 = np.ceil(cy/2)
                self.extend([dotdict({"image":s.image[cy1:-cy2,cx1:-cx2],
                              "label":s.label,
                              "meta":self.__update_meta__(s.meta, "croped by %s px in x and %s px in y"%(cx, cy))}) for s in self])
        return self

    def translate(self, trans_x, trans_y):
        """

        :param trans_x:
        :param trans_y:
        :return:
        """
        for tx in trans_x:
            for ty in trans_y:
                self.extend([dotdict({"image":shift(s.image, [tx, ty]),
                              "label":s.label,
                              "meta":self.__update_meta__(s.meta, "translated by %s px in x and %s px in y"%(tx, ty))}) for s in self])
        return self

    def scale(self, scale_x, scale_y):
        """

        :param scale_x:
        :param scale_y:
        :return:
        """
        for sx in scale_x:
            for sy in scale_y:
                self.extend([dotdict({"image":zoom(s.image,[sx,sy]),
                              "label":s.label,
                              "meta":self.__update_meta__(s.meta, "scaled by factor %s in x and %s in y"%(sx, sy))}) for s in self])
        return self

    def __update_meta__(self, meta, trans):
        meta["transformations"].append(trans)
        return meta





if __name__ == "__main__":
    # import clickpoints
    # ClSet = AugmentedDataSet(25)
    # db = clickpoints.DataFile("/mnt/mmap/GT_Starter.cdb")
    # for m in db.getMarkers(type="GT_Bird"):
    #     print(m.id)
    #     ClSet.load_data(m.image.data, m.x, m.y, m.image.timestamp, m.image.filename, tag="GT_Starter", label=1)
    #
    # db = clickpoints.DataFile("/mnt/mmap/Starter_Full.cdb")
    # n=1000
    # N = db.getMarkers(type="Negatives").count()
    # i = max(int(N/n),1)
    # for m in db.getMarkers(type="Negatives")[:n*i:i]:
    #     print(m.id)
    #     ClSet.load_data(m.image.data, m.x, m.y, m.image.timestamp, m.image.filename, tag="Starter_Full", label=0)

    # ClSet.load_from_clickpoints("/mnt/mmap/GT_Starter.cdb", "GT_Bird", label=1, n=100)
    # ClSet.load_from_clickpoints("/mnt/mmap/Starter_Full.cdb", "Negatives", label=0, n=100)

    # ClSet.save("/home/birdflight/Desktop/ClassifierData.cds")
    ClSet2 = ClassfierDataSet(25)
    ClSet2.load("C:/Users/Alex/Desktop/ClassifierData.cds")
