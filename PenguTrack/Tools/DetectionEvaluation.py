from __future__ import print_function, division
import numpy as np
import clickpoints

from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import VariableSpeed
from PenguTrack.DataFileExtended import DataFileExtended

import matplotlib.pyplot as plt
import seaborn as sn
import my_plot

class Evaluator(object):
    def __init__(self):
        self.System_Markers = {}
        self.GT_Markers = {}
        self.gt_db = None
        self.system_db = None

    def load_GT_marker_from_clickpoints(self, path, type):
        self.gt_db = DataFileExtended(path)
        markers = np.asarray([[m.image.sort_index, m.x, m.y] for m in self.gt_db.getMarkers(type=type)])
        self.GT_Markers.update(dict([[t, markers[markers.T[0]==t].T[1:]] for t in set(markers.T[0])]))

    def load_System_marker_from_clickpoints(self, path, type):
        self.system_db = DataFileExtended(path)
        image_filenames = set([img.filename for img in self.gt_db.getImages()]).intersection([img.filename for img in self.system_db.getImages()])
        markers = np.asarray([[self.gt_db.getImage(filename=m.image.filename).sort_index, m.x, m.y] for m in self.system_db.getMarkers(type=type) if m.image.filename in image_filenames])
        self.System_Markers.update(dict([[t, markers[markers.T[0]==t].T[1:]] for t in set(markers.T[0])]))

    def save_marker_to_GT_db(self, markers, path, type):
        if function is None:
            function = lambda x : x
        self.gt_db = DataFileExtended(path)

        with self.gt_db.db.atomic() as transaction:
            self.gt_db.deleteMarkers(type=type)
            for t in markers:
                self.gt_db.setMarkers(image=self.gt_db.getImage(frame=t),x=markers[t].T[0],y=markers[t].T[1], type=type)

    def save_marker_to_System_db(self, markers, path, type):
        if function is None:
            function = lambda x : x
        self.system_db = DataFileExtended(path)

        with self.system_db.db.atomic() as transaction:
            self.system_db.deleteMarkers(type=type)
            for t in markers:
                self.system_db.setMarkers(image=self.system_db.getImage(frame=t),x=markers[t].T[0],y=markers[t].T[1], type=type)


class Yin_Evaluator(Evaluator):
    def __init__(self, object_size, *args, **kwargs):
        self.SpaceThreshold = kwargs.pop("spacial_threshold", 0.1)
        self.Object_Size = object_size
        self.Matches = {}
        self.True_Positive_Rate = {}
        self.Precision = {}
        self.Positive_Rate = {}
        self.Negative_Rate = {}
        # self.True_Positive_Rate = {}
        # self.True_Negative = {}
        # self.False_Negative = {}
        self.LeeLiu_Rate = {}
        self.Missed_Rate = {}

        super(Yin_Evaluator, self).__init__(*args, **kwargs)

    def spatial_overlap(self, markersA, markersB):
        frames = set(markersA.keys()).intersection(markersB.keys())
        out = {}
        for f in frames:
            out.update({f: np.asarray([[self._intersect_(p1,p2)/self._union_(p1,p2) for p1 in markersA[f].T] for p2 in markersB[f].T])})
        return out

    def _intersect_(self, p1, p2):
        o = self.Object_Size/2.
        l = max(p1[0]-o, p2[0]-o)
        r = min(p1[0]+o, p2[0]+o)
        t = min(p1[1]+o, p2[1]+o)
        b = max(p1[1]-o, p2[1]-o)
        return (t-b)*(r-l) if ((r-l)>0 and (t-b)>0) else 0

    def _union_(self, p1, p2):
        return 2*self.Object_Size**2-self._intersect_(p1,p2)

    def match(self):
        overlap = self.spatial_overlap(self.System_Markers, self.GT_Markers)
        for f in overlap:
            o=overlap[f]
            # print(np.mean(o), np.std(o), np.amin(o), np.amax(o))
            mask = overlap[f]>self.SpaceThreshold
            # print(mask.shape)
            # print(self.System_Markers[f].T.shape, np.any(mask, axis=0).shape)

            P = float(len(self.GT_Markers[f].T))
            M = float(len(self.System_Markers[f].T))
            MP = np.sum(np.any(mask, axis=0)).astype(float)
            TP = np.sum(np.any(mask, axis=1)).astype(float)
            FP = np.sum(np.all(~mask, axis=0)).astype(float)
            MN = FP
            MS = np.sum(np.all(~mask, axis=1)).astype(float)
            print(P,M,MP,TP,FP)
            self.True_Positive_Rate.update({f: TP/P})
            self.Precision.update({f: TP/(TP+FP)})
            self.LeeLiu_Rate.update({f:(TP/P)**2/((TP+FP)/M)})
            self.Negative_Rate.update({f: MN/M})
            self.Positive_Rate.update({f: MP/M})
            self.Missed_Rate.update({f: 1.-MP/P})

            # self.True_Positive.update({f:np.asarray(self.System_Markers[f].T[np.any(mask, axis=0)])})
            # self.False_Positive.update({f:np.asarray(self.System_Markers[f].T[np.all(~mask, axis=0)])})
            # self.True_Negative.update({f:np.asarray(self.GT_Markers[f].T[np.any(mask, axis=1)])})
            # self.False_Negative.update({f:np.asarray(self.GT_Markers[f].T[np.all(~mask, axis=1)])})

    # def TruePositiveRate(self):
    #     frames = set(self.True_Positive.keys()).intersection(self.System_Markers.keys())
    #     return np.mean([float(len(self.True_Positive[f]))/len(self.System_Markers[f].T) for f in frames])
    #
    # def TrueNegativeRate(self):
    #     frames = set(self.True_Negative.keys()).intersection(self.GT_Markers.keys())
    #     return np.mean([float(len(self.True_Negative[f])) / len(self.GT_Markers[f].T) for f in frames])
    #
    # def FalsePositiveRate(self):
    #     frames = set(self.True_Negative.keys()).intersection(self.GT_Markers.keys())
    #     return np.mean([float(len(self.True_Negative[f])) / len(self.GT_Markers[f].T) for f in frames])




    # def FalsePositiveRate(self):
    #     frames = set(self.False_Positive.keys()).intersection(self.GT_Markers.keys())
    #     return np.mean([float(len(self.True_Positive[f]))/len(self.GT_Markers[f].T) for f in frames])



if __name__ == "__main__":
    version = "birdflight"
    if version == "Adelie":
        import os
        with open("/home/birdflight/Desktop/DetectionEvaluation.txt", "w") as myfile:
            myfile.write("n,\tr,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision \n")
        for n in [1,2,3]:
            for r in [1,5,10,20,30,40,50]:
                if os.path.exists("/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r)):
                    pass
                else:
                    continue
                evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
                evaluation.load_GT_marker_from_clickpoints(path="/home/birdflight/Desktop/252_GT_Detections.cdb", type="GT_Detection")
                evaluation.load_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r), type="PT_Track_Marker")
                evaluation.match()
                try:
                    print(evaluation.Negative_Rate[50])
                    print(evaluation.Missed_Rate[50])
                    with open("/home/birdflight/Desktop/DetectionEvaluation.txt", "a") as myfile:
                        myfile.write(",\t".join([str(n),
                                                 str(r),
                                                 str(evaluation.Negative_Rate[50]),
                                                 str(evaluation.Missed_Rate[50]),
                                                 str(evaluation.True_Positive_Rate[50]),
                                                 str(evaluation.Precision[50])]))
                        myfile.write("\n")
                    print("-----------------")
                except KeyError:
                    pass

        data = np.genfromtxt("/home/birdflight/Desktop/DetectionEvaluation.txt", delimiter=",")[1:]
        fig, ax = plt.subplots()
        ax.set_xlim([(1-np.amax(data.T[3]))**1.1,1])
        ax.set_ylim([np.amin(data.T[3])**1.1,1])
        ax.loglog()
        ax.set_xlabel("Correct Detection Rate")
        ax.set_ylabel("Missed Detection Rate")
        c = sn.color_palette(n_colors=3)
        for n in [1,2,3]:
            mask = data.T[0]==n
            print(data[mask].T[2])
            print(data[mask].T[3])
            ax.plot(1-data[mask].T[3],data[mask].T[3], '-o', color=c[n-1])
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation.pdf")
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation.png")


        fig, ax = plt.subplots()
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=3)
        for n in [1,2,3]:
            mask = data.T[0]==n
            ax.plot(data[mask].T[1],data[mask].T[4], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
        ax.legend()
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig,ax,147,90)
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR.pdf")
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=3)
        for n in [1,2,3]:
            mask = data.T[0]==n
            ax.plot(data[mask].T[1],data[mask].T[5], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
        ax.legend(loc="best")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig,ax,147,90)
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE.pdf")
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE.png")
        plt.show()
    elif version == "birdflight":
        import os
        with open("/home/birdflight/Desktop/DetectionEvaluation_bf.txt", "w") as myfile:
            myfile.write("n,\tr,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision \n")
        for r in [20]:
            for f in [1,2,3,4,5]:
                if os.path.exists("/mnt/mmap/Starter_n3_3_r%s_F%s.cdb" % (r, f)):
                    pass
                else:
                    continue
                evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
                evaluation.load_GT_marker_from_clickpoints(path="/mnt/mmap/GT_Starter.cdb",
                                                           type="GT_Bird")
                evaluation.load_System_marker_from_clickpoints(
                    path="/mnt/mmap/Starter_n3_3_r%s_F%s.cdb" % (r, f), type="PT_Track_Marker")
                evaluation.match()
                print("bla")
                try:
                    print("blabla")
                    print(np.mean(evaluation.Negative_Rate.values()))
                    print(np.mean(evaluation.Missed_Rate.values()))
                    with open("/home/birdflight/Desktop/DetectionEvaluation_bf.txt", "a") as myfile:
                        myfile.write(",\t".join([str(r),
                                                 str(f),
                                                 str(np.mean(evaluation.Negative_Rate.values())),
                                                 str(np.mean(evaluation.Missed_Rate.values())),
                                                 str(np.mean(evaluation.True_Positive_Rate.values())),
                                                 str(np.mean(evaluation.Precision.values()))]))
                        myfile.write("\n")
                    print("-----------------")
                except KeyError:
                    pass

        data = np.genfromtxt("/home/birdflight/Desktop/DetectionEvaluation_bf.txt", delimiter=",")[1:]
        # fig, ax = plt.subplots()
        # ax.set_xlim([(1 - np.amax(data.T[3])) ** 1.1, 1])
        # ax.set_ylim([np.amin(data.T[3]) ** 1.1, 1])
        # ax.loglog()
        # ax.set_xlabel("Correct Detection Rate")
        # ax.set_ylabel("Missed Detection Rate")
        # c = sn.color_palette(n_colors=3)
        # for n in [1, 2, 3]:
        #     mask = data.T[0] == n
        #     print(data[mask].T[2])
        #     print(data[mask].T[3])
        #     ax.plot(1 - data[mask].T[3], data[mask].T[3], '-o', color=c[n - 1])
        # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_bf.pdf")
        # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_bf.png")

        fig, ax = plt.subplots()
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=3)
        for n in [1, 2, 3]:
            mask = data.T[0] == n
            ax.plot(data[mask].T[1], data[mask].T[4], '-o', color=c[n - 1], label=r"$N_{min}=%s$" % n)
        ax.legend()
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR_bf.pdf")
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR_bf.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=3)
        for n in [1, 2, 3]:
            mask = data.T[0] == n
            ax.plot(data[mask].T[1], data[mask].T[5], '-o', color=c[n - 1], label=r"$N_{min}=%s$" % n)
        ax.legend(loc="best")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE_bf.pdf")
        plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE_bf.png")
        plt.show()