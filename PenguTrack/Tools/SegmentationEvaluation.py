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

class SegmentationEvaluator(object):
    def __init__(self):
        self.System_Masks = []
        self.GT_Masks = []
        self.gt_db = None
        self.system_db = None
        self.specificity = {}
        self.true_negative_rate = self.specificity
        self.sensitivity = {}
        self.true_positive_rate = self.sensitivity
        self.precision = {}
        self.positive_predictive_value = self.precision
        self.negative_predictive_value = {}
        self.false_positive_rate = {}
        self.fall_out = self.false_positive_rate
        self.false_negative_rate = {}
        self.false_discovery_rate = {}
        self.accuracy = {}
        self.F1_score = {}
        self.MCC = {}
        self.informedness = {}
        self.markedness = {}
        self.positive_rate = {}
        self.LeeLiu_rate = {}

    def load_GT_masks_from_clickpoints(self, path):
        self.gt_db = DataFileExtended(path)
        # self.GT_Masks.extend([str(m.image.timestamp) for m in self.gt_db.getMasks()])
        timestamps = self.gt_db.db.execute_sql("select image.timestamp from image inner join mask on mask.image_id=image.id where mask.data is not null").fetchall()
        self.GT_Masks.extend([str(t[0]) for t in timestamps])
        # self.GT_Masks.extend(set([str(m.image.timestamp) for m in self.gt_db.getMasks()]).update(set(self.GT_Masks)))

    def load_System_masks_from_clickpoints(self, path):
        self.system_db = DataFileExtended(path)
        # self.System_Masks.extend([str(m.image.timestamp) for m in self.system_db.getMasks()])
        timestamps = self.system_db.db.execute_sql("select image.timestamp from image inner join mask on mask.image_id=image.id where mask.data is not null").fetchall()
        self.System_Masks.extend([str(t[0]) for t in timestamps])
        # for im in self.system_db.getImages():
        #     if im.mask is not None:
        #         print(im.timestamp)
        #         self.System_Masks.append(str(im.timestamp))
        # self.System_Masks.extend(set([str(m.image.timestamp) for m in self.system_db.getMasks()]).update(set(self.System_Masks_Masks)))

    def match(self, gt_inverse=True, system_inverse=True):
        stamps = set(self.GT_Masks).intersection(self.System_Masks)
        for stamp in stamps:
            if system_inverse:
                sm = ~self.system_db.getMask(image=self.system_db.getImages(timestamp=stamp)[0]).data.astype(bool)
            else:
                sm = self.system_db.getMask(image=self.system_db.getImages(timestamp=stamp)[0]).data.astype(bool)
            if gt_inverse:
                gt = ~self.gt_db.getMask(image=self.gt_db.getImages(timestamp=stamp)[0]).data.astype(bool)
            else:
                gt = self.gt_db.getMask(image=self.gt_db.getImages(timestamp=stamp)[0]).data.astype(bool)
            P = np.sum(gt).astype(float)
            TP = np.sum(sm & gt).astype(float)
            FP = np.sum(sm & (~gt)).astype(float)
            N = np.sum(~gt).astype(float)
            TN = np.sum((~sm) & (~gt)).astype(float)
            FN = np.sum((~sm) & gt).astype(float)
            self.specificity.update({stamp: TN/N})
            self.sensitivity.update({stamp: TP/P})
            self.precision.update({stamp: TP/(TP+FP)})
            self.negative_predictive_value.update({stamp: TN/(TN+FN)})
            self.false_positive_rate.update({stamp: FP/N})
            self.false_negative_rate.update({stamp: FN/(TP+FN)})
            self.false_discovery_rate.update({stamp: FP/(TP+FP)})
            self.accuracy.update({stamp: (TP+TN)/(TP+FN+TN+FP)})
            self.F1_score.update({stamp: 2*TP/(2*TP+FP+FN)})
            self.MCC.update({stamp: (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))})
            self.informedness.update({stamp: TP/P+TN/N-1})
            self.markedness.update({stamp: TP/(TP+FP)+TN/(TN+FN)-1})
            self.positive_rate.update({stamp: (TP+FP)/(sm.shape[0]*sm.shape[1])})
            self.LeeLiu_rate.update({stamp:(TP/P)**2/((TP+FP)/(sm.shape[0]*sm.shape[1]))})



if __name__ == "__main__":
    version="birdflight"
    if version == "Adelie":
        import os
        with open("/home/birdflight/Desktop/SegmentationEvaluation.txt", "w") as myfile:
            myfile.write("n,\t r,\t Sensitivity, \t Specificity, \t Precision \n")
        for n in [1,2,3]:
            for r in [1,5,7,10,12,15,20,30,40,50]:
                if os.path.exists("/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r)):
                    pass
                else:
                    continue
                seg = SegmentationEvaluator()
                seg.load_GT_masks_from_clickpoints("/home/birdflight/Desktop/252_GT_Detections_Proj.cdb")
                seg.load_System_masks_from_clickpoints("/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r))
                seg.match()
                print(np.mean(seg.sensitivity.values()))
                print(np.mean(seg.specificity.values()))
                with open("/home/birdflight/Desktop/SegmentationEvaluation.txt", "a") as myfile:
                    myfile.write(",\t".join([str(n),
                                             str(r),
                                             str(np.mean(seg.sensitivity.values())),
                                             str(np.mean(seg.specificity.values())),
                                             str(np.mean(seg.precision.values()))]))
                    myfile.write("\n")

        data = np.genfromtxt("/home/birdflight/Desktop/SegmentationEvaluation.txt", delimiter=",")[1:]
        fig, ax = plt.subplots()
        ax.set_xlim([1e-7, 5e-3])
        ax.set_xscale('log')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=6)
        for n in [3,2,1]:
            mask = data.T[0]==n
            # n = 5 if n==43 else n
            ax.fill_between(1.-data[mask].T[3],data[mask].T[2], color=c[n-1], alpha=0.2)
        for n in [1,2,3]:
            mask = data.T[0]==n
            # n = 5 if n==43 else n
            ax.plot(1-data[mask].T[3],data[mask].T[2], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
            # for xyr in zip(1-data[mask].T[3],data[mask].T[2],data[mask].T[1]):
            #     ax.annotate('%s' %xyr[-1], xy=xyr[:-1], textcoords='data')
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        plt.legend(loc='best')
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig,ax,147,90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation.png")

        fig, ax = plt.subplots()
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=6)
        for n in [1,2,3]:
            mask = data.T[0]==n
            # n = 5 if n==43 else n
            ax.plot(data[mask].T[1],data[mask].T[2], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
        ax.legend()
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig,ax,147,90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_TPR.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_TPR.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=6)
        for n in [1,2,3]:
            mask = data.T[0]==n
            # n = 5 if n==43 else n
            ax.plot(data[mask].T[1],data[mask].T[4], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
        ax.legend(loc="center right")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig,ax,147,90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_PRE.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_PRE.png")
        plt.show()
    elif version == "birdflight":
        import os

        with open("/home/birdflight/Desktop/SegmentationEvaluation_bf.txt", "w") as myfile:
            myfile.write("n,\t r,\t Sensitivity, \t Specificity, \t Precision \n")
        for n in [3]:
            for r in [5,10, 20, 30, 40, 50]:
                if os.path.exists("/mnt/mmap/Starter_n%s_%s_r%s.cdb" % (n, n, r)):
                    pass
                else:
                    continue
                seg = SegmentationEvaluator()
                seg.load_GT_masks_from_clickpoints("/mnt/mmap/GT_Starter.cdb")
                # seg.load_System_masks_from_clickpoints("/mnt/mmap/GT_Starter.cdb")
                seg.load_System_masks_from_clickpoints("/mnt/mmap/Starter_n%s_%s_r%s.cdb" % (n, n, r))
                seg.match(gt_inverse=False)
                print(np.mean(seg.sensitivity.values()))
                print(np.mean(seg.specificity.values()))
                with open("/home/birdflight/Desktop/SegmentationEvaluation_bf.txt", "a") as myfile:
                    myfile.write(",\t".join([str(n),
                                             str(r),
                                             str(np.mean(seg.sensitivity.values())),
                                             str(np.mean(seg.specificity.values())),
                                             str(np.mean(seg.precision.values()))]))
                    myfile.write("\n")

        data = np.genfromtxt("/home/birdflight/Desktop/SegmentationEvaluation_bf.txt", delimiter=",")[1:]
        fig, ax = plt.subplots()
        ax.set_xlim([1e-7, 5e-3])
        ax.set_xscale('log')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=6)
        for n in [3]:
            mask = data.T[0] == n
            # n = 5 if n==43 else n
            ax.fill_between(1. - data[mask].T[3], data[mask].T[2], color=c[n - 1], alpha=0.2)
        for n in [3]:
            mask = data.T[0] == n
            # n = 5 if n==43 else n
            ax.plot(1 - data[mask].T[3], data[mask].T[2], '-o', color=c[n - 1], label=r"$N_{min}=%s$" % n)
            # for xyr in zip(1-data[mask].T[3],data[mask].T[2],data[mask].T[1]):
            #     ax.annotate('%s' %xyr[-1], xy=xyr[:-1], textcoords='data')
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        plt.legend(loc='best')
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_bf.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_bf.png")

        fig, ax = plt.subplots()
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=6)
        for n in [3]:
            mask = data.T[0] == n
            # n = 5 if n==43 else n
            ax.plot(data[mask].T[1], data[mask].T[2], '-o', color=c[n - 1], label=r"$N_{min}=%s$" % n)
        ax.legend()
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_TPR_bf.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_TPR_bf.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([0, 55])
        ax.set_ylim([-0.000005, 0.0003])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=6)
        for n in [3]:
            mask = data.T[0] == n
            # n = 5 if n==43 else n
            ax.plot(data[mask].T[1], data[mask].T[4], '-o', color=c[n - 1], label=r"$N_{min}=%s$" % n)
        ax.legend(loc="center right")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_PRE_bf.pdf")
        plt.savefig("/home/birdflight/Desktop/SegmentationEvaluation_PRE_bf.png")
        plt.show()