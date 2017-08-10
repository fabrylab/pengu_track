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
        self.pos_System_Markers = {}
        self.neg_System_Markers = {}
        self.GT_Markers = {}
        self.gt_db = None
        self.system_db = None

    def load_GT_marker_from_clickpoints(self, path, type):
        self.gt_db = DataFileExtended(path)
        markers = np.asarray([[m.image.sort_index, m.x, m.y] for m in self.gt_db.getMarkers(type=type)])
        self.GT_Markers.update(dict([[t, markers[markers.T[0]==t].T[1:]] for t in set(markers.T[0])]))

    def load_pos_System_marker_from_clickpoints(self, path, type):
        self.system_db = DataFileExtended(path)
        markers = np.asarray([[self.gt_db.getImage(filename=m.image.filename).sort_index, m.x, m.y] for m in self.system_db.getMarkers(type=type)
                              if self.gt_db.getImage(filename=m.image.filename) is not None])
        self.pos_System_Markers.update(dict([[t, markers[markers.T[0]==t].T[1:]] for t in set(markers.T[0])]))

    def load_neg_System_marker_from_clickpoints(self, path, type):
        self.system_db = DataFileExtended(path)
        markers = np.asarray([[self.gt_db.getImage(filename=m.image.filename).sort_index, m.x, m.y] for m in self.system_db.getMarkers(type=type)
                              if self.gt_db.getImage(filename=m.image.filename) is not None])
        self.neg_System_Markers.update(dict([[t, markers[markers.T[0]==t].T[1:]] for t in set(markers.T[0])]))

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
        print("positive")
        pos_overlap = self.spatial_overlap(self.pos_System_Markers, self.GT_Markers)
        print("negative")
        neg_overlap = self.spatial_overlap(self.neg_System_Markers, self.GT_Markers)
        for f in neg_overlap:
            # print(np.mean(o), np.std(o), np.amin(o), np.amax(o))
            pos_mask = pos_overlap[f]>self.SpaceThreshold
            neg_mask = neg_overlap[f]>self.SpaceThreshold
            # print(mask.shape)
            # print(self.System_Markers[f].T.shape, np.any(mask, axis=0).shape)

            P = float(len(self.GT_Markers[f].T))
            # P = float(len(self.pos_System_Markers[f].T))
            TP = np.sum(np.any(pos_mask, axis=1)).astype(float)
            FP = np.sum(np.all(~pos_mask, axis=0)).astype(float)
            # MN = FP
            # MS = np.sum(np.all(~mask, axis=1)).astype(float)
            N = float(len(self.neg_System_Markers[f].T))
            TN = np.sum(np.any(neg_mask, axis=1)).astype(float)
            FN = np.sum(np.all(~neg_mask, axis=0)).astype(float)
            print(P,TP,FP, N, TN, FN)

            self.specificity.update({f: TN/N})
            self.sensitivity.update({f: TP/P})
            self.precision.update({f: TP/(TP+FP)})
            self.negative_predictive_value.update({f: TN/(TN+FN)})
            self.false_positive_rate.update({f: FP/N})
            self.false_negative_rate.update({f: FN/(TP+FN)})
            self.false_discovery_rate.update({f: FP/(TP+FP)})
            self.accuracy.update({f: (TP+TN)/(TP+FN+TN+FP)})
            self.F1_score.update({f: 2*TP/(2*TP+FP+FN)})
            self.MCC.update({f: (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))})
            self.informedness.update({f: TP/P+TN/N-1})
            self.markedness.update({f: TP/(TP+FP)+TN/(TN+FN)-1})
            self.positive_rate.update({f: (TP+FP)/(len(self.pos_System_Markers[f])+len(self.neg_System_Markers[f]))})
            self.LeeLiu_rate.update({f:(TP/P)**2/((TP+FP)/(len(self.pos_System_Markers[f])+len(self.neg_System_Markers[f])))})


if __name__ == "__main__":
    import os
    # with open("/home/birdflight/Desktop/DetectionEvaluationROC.txt", "w") as myfile:
        # myfile.write("n,\t r,\t Sensitivity, \t Specificity, \t Precision \n")
    # for n in [1,2,3]:
        # for r in [1,5,10,20,30,40,50]:
    #         if os.path.exists("/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r)):
    #             pass
    #         else:
    #             continue
    #         evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
    #         evaluation.load_GT_marker_from_clickpoints(path="/home/birdflight/Desktop/252_GT_Detections_Proj.cdb", type="GT_Detections")
    #         try:
    #             evaluation.load_pos_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r), type="Pos_PT_Marker")
    #             evaluation.load_neg_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r), type="Neg_PT_Marker")
    #         except clickpoints.MarkerTypeDoesNotExist:
    #             db = clickpoints.DataFile("/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb" % (n,r))
    #             pos_type = db.setMarkerType(name="Pos_PT_Marker", color="#00FF00")
    #             neg_type = db.setMarkerType(name="Neg_PT_Marker", color="#FF0000")
    #             pos_markers = [m for m in db.getMarkers(type="PT_Detection_Marker") if not m.text.count("inf")]
    #             neg_markers = [m for m in db.getMarkers(type="PT_Detection_Marker") if m.text.count("inf")]
    #             images, x, y, text = np.asarray([[m.image, m.x, m.y, m.text] for m in pos_markers]).T
    #             db.setMarkers(image=images, x=x, y=y, text=text, type=pos_type)
    #             images, x, y, text = np.asarray([[m.image, m.x, m.y, m.text] for m in neg_markers]).T
    #             db.setMarkers(image=images, x=x, y=y, text=text, type=neg_type)
    #             evaluation.load_pos_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r), type="Pos_PT_Marker")
    #             evaluation.load_neg_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n%s_r%s.cdb"%(n,r), type="Neg_PT_Marker")
    #         evaluation.match()
    #         print(np.mean(evaluation.sensitivity.values()))
    #         print(np.mean(evaluation.specificity.values()))
    #         with open("/home/birdflight/Desktop/DetectionEvaluationROC.txt", "a") as myfile:
    #             myfile.write(",\t".join([str(n),
    #                                      str(r),
    #                                      str(np.mean(evaluation.sensitivity.values())),
    #                                      str(np.mean(evaluation.specificity.values())),
    #                                      str(np.mean(evaluation.precision.values()))]))
    #             myfile.write("\n")



    with open("/home/birdflight/Desktop/DetectionAreaEvaluationROC.txt", "w") as myfile:
        myfile.write("r,\t A,\t Sensitivity, \t Specificity, \t Precision \n")
    for r in [7,10]:
        for A in [0,10,20,30,40,50,60,70,100]:
            if os.path.exists("/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A)):
                pass
            else:
                continue
            evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
            evaluation.load_GT_marker_from_clickpoints(path="/home/birdflight/Desktop/252_GT_Detections_Proj.cdb", type="GT_Detections")
            try:
                evaluation.load_pos_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A), type="Pos_PT_Marker")
                evaluation.load_neg_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A), type="Neg_PT_Marker")
            except clickpoints.MarkerTypeDoesNotExist:
                db = clickpoints.DataFile("/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A))
                pos_type = db.setMarkerType(name="Pos_PT_Marker", color="#00FF00")
                neg_type = db.setMarkerType(name="Neg_PT_Marker", color="#FF0000")
                pos_markers = [m for m in db.getMarkers(type="PT_Detection_Marker") if not m.text.count("inf")]
                neg_markers = [m for m in db.getMarkers(type="PT_Detection_Marker") if m.text.count("inf")]
                images, x, y, text = np.asarray([[m.image, m.x, m.y, m.text] for m in pos_markers]).T
                db.setMarkers(image=images, x=x, y=y, text=text, type=pos_type)
                images, x, y, text = np.asarray([[m.image, m.x, m.y, m.text] for m in neg_markers]).T
                db.setMarkers(image=images, x=x, y=y, text=text, type=neg_type)
                evaluation.load_pos_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A), type="Pos_PT_Marker")
                evaluation.load_neg_System_marker_from_clickpoints(path="/home/birdflight/Desktop/PT_Test_n3_r%s_A%s.cdb"%(r,A), type="Neg_PT_Marker")
            # except Exception:
            #     continue
            if A == 0:
                evaluation.neg_System_Markers.update(dict([[f, np.asarray([[np.inf],[np.inf]])] for f in evaluation.pos_System_Markers]))
            evaluation.match()
            print(np.mean(evaluation.sensitivity.values()))
            print(np.mean(evaluation.specificity.values()))
            with open("/home/birdflight/Desktop/DetectionAreaEvaluationROC.txt", "a") as myfile:
                myfile.write(",\t".join([str(r),
                                         str(A),
                                         str(np.mean(evaluation.sensitivity.values())),
                                         str(np.mean(evaluation.specificity.values())),
                                         str(np.mean(evaluation.precision.values()))]))
                myfile.write("\n")

    # data = np.genfromtxt("/home/birdflight/Desktop/DetectionAreaEvaluationROC.txt", delimiter=",")[1:]
    # fig, ax = plt.subplots()
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlabel("False Positive Rate")
    # ax.set_ylabel("True Positive Rate")
    # c = sn.color_palette(n_colors=3)
    # for n in [1,2,3]:
    #     mask = data.T[0]==n
    #     print(data[mask].T[2])
    #     print(data[mask].T[3])
    #     ax.plot(1-data[mask].T[3],data[mask].T[2], '-o', color=c[n-1])
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluationROC.pdf")
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluationROC.png")
    #
    # fig, ax = plt.subplots()
    # ax.set_xlim([0, 55])
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlabel("ViBe Sensititivity Parameter")
    # ax.set_ylabel("True Positive Rate")
    # c = sn.color_palette(n_colors=3)
    # for n in [1,2,3]:
    #     mask = data.T[0]==n
    #     ax.plot(data[mask].T[1],data[mask].T[2], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
    #     # ax.fill_between(data[mask].T[1],data[mask].T[2], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n))
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR.pdf")
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_TPR.png")
    #
    # fig, ax = plt.subplots()
    # ax.set_xlim([0, 55])
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlabel("ViBe Sensititivity Parameter")
    # ax.set_ylabel("Precision")
    # c = sn.color_palette(n_colors=3)
    # for n in [1,2,3]:
    #     mask = data.T[0]==n
    #     ax.plot(data[mask].T[1],data[mask].T[4], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n)
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE.pdf")
    # plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_PRE.png")

    GT_DB = clickpoints.DataFile("/home/birdflight/Desktop/252_GT_Detections_Proj.cdb")
    from skimage.measure import label, regionprops
    mask = ~GT_DB.getMasks()[0].data.astype(bool)
    areas = [region.area for region in regionprops(label(mask, connectivity=2))]
    fig, ax = plt.subplots()
    sn.distplot(areas, bins=(np.amax(areas)-np.amin(areas))//8+1, label="Ground Truth Penguin Regions")
    ax.set_title("Histogramm of Object Areas")
    ax.set_xlabel("Area")
    ax.set_ylabel("Normed Frequency")
    legend = ax.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)
    legend.get_frame().set_facecolor('none')
    my_plot.despine(ax)
    my_plot.setAxisSizeMM(fig,ax,147,90)
    ax.set_xlim([0,400])
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaGT.pdf")
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaGT.png")

    Sys_DB = clickpoints.DataFile("/home/birdflight/Desktop/PT_Test_n3_r7.cdb")
    from skimage.measure import label, regionprops
    masks = [~mask.data.astype(bool) for mask in Sys_DB.getMasks()[27:34]]
    areas = []
    for mask in masks:
        areas.extend([region.area for region in regionprops(label(mask, connectivity=2))])
    fig, ax = plt.subplots()
    a= sn.distplot(areas, bins=(np.amax(areas)-np.amin(areas))//8+1, label="System Detected Penguin Regions")
    ax.set_title("Histogramm of Object Areas")
    ax.set_xlabel("Area")
    ax.set_ylabel("Normed Frequency")
    legend = ax.legend(loc='best', fancybox=True, framealpha=0.5, frameon=False)
    my_plot.despine(ax)
    my_plot.setAxisSizeMM(fig,ax,147,90)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaSys.pdf")
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaSys.png")

    data = np.genfromtxt("/home/birdflight/Desktop/DetectionAreaEvaluationROC.txt", delimiter=",")[1:]
    fig, ax = plt.subplots()
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    c = sn.color_palette(n_colors=3)
    for i,r in enumerate([7,10]):
        mask = data.T[0]==r
        print(data[mask].T[2])
        print(data[mask].T[3])
        ax.plot(1-data[mask].T[3],data[mask].T[2], '-o', color=c[i-1])
    plt.legend(loc='best')
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaROC.pdf")
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaROC.png")

    fig, ax = plt.subplots()
    ax.set_xlim([0, 105])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Lower Area Threshold")
    ax.set_ylabel("True Positive Rate")
    c = sn.color_palette(n_colors=3)
    for i,r in enumerate([7,10]):
        mask = data.T[0]==r
        ax.plot(data[mask].T[1],data[mask].T[2], '-o', color=c[i-1], label=r"$r=%s$"%r)
        # ax.fill_between(data[mask].T[1],data[mask].T[2], '-o', color=c[n-1], label=r"$N_{min}=%s$"%n))
    plt.legend(loc='best')
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaTPR.pdf")
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaTPR.png")

    fig, ax = plt.subplots()
    ax.set_xlim([0, 105])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Lower Area Threshold")
    ax.set_ylabel("Precision")
    c = sn.color_palette(n_colors=3)
    for i,r in enumerate([7,10]):
        mask = data.T[0]==r
        ax.plot(data[mask].T[1],data[mask].T[4], '-o', color=c[i-1], label=r"$r=%s$"%r)
    plt.legend(loc='best')
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaPRE.pdf")
    plt.savefig("/home/birdflight/Desktop/DetectionEvaluation_AreaPRE.png")

    # from PenguTrack.Detectors import SiAdViBeSegmentation
    # starter_db = clickpoints.DataFile()
    # SiAdViBeSegmentation(horizonmarkers, 14e-3, [17e-3,9e-3], penguin_markers, penguing_height, 500)
    plt.show()