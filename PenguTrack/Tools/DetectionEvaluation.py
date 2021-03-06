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
sn.set_style("white")

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
        # if function is None:
        #     function = lambda x : x
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
    version = "cell_clean"
    calc=False
    if version == "Adelie":
        if calc:
            import os
            with open("/home/alex/Masterarbeit/Data/Adelies/Evaluation/DetectionEvaluation_ad.txt", "w") as myfile:
                myfile.write("r,\ta_min,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision \n")
            for r in [7,10]:
                for A_min in [0,5,10,20,30,40,50,60,70,80,100]:
                    if os.path.exists("/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r%s_A%s.cdb"%(r,A_min)):
                        pass
                    else:
                        continue
                    evaluation = Yin_Evaluator(15, spacial_threshold=0.3)
                    # evaluation.load_GT_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/252_GT_Detections.cdb", type="GT_Detection")
                    evaluation.load_GT_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/252_GT_Detections_Proj.cdb", type="GT_Detections")
                    # evaluation.load_GT_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/252_GT_Detections.cdb", type="GT_Detection_Active")
                    # mask = evaluation.GT_Markers[50][1]>589
                    # evaluation.GT_Markers.update({50:evaluation.GT_Markers[50].T[mask].T})
                    # evaluation.load_System_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r%s_A%s.cdb"%(r,A_min), type="PT_Track_Marker")
                    try:
                        evaluation.load_System_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r%s_A%s.cdb"%(r,A_min), type="Pos_PT_Marker")
                    except clickpoints.MarkerTypeDoesNotExist:
                        db = clickpoints.DataFile("/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r%s_A%s.cdb"%(r,A_min))
                        pos_type = db.setMarkerType(name="Pos_PT_Marker", color="#00FF00")
                        x, y, img, text = np.array([[m.x, m.y, m.image, m.text] for m in db.getMarkers(type="PT_Detection_Marker") if not m.text.count("inf")]).T
                        db.setMarkers(image=img, x=x,y=y,text=text, type=pos_type)
                        neg_type = db.setMarkerType(name="Neg_PT_Marker", color="#FF0000")
                        x, y, img, text = np.array([[m.x, m.y, m.image, m.text] for m in db.getMarkers(type="PT_Detection_Marker") if m.text.count("inf")]).T
                        db.setMarkers(image=img, x=x,y=y,text=text, type=neg_type)
                        evaluation.load_System_marker_from_clickpoints(path="/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r%s_A%s.cdb"%(r,A_min), type="Pos_PT_Marker")
                    evaluation.match()
                    # try:
                    #     print(evaluation.Negative_Rate[50])
                    #     print(evaluation.Missed_Rate[50])
                    #
                    with open("/home/alex/Masterarbeit/Data/Adelies/Evaluation/DetectionEvaluation_ad.txt", "a") as myfile:
                        # myfile.write(",\t".join([str(r),
                        #                          str(A_min),
                        #                          str(evaluation.Negative_Rate[50]),
                        #                          str(evaluation.Missed_Rate[50]),
                        #                          str(evaluation.True_Positive_Rate[50]),
                        #                          str(evaluation.Precision[50])]))
                        myfile.write(",\t".join([str(r),
                                                 str(A_min),
                                                 str(np.nanmean(evaluation.Negative_Rate.values())),
                                                 str(np.nanmean(evaluation.Missed_Rate.values())),
                                                 str(np.nanmean(evaluation.True_Positive_Rate.values())),
                                                 str(np.nanmean(evaluation.Precision.values()))]))
                        # myfile.write(",\t".join([str(r),
                        #                          str(A_min),
                        #                          # str(np.nanmean(evaluation.Negative_Rate.values())),
                        #                          # str(np.nanmean(evaluation.Missed_Rate.values())),
                        #                          str(1-np.nanmean(evaluation.Precision.values())),
                        #                          str(1-np.nanmean(evaluation.True_Positive_Rate.values()))]))
                        myfile.write("\n")
                    print("-----------------")
                    # except KeyError:
                    #     pass

        data = np.genfromtxt("/home/alex/Masterarbeit/Data/Adelies/Evaluation/DetectionEvaluation_ad.txt", delimiter=",")[1:]
        # fig, ax = plt.subplots()
        # ax.set_xlim([(1-np.amax(data.T[3]))**1.1,1])
        # ax.set_ylim([np.amin(data.T[3])**1.1,1])
        # ax.loglog()
        # ax.set_xlabel("Correct Detection Rate")
        # ax.set_ylabel("Missed Detection Rate")
        # c = sn.color_palette(n_colors=3)
        # for j,r in enumerate([7,10]):
        #     mask = data.T[0]==r
        #     print(data[mask].T[2])
        #     print(data[mask].T[3])
        #     ax.plot(1-data[mask].T[3],data[mask].T[3], '-o', color=c[j])
        # plt.savefig("/home/alex/Desktop/DetectionEvaluation.pdf")
        # plt.savefig("/home/alex/Desktop/DetectionEvaluation.png")


        fig, ax = plt.subplots()
        ax.set_xlim([-5, 105])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("Lower Area Threshold")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=3)[::-2]
        for j,r in enumerate([7,10]):
            mask = data.T[0]==r
            ax.plot(data[mask].T[1],data[mask].T[4], '-o', color=c[j], label=r"$r=%s$"%r)
        ax.legend(loc="best", prop={"size":10})
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147/2, 90/2)
        plt.tight_layout()
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_ad.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_ad.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([-5, 105])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("Lower Area Threshold")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=3)[::-2]
        for j,r in enumerate([7,10]):
            mask = data.T[0]==r
            ax.plot(data[mask].T[1],data[mask].T[5], '-o', color=c[j], label=r"$r%s$"%r)
        ax.legend(loc="best", prop={"size":10})
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147/2, 90/2)
        plt.tight_layout()
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_ad.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_ad.png")
        plt.show()
    elif version == "birdflight":
        if calc:
            import os
            def std_err(l):
                return np.std(l)/len(l)**0.5
            with open("/home/birdflight/Desktop/DetectionEvaluation_bf.txt", "w") as myfile:
                myfile.write("n,\tr,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision, \t TruePositiveError, \t PrecisionError \n")
            for r in [20]:
                for f in [1,2,3,4,5]:
                    if os.path.exists("/mnt/mmap/Starter_n3_3_r%s_F%s.cdb" % (r, f)):
                        pass
                    else:
                        continue
                    evaluation = Yin_Evaluator(10, spacial_threshold=0.1)
                    evaluation.load_GT_marker_from_clickpoints(path="/mnt/mmap/GT_Starter.cdb",
                                                               type="GT_Bird")
                    evaluation.load_System_marker_from_clickpoints(
                        path="/mnt/mmap/Starter_n3_3_r%s_F%s.cdb" % (r, f), type="PT_Track_Marker")
                    evaluation.match()
                    # print("bla")
                    try:
                        # print("blabla")
                        print(np.mean(evaluation.Negative_Rate.values()))
                        print(np.mean(evaluation.Missed_Rate.values()))
                        with open("/home/birdflight/Desktop/DetectionEvaluation_bf.txt", "a") as myfile:
                            myfile.write(",\t".join([str(r),
                                                     str(f),
                                                     str(np.mean(evaluation.Negative_Rate.values())),
                                                     str(np.mean(evaluation.Missed_Rate.values())),
                                                     str(np.mean(evaluation.True_Positive_Rate.values())),
                                                     str(np.mean(evaluation.Precision.values())),
                                                     str(std_err(evaluation.True_Positive_Rate.values())),
                                                     str(std_err(evaluation.Precision.values()))]))
                            myfile.write("\n")
                        print("-----------------")
                    except KeyError:
                        pass

        data = np.genfromtxt("/home/alex/Masterarbeit/Data/Birds/Evaluation/DetectionEvaluation_bf.txt", delimiter=",")[1:]
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
        ax.set_xlim([0, 5])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=3)
        for j,n in enumerate([20]):
            mask = data.T[0] == n
            ax.plot(data[mask].T[1], data[mask].T[4], '-o', color=c[j - 1], label=r"$N_{min}=3$ \n $N=3$")
        ax.legend(loc="best")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_bf.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_bf.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([0, 5])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=3)
        for j,n in enumerate([20]):
            mask = data.T[0] == n
            ax.plot(data[mask].T[1], data[mask].T[5], '-o', color=c[j - 1], label=r"$N_{min}=3$ \n $N=3$")
        ax.legend(loc="best")
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147, 90)
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_bf.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_bf.png")


        # fig, ax = plt.subplots()
        N = len(data)
        means = tuple(data.T[5])#(1.975373213, 2.25414394, 2.363441898, 2.620267547)
        std = [tuple(np.zeros(N)), tuple(data.T[7])]#[(0, 0, 0, 0), (0.095275084, 0.091739906, 0.126287507, 0.026789034)]

        ind = np.arange(0.5, 0.5*(N+1), 0.5)#0.5, 1, 1.5, 2, 2.5)  # the x locations for the groups

        fig = plt.figure()
        # fig.set_size_inches(147*0.039370, 90*0.039370)
        ax = plt.subplot(111)

        rects1 = ax.bar(ind, means,
                        width=0.4,
                        color= c[-1],#'#BAB6B4',
                        yerr=std,
                        edgecolor='black',
                        linewidth=1.5,
                        error_kw=dict(ecolor='black',
                                      lw=1.5,
                                      capsize=4,
                                      capthick=1.5))

        ax.set_ylabel('Precision', fontsize=10)
        ax.set_xlabel('Number of Consecutive Detection-Filters', fontsize=10)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=1.5)

        ax.yaxis.major.locator.set_params(nbins=4)
        ax.yaxis.set_tick_params(width=1.5)

        plt.xticks(ind, ("       pure Area", "      + Solidity", "  + Eccentricity", "     + Visibility", "+ Mean Intensity"), ha="center", rotation=45)
        plt.yticks()
        plt.tick_params(axis='both', which='major', labelsize=10)



        def MoveLabels(value):
            import types
            import matplotlib
            for label in plt.gca().xaxis.get_majorticklabels():
                label.customShiftValue = value
                label.set_x = types.MethodType(
                    lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue), label)


        MoveLabels(+0.3)
        plt.ylim([0, 0.1])
        plt.xlim([0.25, N/2.+0.5])
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147*0.5, 90/2)
        fig1 = plt.gcf()
        plt.tight_layout()
        plt.show()
        plt.draw()
        fig1.savefig('/home/alex/Desktop/DetectionEvaluation_PRE_BAR_bf.png', dpi=300)
        fig1.savefig('/home/alex/Desktop/DetectionEvaluation_PRE_BAR_bf.pdf')
        plt.show()
    elif version == "cell":
        if calc:
            import os

            with open("/home/alex/Desktop/DetectionEvaluation_ce.txt", "w") as myfile:
                myfile.write("A_min,\tA_max,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision \n")
            for A_min in [0,25,50,75,100,150,200]:
                for A_max in [300,400,np.inf]:
                    if os.path.exists("/home/alex/Desktop/PT_Cell_T850_A%s_%s.cdb"%(A_min,A_max)):
                        pass
                    else:
                        continue
                    evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
                    evaluation.load_GT_marker_from_clickpoints(path="/home/alex/Desktop/PT_Cell_Test_GT.cdb",
                                                               type="GroundTruth")
                    evaluation.load_System_marker_from_clickpoints(
                        path="/home/alex/Desktop/PT_Cell_T850_A%s_%s.cdb"%(A_min,A_max), type="PT_Track_Marker")
                    evaluation.match()
                    # print("bla")
                    try:
                        # print("blabla")
                        print(np.mean(evaluation.Negative_Rate.values()))
                        print(np.mean(evaluation.Missed_Rate.values()))
                        with open("/home/alex/Desktop/DetectionEvaluation_ce.txt", "a") as myfile:
                            myfile.write(",\t".join([str(A_min),
                                                     str(A_max),
                                                     str(np.mean(evaluation.Negative_Rate.values())),
                                                     str(np.mean(evaluation.Missed_Rate.values())),
                                                     str(np.mean(evaluation.True_Positive_Rate.values())),
                                                     str(np.mean(evaluation.Precision.values()))]))
                            myfile.write("\n")
                        print("-----------------")
                    except KeyError:
                        pass

        data = np.genfromtxt("/home/alex/Masterarbeit/Data/Cells/Evaluation/DetectionEvaluation_ce.txt", delimiter=",")[1:]
        fig, ax = plt.subplots()
        ax.set_xlim([(1 - np.amax(data.T[3])) ** 1.1, 1])
        ax.set_ylim([np.amin(data.T[3]) ** 1.1, 1])
        ax.loglog()
        ax.set_xlabel("Correct Detection Rate")
        ax.set_ylabel("Missed Detection Rate")
        c = sn.color_palette(n_colors=3)
        for j,A_max in enumerate([500, np.inf]):
            mask = data.T[1] == A_max
            print(data[mask].T[2])
            print(data[mask].T[3])
            ax.plot(1 - data[mask].T[3], data[mask].T[3], '-o', color=c[j])
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_ce.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_ce.png")


        fig, ax = plt.subplots()
        ax.set_xlim([-1, 250])
        ax.set_ylim([0.4+-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("True Positive Rate")
        c = sn.color_palette(n_colors=4)
        for j, A_max in enumerate([300,400,np.inf][::-1]):
            mask = data.T[1] == A_max
            ax.plot(data[mask].T[0], data[mask].T[4], '-o', color=c[j], label=r"$A_{max}=%s$" % A_max)
        ax.legend(loc="best", prop={"size":10})
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147/2, 90/2)
        plt.tight_layout()
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_ce.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_TPR_ce.png")

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlim([-1, 250])
        ax.set_ylim([0.4+-0.05, 1.05])
        ax.set_xlabel("ViBe Sensititivity Parameter")
        ax.set_ylabel("Precision")
        c = sn.color_palette(n_colors=4)
        for j, A_max in enumerate([300,400,np.inf][::-1]):
            mask = data.T[1] == A_max
            ax.plot(data[mask].T[0], data[mask].T[5], '-o', color=c[j], label=r"$A_{max}=%s$" % A_max)
        ax.legend(loc="best", prop={"size":10})
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147/2, 90/2)
        plt.tight_layout()
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_ce.pdf")
        plt.savefig("/home/alex/Desktop/DetectionEvaluation_PRE_ce.png")
        plt.show()
    elif version == "cell_clean":
        if calc:
            import os

            with open("/home/alex/Desktop/DetectionEvaluation_ce.txt", "w") as myfile:
                myfile.write("A_min,\tA_max,\t NegativeRate, \t MissedRate, \t TruePositive, \t Precision \n")
            for A_min in [0,25,50,75,100,150,200]:
                for A_max in [300,400,np.inf]:
                    if os.path.exists("/home/alex/Desktop/PT_Cell_T850_A%s_%s.cdb"%(A_min,A_max)):
                        pass
                    else:
                        continue
                    evaluation = Yin_Evaluator(10, spacial_threshold=0.05)
                    evaluation.load_GT_marker_from_clickpoints(path="/home/alex/Desktop/PT_Cell_Test_GT.cdb",
                                                               type="GroundTruth")
                    evaluation.load_System_marker_from_clickpoints(
                        path="/home/alex/Desktop/PT_Cell_T850_A%s_%s.cdb"%(A_min,A_max), type="PT_Track_Marker")
                    evaluation.match()
                    # print("bla")
                    try:
                        # print("blabla")
                        print(np.mean(evaluation.Negative_Rate.values()))
                        print(np.mean(evaluation.Missed_Rate.values()))
                        with open("/home/alex/Desktop/DetectionEvaluation_ce.txt", "a") as myfile:
                            myfile.write(",\t".join([str(A_min),
                                                     str(A_max),
                                                     str(np.mean(evaluation.Negative_Rate.values())),
                                                     str(np.mean(evaluation.Missed_Rate.values())),
                                                     str(np.mean(evaluation.True_Positive_Rate.values())),
                                                     str(np.mean(evaluation.Precision.values()))]))
                            myfile.write("\n")
                        print("-----------------")
                    except KeyError:
                        pass

        data = np.genfromtxt("/home/alex/Masterarbeit/Data/Cells/Evaluation/DetectionEvaluation_ce.txt", delimiter=",")[1:]


        fig, ax = plt.subplots()
        ax.set_xlim([-1, 250])
        ax.set_ylim([0.55+-0.05, 1.05])
        ax.set_xlabel("Lower Area Threshold")
        ax.set_ylabel("Rate")
        c = sn.color_palette(n_colors=2)
        for j, A_max in enumerate([np.inf][::-1]):
            mask = data.T[1] == A_max
            ax.plot(data[mask].T[0], data[mask].T[4], '-o', color=c[j], label=r"Sensitivity")
        for j, A_max in enumerate([np.inf][::-1]):
            mask = data.T[1] == A_max
            ax.plot(data[mask].T[0], data[mask].T[5], '-o', color=c[j+1], label=r"Precision")
        ax.legend(loc="best", prop={"size":10})
        my_plot.despine(ax)
        my_plot.setAxisSizeMM(fig, ax, 147/2, 90/2)
        plt.tight_layout()
        plt.savefig("/home/alex/Masterarbeit/Pictures/DetectionEvaluation_clean_ce.pdf")
        plt.savefig("/home/alex/Masterarbeit/Pictures/DetectionEvaluation_clean_ce.png")
        plt.show()
