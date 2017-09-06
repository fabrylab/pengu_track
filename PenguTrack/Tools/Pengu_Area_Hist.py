from __future__ import print_function,division
import clickpoints
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style("white")
import my_plot
# my_plot.set_style("white")

from skimage.measure import label, regionprops

db_gt = clickpoints.DataFile("/home/alex/Masterarbeit/Data/Adelies/DataBases/252_GT_Detections_Proj.cdb")
db = clickpoints.DataFile("/home/alex/Masterarbeit/Data/Adelies/DataBases/PT_Test_n3_r7_A20.cdb")
area_gt = np.array([prop.area for prop in regionprops(label(~(db_gt.getMasks()[0].data).astype(bool)))])
print(area_gt)
calc = False

if calc:
    area_sys = np.hstack([[prop.area for prop in regionprops(label(~(m.data).astype(bool)))] for m in db.getMasks()])
    with open("/home/alex/Masterarbeit/Data/Adelies/Evaluation/Areas_n3_r7_A20.txt","w") as myfile:
        myfile.write("\t".join(["Size"])+"\n")
    with open("/home/alex/Masterarbeit/Data/Adelies/Evaluation/Areas_n3_r7_A20.txt","a") as myfile:
        # myfile.write("\n".join(["\t".join([aa for aa in a]) for a in area_sys]))
        myfile.write("\n".join(["\t".join([str(a)]) for a in area_sys]))

with open("/home/alex/Masterarbeit/Data/Adelies/Evaluation/Areas_n3_r7_A20.txt","r") as myfile:
    data = np.genfromtxt(myfile,delimiter="\t",skip_header=1)

hist, bins = np.histogram(data, bins=40, range=[1,400], normed=True)
hist2, bins2 = np.histogram(area_gt, bins=40, range=[1,400], normed=True)

fig, ax = plt.subplots()
ax.set_ylabel("Normed Frequency")
ax.set_xlabel("Area")
ax.set_xlim([0,400])
ax.set_ylim([0,sorted(hist2)[-1]])
sn.distplot(area_gt, bins=bins, norm_hist=True, label="Ground Truth \nPenguin Regions", hist_kws={"edgecolor":"k"}, ax=ax)
ax.legend(loc="upper right", prop={"size":10}, bbox_to_anchor=(1.1,1.1))
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, 147/2,90/2)
fig.tight_layout()
plt.savefig("/home/alex/Desktop/DetectionEvaluation_AreaGT.png")
plt.savefig("/home/alex/Desktop/DetectionEvaluation_AreaGT.pdf")

fig, ax = plt.subplots()
ax.set_ylabel("Normed Frequency")
ax.set_xlabel("Area")
ax.set_xlim([0,400])
ax.set_ylim([0,sorted(hist2)[-1]])
sn.distplot(data, bins=bins,  norm_hist=True, label="System Detected \nPenguin Regions", hist_kws={"edgecolor":"k"}, ax=ax)
ax.legend(loc="best", prop={"size":10})
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, 147/2,90/2)
fig.tight_layout()
plt.savefig("/home/alex/Desktop/DetectionEvaluation_AreaSys.png")
plt.savefig("/home/alex/Desktop/DetectionEvaluation_AreaSys.pdf")

