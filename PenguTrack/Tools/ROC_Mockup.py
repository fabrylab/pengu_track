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

fig, ax = plt.subplots()
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
c = sn.color_palette(n_colors=3)
# data = {1: np.array([[0,0,1],[0,1,1], np.arange(3)[::1]]),
#         2: np.array([np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), np.arange(11)[::1]]),
#         3: np.array([np.arange(0.,1.1,0.1), 1-(0.1/np.arange(0.1,1.2,0.1))**2, np.arange(11)[::1]])}
#         # 3: np.array([np.arange(0.1,1,0.1), np.log(np.arange(0.1,1,0.1))])}
X = np.arange(10, dtype=float)
a=3.
b=2.
data = {1: np.array([X, X<2, X<1]),
# 1: np.array([X, (1/(X+1)).astype(int), 1-(1/(X+1)).astype(int)]),
        2: np.array([X, (9-X)/9, (9-X)/9]) ,
        # 3: np.array([X, ((1./(X+1))**(1./a)-(1./10)**(1./a))/(1-(1./10)**(1./a)), ((1./(X+1))**b-(1./10)**b)/(1-(1./10)**b)])}
        3: np.array([X, ((1./(X+1))**(1./a)), ((1./(X+1))**b-(1./10)**b)/(1-(1./10)**b)])}
        # 3: np.array([np.arange(0.1,1,0.1), np.log(np.arange(0.1,1,0.1))])}
label ={1: "Ideal Classifier",
        2: "Random Classifier",
        3: "Typical Classifier"}

for n in [1,2,3]:
    ax.plot(data[n][2], data[n][1], '-o', color=c[n - 1], label=label[n])
ax.fill_between(data[3][2], data[3][1], color = c[2], alpha=0.2)
ax.legend(loc="best", prop={'size':12})
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, 147, 90)
plt.savefig("/home/birdflight/Desktop/Mock_ROC.pdf")
plt.savefig("/home/birdflight/Desktop/Mock_ROC.png")


fig, ax = plt.subplots()
ax.set_xlim([-0.05, 10.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel("Classifier Parameter")
ax.set_ylabel("True Positive Rate")
for n in [1,2,3]:
    ax.plot(data[n][0], data[n][1], '-o', color=c[n - 1], label=label[n].split(" ")[0])
# ax.fill_between(data[][0], data[3][1], color = c[2], alpha=0.2)
ax.legend(loc="best",ncol=1 , prop={'size':12})
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, 147, 90)
plt.tight_layout()
plt.savefig("/home/birdflight/Desktop/Mock_TPR.pdf")
plt.savefig("/home/birdflight/Desktop/Mock_TPR.png")


fig, ax = plt.subplots()
ax.set_xlim([-0.05, 10.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel("Classifier Parameter")
ax.set_ylabel("Precision")
for n in [1,2,3]:
    prec = data[n][1] / (data[n][1] + data[n][2])
    print(prec)
    prec[np.isnan(prec)]=1.
    ax.plot(data[n][0], prec, '-o', color=c[n - 1], label=label[n].split(" ")[0])
# ax.fill_between(data[][0], data[3][1], color = c[2], alpha=0.2)
ax.legend(loc="best", ncol=1 ,prop={'size': 12})
my_plot.despine(ax)
my_plot.setAxisSizeMM(fig, ax, 147, 90)
plt.tight_layout()
plt.savefig("/home/birdflight/Desktop/Mock_PRE.pdf")
plt.savefig("/home/birdflight/Desktop/Mock_PRE.png")

plt.show()