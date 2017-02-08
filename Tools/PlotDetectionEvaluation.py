from __future__ import print_function, division
import numpy as np
import pandas as pd

with open('eval.txt', "r") as myfile:
    text_data = myfile.read()

immobile_text = text_data.split('----')[0]
mobile_text = text_data.split('----')[-1]

immobile_nums = [[float(s) for s in line.split() if not (s.isalpha() or s.count('-')>0 or s.count(',') or s.count(':') or s.count('%'))] for line in immobile_text.split('P-Faktor')][1:]
mobile_nums = [[float(s) for s in line.split() if not (s.isalpha() or s.count('-')>0 or s.count(',') or s.count(':') or s.count('%'))] for line in mobile_text.split('P-Faktor')][1:]

columns = ['P-Faktor', 'N-Total GT', 'N-Total Auto', 'absolute Correct Detections', 'relative Correct Detections',
           'absolute False Positives', 'relative False Positives',
           'absolute False Negative', 'relative False Negative',
           'absolute Total RMS-Error', 'relative Total RMS-Error']
df_mobile = pd.DataFrame(mobile_nums, columns=columns)
df_mobile = df_mobile.sort_values('relative False Positives')
df_immobile = pd.DataFrame(immobile_nums, columns=columns)
df_immobile = df_immobile.sort_values('relative False Positives')

import matplotlib.pyplot as plt
# import seaborn as sn
fig, ax = plt.subplots(1,1)
# ax1, ax2 = ax
ax1 = ax

ax1.set_title("ROC with mobile Penguins")
ax1.set_ylim([0,100])
ax1.set_xlim([0,100])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('Detection Rate', color='k')
ax1.tick_params('y', colors='k')
ax1.plot(df_mobile['relative False Positives'], df_mobile['relative Correct Detections'], color="g", marker='o')
ax1.plot(df_mobile['relative False Positives'], df_mobile['relative Total RMS-Error'], color="r", marker='o')

ax1_1 = ax1.twinx()
ax1_1.set_ylabel('Value of P', color='b')
ax1_1.tick_params('y', colors='b')
ax1_1.plot(df_mobile['relative False Positives'], df_mobile['P-Faktor'], color="b", marker='o')

ax1.legend()
ax1_1.legend(loc='lower left')


# ax2.set_title("ROC all Penguins")
# ax2.set_ylim([0,100])
# ax2.set_xlim([0,100])
# ax2.set_xlabel('False Positive Rate')
# ax2.set_ylabel('Detection Rate', color='k')
# ax2.tick_params('y', colors='k')
# ax2.plot(df_immobile['relative False Positives'], df_immobile['relative Correct Detections'], color="g", marker='o')
# ax2.plot(df_immobile['relative False Positives'], df_immobile['relative Total RMS-Error'], color="r", marker='o')
#
# ax2_1 = ax2.twinx()
# ax2_1.set_ylabel('Value of P', color='b')
# ax2_1.tick_params('y', colors='b')
# ax2_1.plot(df_immobile['relative False Positives'], df_immobile['P-Faktor'], color="b", marker='o')
#
# ax2.legend()
# ax2_1.legend(loc='lower left')

fig.tight_layout()
plt.show()