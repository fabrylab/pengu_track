from __future__ import print_function, division
import clickpoints
import numpy as np
from Detectors import Measurement
from Filters import Filter
from Models import VariableSpeed
from DataFileExtended import DataFileExtended
import matplotlib.pyplot as plt
import time
import seaborn as sns
import scipy.stats as ss
from mpl_toolkits.mplot3d import Axes3D
import glob
import matplotlib.gridspec as gridspec

def load_tracks(path, type):
    db = DataFileExtended(path)
    Tracks = {}
    if db.getTracks(type=type)[0].markers[0].measurement is not None:
        for track in db.getTracks(type=type):
            Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
            for m in track.markers:
                meas = Measurement(1., [m.measurement[0].x,
                                        m.measurement[0].y,
                                        m.measurement[0].z])
                Tracks[track.id].update(z=meas, i=m.image.sort_index)
    else:
        for track in db.getTracks(type=type):
            Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
            for m in track.markers:
                meas = Measurement(1., [m.x,
                                        m.y,
                                        0])
                Tracks[track.id].update(z=meas, i=m.image.sort_index)
    return Tracks

def getZpos(Tracks,dt,v_fac,frame):
    mean_z = []
    dist_list = []
    z_list = []
    XY_positions = {}
    XY_pos_fra = {}
    for track in Tracks:
        track_data = Tracks[track]
        t_start = min(track_data.X.keys())
        t_end = max(track_data.X.keys())
        meas = Tracks[track].Measurements
        z_positions = []
        xy_positions = []
        xy_pos_fra = []
        for i in range(t_start,t_end+1):
            try:
                z_positions.append(meas[i].PositionZ)
            except KeyError:
                z_positions.append(np.nan)
            try:
                xy_positions.append([meas[i].PositionX,meas[i].PositionY])
            except KeyError:
                xy_positions.append([np.nan,np.nan])
        for i in range(frame+1):
            try:
                xy_pos_fra.append([meas[i].PositionX,meas[i].PositionY])
            except KeyError:
                xy_pos_fra.append([np.nan,np.nan])
        for i in range(1, len(xy_positions)):
            if not np.isnan(xy_positions[i]).any() and not np.isnan(xy_positions[i-1]).any():
                dist = np.sqrt((xy_positions[i][0] - xy_positions[i - 1][0]) ** 2. + (
                xy_positions[i][1] - xy_positions[i - 1][1]) ** 2.)
            else:
                dist = np.nan
            dist_list.append([dist*v_fac,(z_positions[i]+z_positions[i-1])/2.])
        alter_dist = 0.0001
        count = 0
        for i in range(1, len(xy_positions)):
            if not np.isnan(xy_positions[i]).any() and not np.isnan(xy_positions[i-1]).any():
                alter_dist += np.sqrt((xy_positions[i][0] - xy_positions[i - 1][0]) ** 2. + (
                xy_positions[i][1] - xy_positions[i - 1][1]) ** 2.)
                count += 1
        if count == 0:
            count += 1
        alter_dist = (alter_dist / float(count))*v_fac
        mean_z.append([track,np.nanmean(z_positions),len(z_positions),alter_dist])
        XY_positions.update({track:xy_positions})
        XY_pos_fra.update({track:xy_pos_fra})
    return mean_z,dist_list,XY_positions,XY_pos_fra

data_list = glob.glob("/home/user/TCell/2017-08-29/layers_2017_08_29_24h_24hiG_Kon_RB_5_pos1_12Gel_stitched2.cdb")
data_list = [data for data in data_list if data.count('stitched2')]
path = "/home/user/TCell/2017-05-30/layers_2017_05_30_24h_1hiG_CU_BIS_stitched2.cdb"
type = "PT_Track_Marker"
# dt = 15
for i in data_list:
    if i.count("RB_3_") or i.count("RB_4_") or i.count("RB_5_") or i.count("RB_6_"):
        dt = 30
    else:
        dt = 15
    v_fac = 0.645/(dt/60)
    n = 1.33
    frames = 120
    Tracks = load_tracks(i,type)
    Z_posis, velocity, XY_pos, XY_pos_fra = getZpos(Tracks,dt,v_fac,frames)
    Z_posis = np.asarray(Z_posis)
    # Z_posis[:,1]*=n
    Z_pos_ab = []
    Z_pos_be = []
    for i in Z_posis:
        if i[1]>=200 and i[3]>10:
            Z_pos_ab.append(i[1])
        if i[1]<=200 and i[3]>10:
            Z_pos_be.append(i[1])
    if len(Z_pos_ab)<=1:
        Z_pos_ab = np.asarray([0,0])
    else:
        Z_pos_ab = np.asarray(Z_pos_ab)
    if len(Z_pos_be)<=1:
        Z_pos_be = np.asarray([0,0])
    else:
        Z_pos_be = np.asarray(Z_pos_be)
    perc_ab = np.percentile(Z_pos_ab,10) - 20
    perc_be = np.percentile(Z_pos_be,10) + 20
    velocity = np.asarray(velocity)
    tracks_to_delete = []
    for i in Z_posis:
        if i[1]>=perc_ab or i[1]<=perc_be:
            tracks_to_delete.append(int(i[0]))
    all = 0
    for i in Z_posis:
        all+=i[2]
    a = 800
    z_pos_plot = []
    for i in range(a):
        count = 0
        for j in Z_posis:
            if j[1]<=n*i:
                count += j[2]
        z_pos_plot.append(count/float(all))

    velocity_test=np.asarray([i for i in velocity if not np.isnan(i).any()])

    # plt.figure()
    # plt.plot(range(a),z_pos_plot)
    # plt.tick_params(axis = "both", which = "major", labelsize = 20)
    # plt.xlim(0,a)
    # plt.ylim(0,1)
    # plt.xlabel("Z-Position [$\mu m$]", fontsize = 25)
    # plt.ylabel("Track Anteil [%]", fontsize = 25)
    # plt.tight_layout()
    # plt.show()

    plt.figure()
    plt.plot(Z_posis[:,3],Z_posis[:,1]*n,'.')
    plt.tick_params(axis = "both", which = "major", labelsize = 20)
    plt.xlim(0,25)
    plt.ylim(0,800)
    plt.ylabel("Mittlere Z-Position [$\mu m$]", fontsize = 20)
    plt.xlabel(r'Mittlere Geschwindigkeit [$\frac{\mathrm{\mu m}}{\mathrm{min}}$]', fontsize = 20)
    plt.tight_layout()
    plt.show()

    # fig = plt.figure()
    # border_width = 0.18
    # ax_size = [0 + border_width, 0 + border_width, 1 - 2 * border_width, 1 - 2 * border_width]
    # ax = fig.add_axes(ax_size)
    # cm = plt.cm.get_cmap('jet')
    # xy = np.vstack([velocity_test[:,0], velocity_test[:,1]])
    # kd = ss.gaussian_kde(xy)(xy)
    # idx = kd.argsort()
    # x, y, z = velocity_test[idx,0], velocity_test[idx,1], kd[idx]
    # ax.scatter(x, y, c=z, s=35, edgecolor='', alpha=1.0, cmap=cm)
    # plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
    # plt.xlabel(r'Speed in $\frac{\mathrm{\mu m}}{\mathrm{min}}$', fontsize=20)
    # plt.ylabel('Z-position', fontsize=20)
    # plt.xlim(0,40)
    # plt.ylim(0,700)
    # # plt.tight_layout()
    # plt.show()

    Z_posis_above = []
    Z_posis_below = []
    Z_posis_between = []
    for i in range(len(Z_posis)):
        if Z_posis[i,1]>=(600/n) and Z_posis[i,3]>=5:
            Z_posis_above.append(Z_posis[i])
        if Z_posis[i,1]<=(170/n) and Z_posis[i,3]>=5:
            Z_posis_below.append(Z_posis[i])
        if Z_posis[i,1]>=(170/n) and Z_posis[i,1]<=(600/n):
            Z_posis_between.append(Z_posis[i])
    Z_posis_above = np.asarray(Z_posis_above)
    Z_posis_below = np.asarray(Z_posis_below)
    Z_posis_between = np.asarray(Z_posis_between)

    a = 0
    b = -1
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.15, hspace=0.15)
    plt.subplot(gs1[1])
    for i in Z_posis_above[:,0]:
        xy = np.asarray(XY_pos_fra[i][a:b])
        plt.plot(xy[:,0],xy[:,1],color = 'b')
    for i in Z_posis_below[:,0]:
        xy = np.asarray(XY_pos_fra[i][a:b])
        plt.plot(xy[:,0],xy[:,1],color = 'r')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('X-Position [$\mu m$]', fontsize=20)
    plt.subplot(gs1[0])
    for i in Z_posis_between[:,0]:
        xy = np.asarray(XY_pos_fra[i][a:b])
        plt.plot(xy[:,0],xy[:,1],color = 'g')
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')  # labels along the bottom edge are off
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('X-Position [$\mu m$]', fontsize=20)
    plt.ylabel('Y-Position [$\mu m$]', fontsize=20)
    # plt.title('Cell movement above and below the collagen Gel', fontsize=20)
    plt.show()