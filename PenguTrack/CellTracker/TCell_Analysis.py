from __future__ import division,print_function
import clickpoints
import glob
import numpy as np
from PenguTrack.Detectors import Measurement
from PenguTrack.Filters import Filter
from PenguTrack.Models import VariableSpeed
# from DataFileExtended import DataFileExtended as DataFile
from PenguTrack.Stitchers import Heublein_Stitcher

import os.path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys
from scipy.optimize import leastsq,curve_fit

from sklearn.mixture import GaussianMixture

import seaborn as sns

import cv2
import peewee
from qtpy import QtGui, QtCore, QtWidgets
from qimage2ndarray import array2qimage
import qtawesome as qta

from PenguTrack.Filters import KalmanFilter
from PenguTrack.Filters import MultiFilter
from PenguTrack.Models import RandomWalk
from PenguTrack.Detectors import SimpleAreaDetector2 as AreaDetector
from PenguTrack.Detectors import TCellDetector
from PenguTrack.Detectors import NKCellDetector
from PenguTrack.Detectors import NKCellDetector2
from PenguTrack.Detectors import TresholdSegmentation, VarianceSegmentation
from PenguTrack.Detectors import Measurement as PT_Measurement
from PenguTrack.DataFileExtended import DataFileExtended

import scipy.stats as ss
from scipy.ndimage.measurements import label
from scipy.ndimage.filters import gaussian_filter
try:
    from skimage.filters import threshold_niblack
except IOError:
    from skimage.filters import threshold_otsu as threshold_niblack
import time


def func(x,p0,p1):
    return p0 * x + p1


def ndargmin(array):
    """
    Works like numpy.argmin, but returns array of indices for multi-dimensional input arrays. Ignores NaN entries.
    """
    return np.unravel_index(np.nanargmin(array), array.shape)


def resulution_correction(pos):
    x, y, z = pos
    res = 6.45 / 10
    return y / res, x / res, z / 10.


def Create_DB(Name,pic_path,db_path,pic_pos="pos0"):
    """
    Creates a Clickpointsdatabase
        Parameters
    --------------
    Name: String
    Specifies path with the name for the database
    pic_path: String
    The path to all the pictures
    db_path: String
    The path to the folder in which the pictures can be found
    pic_pos: String
    """
    db = clickpoints.DataFile(Name, 'w')
    images = glob.glob(pic_path)
    print(len(images))
    layer_dict = {"MinP": 0, "MinIndices": 1, "MaxP":2, "MaxIndices": 3}
    db.setPath(db_path, 1)
    for image in images:
        path = os.path.sep.join(image.split(os.path.sep)[:-1])
        file = image.split("/")[-1]
        idx = file.split("_")[2][3:]
        layer_str = file.split("_")[-1][1:-4]
        if not file.count(pic_pos):
            continue
        if layer_str.count("MinProj"):
            layer = 0
        elif layer_str.count("MinIndices"):
            layer = 1
        elif layer_str.count("MaxProj"):
            layer = 2
        elif layer_str.count("MaxIndices"):
            layer = 3
        else:
            raise ValueError("No known layer!")
        print(idx, layer)
        image = db.setImage(filename=file, path=1, layer=layer)#, frame=int(idx))
        image.sort_index = int(idx)
        image.save()


def load_tracks(path, type):
    database = DataFileExtended(path)
    Tracks = {}
    if database.getTracks(type=type)[0].markers[0].measurement is not None:
        for track in database.getTracks(type=type):
            Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
            for m in track.markers:
                meas = Measurement(1., [m.measurement[0].x,
                                        m.measurement[0].y,
                                        m.measurement[0].z])
                Tracks[track.id].update(z=meas, i=m.image.sort_index)
    else:
        for track in database.getTracks(type=type):
            Tracks.update({track.id: Filter(VariableSpeed(dim=3))})
            for m in track.markers:
                meas = Measurement(1., [m.x,
                                        m.y,
                                        0])
                Tracks[track.id].update(z=meas, i=m.image.sort_index)
    return Tracks


def get_Tracks_to_delete(Z_posis,perc):
    """
    Returns a list of track ids
        Parameters
    --------------
    Z_posis: List
    perc: int
    """
    Z_pos_ab = []
    Z_pos_be = []
    for i in Z_posis:
        if i[1]>=200 and i[3]>10:
            Z_pos_ab.append(i[1])
        if i[1]<=200 and i[3]>10:
            Z_pos_be.append(i[1])
    if len(Z_pos_ab)<=10:
        Z_pos_ab = np.asarray([9999,9999])
    else:
        Z_pos_ab = np.asarray(Z_pos_ab)
    if len(Z_pos_be)<=10:
        Z_pos_be = np.asarray([0,0])
    else:
        Z_pos_be = np.asarray(Z_pos_be)
    perc_ab = np.percentile(Z_pos_ab,perc) - 20
    perc_be = np.percentile(Z_pos_be,perc) + 20
    print("--------------------------------",perc_ab,perc_be)
    tracks_to_delete = [99999]
    for i in Z_posis:
        if i[1]>=perc_ab or i[1]<=perc_be:
            tracks_to_delete.append(int(i[0]))
    return tracks_to_delete


def getZpos(Tracks,v_fac):
    mean_z = []
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
    return np.array(mean_z)


def Stitch(db_path,db_path2,a1,a2,a3,a4,a5,max_cost,limiter):
    import shutil
    shutil.copy(db_path,db_path2)
    stitcher = Heublein_Stitcher(a1,a2,a3,a4,a5,max_cost,limiter)
    stitcher.load_tracks_from_clickpoints(db_path2, "PT_Track_Marker")
    stitcher.stitch()
    stitcher.save_tracks_to_db(db_path2, "PT_Track_Marker", function=resulution_correction)
    print ("-----------Written to DB-----------")


def create_list(Frames,db,drift=None, Drift=False, Missing=False):
    list = []
    if Missing:
        frames = Frames
    else:
        frames = Frames+1
    for i in range(frames):
        list.append({})
        PT_markers = db.getMarkers(frame=i, type='PT_Track_Marker')
        if Drift==True:
            for m in PT_markers:
                if i == 0:
                    list[i].update({m.track.id: m.correctedXY()})
                else:
                    list[i].update({m.track.id: (m.correctedXY() - drift[i-1]).tolist()})
        else:
            for m in PT_markers:
                list[i].update({m.track.id: m.correctedXY()})
    return list


def create_list2(db):
    ptTracks = db.getTracks(type='PT_Track_Marker')[:]
    list2 = []
    for track in ptTracks:
        min_frame = np.min(track.frames)
        max_frame = np.max(track.frames)
        positions = []
        for i in range(min_frame, max_frame+1):
            marker = db.getMarkers(frame=i, type='PT_Track_Marker', track=track)
            if len(marker) == 1:
                pos = marker[0].correctedXY()
            else:
                pos = [9999.9, 9999.9]
            positions.append(pos)
        list2.append(positions)
    return list2

def list2_with_drift(db, drift, tracks_del=None, del_track=False):
    ptTracks = db.getTracks(type='PT_Track_Marker')[:]
    list2 = []
    for track in ptTracks:
        if del_track == True:
            if int(track.id) in tracks_del:
                continue
        min_frame = np.min(track.frames)
        max_frame = np.max(track.frames)
        positions = []
        for i in range(min_frame, max_frame):
            if i == 0:
                marker = db.getMarkers(frame=i, type='PT_Track_Marker', track=track)
                if len(marker) == 1:
                    pos = marker[0].correctedXY()
                else:
                    pos = [9999.9, 9999.9]
                positions.append(pos)
            else:
                marker = db.getMarkers(frame=i, type='PT_Track_Marker', track=track)
                if len(marker) == 1:
                    pos = (marker[0].correctedXY() - drift[i-1]).tolist()
                else:
                    pos = [9999.9, 9999.9]
                positions.append(pos)
        list2.append(positions)
    return list2


def values(Direction_plot,velocities,db,dirt,alter_velocities, tracks_to_delete=None, del_Tracks=False):
    ptTracks = db.getTracks(type='PT_Track_Marker')[:]
    if del_Tracks:
        ptTracks_cor = []
        for i in ptTracks:
            if i.id in tracks_to_delete:
                continue
            else:
                ptTracks_cor.append(i)
    else:
        ptTracks_cor=ptTracks
    number = len(ptTracks_cor)
    counter0 = 0
    counter1 = 0
    len_count = 0
    for track in ptTracks_cor:
        min_frame = np.min(track.frames)
        max_frame = np.max(track.frames)
        if max_frame-min_frame>=20:
            len_count+=1
    vel = []
    alter_vel = []
    dire = []
    alter_dire = []
    for i,v in enumerate(Direction_plot):
        if v>=0 and velocities[i]>=1:
            counter0 += 1
            vel.append(velocities[i])
            dire.append(v)
        if v>=0 and alter_velocities[i]>=1:
            counter1 += 1
            alter_vel.append(alter_velocities[i])
            alter_dire.append(v)
    mean_v = np.mean(vel)
    mean_dire = np.mean(dire)
    mean_alter_v = np.mean(alter_vel)
    mean_alter_dire = np.mean(alter_dire)
    motile_percentage = 100*(counter0/float(len(Direction_plot)-dirt))
    motile_percentage_alter = 100*(counter1/float(len(Direction_plot)-dirt))
    return motile_percentage,mean_v,mean_dire,number,len_count, motile_percentage_alter,mean_alter_v,mean_alter_dire


def measure(step, dt, liste, Frames):
    dirt = 0
    v_factor = 0.645 / ((step - 1) * dt / 60)
    track_distances = []
    track_distances_for_mean = []
    alternative_track_distances = []
    alternative_track_distances_for_mean = []
    Directions = []
    Directions_for_mean = []
    a = range(0, Frames - step, 1)
    for j in a:
        PT_tracks = liste[j].keys()
        PT_tracks_corrected = []
        # for i in range(len(PT_tracks)):
        for track in PT_tracks:
            # PT_end_marker = liste[j + step].has_key(PT_tracks[i])
            PT_end_marker = track in liste[j + step]
            if PT_end_marker == 1:
                PT_tracks_corrected.append(track)
        track_distance = []
        track_distance_for_mean = []
        alternative_track_distance = []
        alternative_track_distance_for_mean = []
        Directions2 = []
        Directions2_for_mean = []
        for track in PT_tracks_corrected:
            PT_positions = np.asarray(
                [liste[e].get(track) for e in range(j, j + step) if liste[e].get(track) is not None])
            PT_positions1 = [liste[e].get(track) for e in range(j, j + step)]
            for i, pos in enumerate(PT_positions1):
                if np.any(pos == None):
                    PT_positions1[i] = [9999.9,9999.9]
            #Speed Start
            PT_maxx = np.max(PT_positions[:, 0])
            PT_minx = np.min(PT_positions[:, 0])
            PT_maxy = np.max(PT_positions[:, 1])
            PT_miny = np.min(PT_positions[:, 1])
            if np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.) >=0.01:
                track_distance.append(np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.))
                track_distance_for_mean.append({track:np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.)*v_factor})
            else:
                print("Case2")
                dirt+=1
                track_distance.append(np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.))
                track_distance_for_mean.append({track: np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.)*v_factor})
            #Speed End
            #Alternative Speed Start
            alter_dist = 0.0001
            count = 0
            for i in range(1,len(PT_positions1)):
                if np.all(PT_positions1[i]!=[9999.9,9999.9]) and np.all(PT_positions1[i-1]!=[9999.9,9999.9]):
                    alter_dist += np.sqrt((PT_positions1[i][0] - PT_positions1[i-1][0])**2. + (PT_positions1[i][1] - PT_positions1[i-1][1])**2.)
                    count += 1
            if count == 0:
                count +=1
            alter_dist = alter_dist/float(count)
            alternative_track_distance.append(alter_dist)
            alternative_track_distance_for_mean.append({track:(alter_dist* 0.645) / (dt / 60)})
            #Alternative Speed End
            # Directions start
            directions = []
            for i in range(1, len(PT_positions1) - 1):
                if (PT_positions1[i - 1] != [9999.9, 9999.9]).all() and (PT_positions1[i] != [9999.9, 9999.9]).all() and \
                        (PT_positions1[i + 1] != [9999.9, 9999.9]).all():
                    Vector1 = [PT_positions1[i][0] - PT_positions1[i - 1][0],
                               PT_positions1[i][1] - PT_positions1[i - 1][1]]
                    Vector2 = [PT_positions1[i + 1][0] - PT_positions1[i][0],
                               PT_positions1[i + 1][1] - PT_positions1[i][1]]
                    Vector1_len = np.sqrt(Vector1[0] ** 2. + Vector1[1] ** 2.)
                    Vector2_len = np.sqrt(Vector2[0] ** 2. + Vector2[1] ** 2.)
                    if Vector1_len * Vector2_len != 0:
                        Angle = np.dot(Vector1, Vector2) / (Vector1_len * Vector2_len)
                        directions.append(Angle)
            if len(directions) != 0:
                direction = np.sum(directions) / len(directions)
                Directions2.append(direction)
                Directions2_for_mean.append({track:direction})
            else:
                print("Case1")
                Directions2.append(-0.99)
                Directions2_for_mean.append({track:-0.99})
            # Directions end
        track_distances.append(track_distance)
        track_distances_for_mean.append(track_distance_for_mean)
        alternative_track_distances.append(alternative_track_distance)
        alternative_track_distances_for_mean.append(alternative_track_distance_for_mean)
        Directions.append(Directions2)
        Directions_for_mean.append(Directions2_for_mean)
    Directions_plot = []
    for direction1 in Directions:
        for i in direction1:
            Directions_plot.append(i)
    track_distances_plot = []
    for track in track_distances:
        for i in track:
            track_distances_plot.append(i)
    alter_distance = []
    for speed in alternative_track_distances:
        for i in speed:
            alter_distance.append(i)
    track_distances_plot = np.asarray(track_distances_plot)
    velocity = (track_distances_plot * 0.645) / ((step - 1) * dt / 60)
    alter_distance = np.asarray(alter_distance)
    alter_vel = (alter_distance * 0.645) / (dt / 60)
    print('Debug2')
    return Directions_plot,velocity,dirt, alter_vel, track_distances_for_mean, Directions_for_mean, alternative_track_distances_for_mean


def Drift(Frames, list2, percentile):
    Drift_list = []
    Drift_list_cor = []
    Drift_distance = []
    Drift_distance_cor = []
    test = 0
    missing = 0
    for track in list2:
        if len(track) >= Frames+1:
            test += 1
            track1 = np.asarray(track)
            PT_maxx = np.max(track1[:, 0][track1[:, 0] < 9999])
            PT_minx = np.min(track1[:, 0])
            PT_maxy = np.max(track1[:, 1][track1[:, 1] < 9999])
            PT_miny = np.min(track1[:, 1])
            Drift_list.append(track)
            Drift_distance.append(np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.))
    if test == 0:
        missing = 1
        print ("119 Missing")
        for track in list2:
            if len(track) >= Frames:
                track1 = np.asarray(track)
                PT_maxx = np.max(track1[:, 0][track1[:, 0] < 9999])
                PT_minx = np.min(track1[:, 0])
                PT_maxy = np.max(track1[:, 1][track1[:, 1] < 9999])
                PT_miny = np.min(track1[:, 1])
                Drift_list.append(track)
                Drift_distance.append(np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.))
    Drift_distance = np.asarray(Drift_distance)
    if len(Drift_distance)<=25:
        percentile+=10
    p = np.percentile(Drift_distance, percentile)
    for i, dist in enumerate(Drift_distance):
        if dist <= p:
            Drift_distance_cor.append(dist)
            Drift_list_cor.append(Drift_list[i])
    Frame_drifts = []
    for dri in Drift_list_cor:
        Frame_drift = []
        for i in range(1,len(dri)):
            if (dri[i-1] != [9999.9,9999.9]).all() and (dri[i] != [9999.9,9999.9]).all():
                x = dri[i][0] - dri[i - 1][0]
                y = dri[i][1] - dri[i - 1][1]
                Frame_drift.append([x,y])
            else:
                Frame_drift.append([np.nan,np.nan])
        Frame_drifts.append(Frame_drift)
    Frame_drifts = np.asarray(Frame_drifts)
    frames = Frames
    if test == 0:
        frames -= 1
    drift_mean = []
    for i in range(frames):
        drift_mean.append([np.nanmean(Frame_drifts[:, i, 0]), np.nanmean(Frame_drifts[:, i, 1])])
    drift_mean = np.asarray(drift_mean)
    s = np.isnan(drift_mean)
    drift_mean[s] = 0.0
    sum_drift = np.cumsum(drift_mean, axis=0)
    return sum_drift, Drift_list_cor, missing


def motiletruedist(list2):
    True_distance = []
    Track_length = []
    real_dirt = 0
    for track in list2:
        if len(track)<1:
            continue
        Track_length.append(len(track))
        track1 = np.asarray(track)
        PT_maxx = np.max(track1[:, 0][track1[:, 0] < 9999])
        PT_minx = np.min(track1[:, 0])
        PT_maxy = np.max(track1[:, 1][track1[:, 1] < 9999])
        PT_miny = np.min(track1[:, 1])
        if np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.) >= 0.1:
            True_distance.append(np.sqrt((PT_maxx - PT_minx) ** 2. + (PT_maxy - PT_miny) ** 2.)*0.645)
        else:
            real_dirt += 1
            True_distance.append(0.01)
    all = 0
    motile_true_dist = 0
    for i,v in enumerate(True_distance):
        all += Track_length[i]
        if v>=10:
            motile_true_dist += Track_length[i]
    motile_percentage_true_dist = 100*(motile_true_dist/float(all))
    return motile_percentage_true_dist, real_dirt


def Colorplot(directions, velocities, Scatterplot_Name, path = None, Save = False):
    if Save:
        plt.ioff()
    else:
        plt.ion()
    direction_array = np.asarray(directions)
    velocities_array = np.asarray(velocities)
    velocities_array2 = np.log10(velocities_array)
    fig = plt.figure(figsize=(10, 10))
    border_width = 0.18
    ax_size = [0 + border_width, 0 + border_width, 1 - 2 * border_width, 1 - 2 * border_width]
    ax = fig.add_axes(ax_size)
    cm = plt.cm.get_cmap('jet')
    xy = np.vstack([direction_array, velocities_array2])
    kd = ss.gaussian_kde(xy)(xy)
    idx = kd.argsort()
    x, y, z = direction_array[idx], velocities_array2[idx], kd[idx]
    ax.scatter(x, y, c=z, s=35, edgecolor='', alpha=1.0, cmap=cm)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.5, 1.5])
    xticks = ([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.xticks(xticks, ['-1.0', '-0.5 ', '0.0 ', '0.5', '1.0'])
    yticks = np.log10(
        [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0, 7.0, 8, 9, 10, 20, 30])
    plt.yticks(yticks, [' ', ' ', ' ', ' ', ' ', ' ', ' ', '0.1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                        '1', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '10'])
    ax.tick_params(width=1, length=5)
    ax.tick_params(direction='in')
    plt.suptitle(Scatterplot_Name, fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xlabel('Directionality', fontsize=30)
    plt.ylabel(r'Speed [$\frac{\mathrm{\mu m}}{\mathrm{min}}$]', fontsize=30)
    if Save:
        fig.savefig(path + Scatterplot_Name + '.pdf')
    else:
        plt.show()


def run(Log_Prob_Tresh, Detection_Error, Prediction_Error, Min_Size, Max_Size, db, res, start_frame=0, progress_bar=None):
    # Checks if the marker already exist. If they don't, creates them
    marker_type = db.getMarkerType(name="PT_Detection_Marker")
    if not marker_type:
        marker_type = db.setMarkerType(name="PT_Detection_Marker", color="#FF0000", style='{"scale":1.2}')

    marker_type2 = db.getMarkerType(name="PT_Track_Marker")
    if not marker_type2:
        marker_type2 = db.setMarkerType(name="PT_Track_Marker", color="#00FF00", mode=db.TYPE_Track)

    marker_type3 = db.getMarkerType(name="PT_Prediction_Marker")
    if not marker_type3:
        marker_type3 = db.setMarkerType(name="PT_Prediction_Marker", color="#0000FF")

    if not db.getMaskType(name="PT_SegMask"):
        mask_type = db.setMaskType(name="PT_SegMask", color="#FF59E3")
    else:
        mask_type = db.getMaskType(name="PT_SegMask")

    images = db.getImageIterator(start_frame=start_frame)
    model = RandomWalk(dim=3)  # Model to predict the cell movements
    # Set uncertainties
    q = int(Detection_Error)
    r = int(Prediction_Error)
    object_area = (Min_Size + Max_Size) / 2
    object_size = int(np.sqrt(object_area) / 2.)

    Q = np.diag(
        [q * object_size, q * object_size, q * object_size])  # Prediction uncertainty
    R = np.diag([r * object_size, r * object_size,
                 r * object_size])  # Measurement uncertainty
    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

    # Initialize Tracker
    FilterType = KalmanFilter
    Tracker = MultiFilter(FilterType, model, np.diag(Q),
                               np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist)
    Tracker.AssignmentProbabilityThreshold = 0.
    Tracker.MeasurementProbabilityThreshold = 0.
    Tracker.LogProbabilityThreshold = Log_Prob_Tresh

    q = int(Detection_Error)
    r = int(Prediction_Error)
    min_area = int((Min_Size / 2.) ** 2 * np.pi)
    max_area = int((Max_Size / 2.) ** 2 * np.pi)
    object_area = int((min_area + max_area) / 2.)
    object_size = int(np.sqrt(object_area) / 2.)

    Q = np.diag(
        [q * object_size * res, q * object_size * res, q * object_size * res])
    R = np.diag([r * object_size * res, r * object_size * res,
                 r * object_size * res])
    State_Dist = ss.multivariate_normal(cov=Q)
    Meas_Dist = ss.multivariate_normal(cov=R)

    Tracker.filter_args = [np.diag(Q), np.diag(R)]
    Tracker.filter_kwargs = {"meas_dist": Meas_Dist, "state_dist": State_Dist}
    Tracker.Filters.clear()
    Tracker.ActiveFilters.clear()
    Tracker.predict(u=np.zeros((model.Control_dim,)).T, i=start_frame)
    # Delete already existing tracks
    db.deleteTracks(type="PT_Track_Marker")
    db.deleteMarkers(type=marker_type)
    db.deleteMarkers(type=marker_type2)
    db.deleteMarkers(type=marker_type3)
    for image in images:

        if progress_bar is not None:
            progress_bar.increase()

        i = image.sort_index
        print("Doing Frame %s" % i)

        Index_Image = db.getImage(frame=i, layer=1).data

        # Prediction step
        Tracker.predict(u=np.zeros((model.Control_dim,)).T, i=i)

        minIndices = db.getImage(frame=i, layer=1)
        minProj = db.getImage(frame=i, layer=0)

        Positions, mask = TCellDetector().detect(minProj, minIndices)
        db.setMask(frame=i, layer=0, data=(~mask).astype(np.uint8))

        for pos in Positions:
            db.setMarker(frame=i, layer=0, y=pos.PositionX / res, x=pos.PositionY / res,
                              type=marker_type)
        if len(Positions) != 0:

            # Update Filter with new Detections
            try:
                Tracker.update(z=Positions, i=i)
            except TypeError:
                print(Tracker.filter_args)
                print(Tracker.filter_kwargs)
                raise

            # Do all DB-writing as atomic transaction (at once)
            with db.db.atomic() as transaction:

                # Get Tracks from Filter
                for k in Tracker.Filters.keys():
                    if i in Tracker.Filters[k].Measurements.keys():
                        meas = Tracker.Filters[k].Measurements[i]
                        x = meas.PositionX
                        y = meas.PositionY
                        z = meas.PositionZ
                        prob = Tracker.Filters[k].log_prob(keys=[i])
                    else:
                        x = y = z = np.nan
                        prob = np.nan

                    # Write predictions to Database
                    if i in Tracker.Filters[k].Predicted_X.keys():
                        pred_x, pred_y, pred_z = Tracker.Model.measure(Tracker.Filters[k].Predicted_X[i])

                        pred_x_img = pred_y / res
                        pred_y_img = pred_x / res

                        pred_marker = db.setMarker(frame=i, layer=0, x=pred_x_img, y=pred_y_img,
                                                   text="Track %s" % (1000 + k), type=marker_type3)

                    x_img = y / res
                    y_img = x / res
                    # Write assigned tracks to ClickPoints DataBase
                    if np.isnan(x) or np.isnan(y):
                        pass
                    else:
                        if db.getTrack(k + 1000):
                            track_marker = db.setMarker(frame=i, layer=0, type=marker_type2,
                                                        track=(1000 + k),
                                                        x=x_img, y=y_img,
                                                        text='Track %s, Prob %.2f, Z-Position %s' % (
                                                            (1000 + k), prob, z))

                            print('Set Track(%s)-Marker at %s, %s' % ((1000 + k), x_img, y_img))
                        else:
                            db.setTrack(marker_type2, id=1000 + k, hidden=False)
                            if k == Tracker.CriticalIndex:
                                db.setMarker(image=i, layer=0, type=marker_type2, x=x_img, y=y_img,
                                             text='Track %s, Prob %.2f, CRITICAL' % ((1000 + k), prob))
                            track_marker = db.setMarker(image=image, type=marker_type2,
                                                        track=1000 + k,
                                                        x=x_img,
                                                        y=y_img,
                                                        text='Track %s, Prob %.2f, Z-Position %s' % (
                                                            (1000 + k), prob, z))
                            print('Set new Track %s and Track-Marker at %s, %s' % ((1000 + k), x_img, y_img))

                        meas_entry = db.setMeasurement(marker=track_marker, log=prob, x=x, y=y, z=z)
                        meas_entry.save()

if __name__ == "__main__":
    ######## Main Iteration over Folder
    path_list = glob.glob('/home/user/NK Test/DoubleStitchTest.cdb')
    # path_list2 = glob.glob('/home/user/TCell/2017-08-08/layers_2017_08_08_24h_24hiG*_1_pos0_12Gel.cdb')
    # path_list += path_list2
    path_list = [path for path in path_list if not path.count("stitched")]
    # path_list = [path for path in path_list if not path.count("24h_1hiG_MC_LIM")]
    # path_list = [path for path in path_list if not path.count("2017_07_25_24h_24hiG_Kon_SZ_12Gel.cdb")]
    File_Name ='/home/user/TCell/Analysis.txt'
    File_Name2 = '/home/user/NKCell/Analysis.txt'
    File_Name_Drift = '/home/user/TCell/AnalysisDrCor.txt'
    File_Name_Drift2 = '/home/user/NKCell/AnalysisDrCor.txt'
    File_Name_G = '/home/user/TCell/AnalysisG.txt'
    File_Name_G2 = '/home/user/NKCell/AnalysisG.txt'
    pic_path_T = '/home/user/TCell/Scatterplots/'
    pic_path_NK = '/home/user/NKCell/Scatterplots/'
    pic_path_T_Drift = '/home/user/TCell/ScatterplotsDrCor/'
    pic_path_NK_Drift = '/home/user/NKCell/ScatterplotsDrCor/'
    pic_path_T_Elip = '/home/user/TCell/Elipplots/'
    pic_path_NK_Elip ='/home/user/NKCell/Elipplots/'
    # with open(File_Name, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
    # with open(File_Name2, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
    # with open(File_Name_G, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile %\t\t\tNon-Motile %\t\t\tDead %  \t\t\tMean velocity\t\t\tMean Directionality\t\t\tStandard Deviation Vel\t\t\tStandard Deviation Dir\t\t\tRho   \n')
    # with open(File_Name_G2, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile %\t\t\tNon-Motile %\t\t\tDead %  \t\t\tMean velocity\t\t\tMean Directionality\t\t\tStandard Deviation Vel\t\t\tStandard Deviation Dir\t\t\tRho   \n')
    # with open(File_Name_Drift, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
    # with open(File_Name_Drift2, 'wb') as f:
    #     f.write('Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
    print(path_list)
    for i in path_list:
        if i.count('RB_1_') or i.count('RB_2_') or i.count('RB_3_') or i.count('RB_5_'):
            time_step = 30
        else:
            time_step = 110
        # time_step = 30
        v_fac = 0.645 / (time_step / 60.)
        perc = 30
        type = 'PT_Track_Marker'
        Scatterplot_Name = i[i.find("layers_"):][7:]
        Scatterplot_Name = Scatterplot_Name.split('.')[0]
        Text_Name = Scatterplot_Name[11:]
        Scatterplot_Name = Scatterplot_Name[5:]
        # Text_Name = Scatterplot_Name
        Day = i.split("/")[4]
        db_path = i
        db_path2 = i.split('.')[0] + "_stitched.cdb"
        db_path3 = i.split('.')[0] + "_stitched2.cdb"
        # if os.path.isfile(db_path2):
        #     os.remove(db_path2)
        # if os.path.isfile(db_path3):
        #     os.remove(db_path3)
        step = 20
        db = DataFileExtended(db_path)
        images = db.getImageIterator(start_frame=0)
        Frame_list = []
        for m in images:
            Frame_list.append(m.sort_index)
        Frames = np.max(Frame_list)
        res = 6.45 / 10
        run(-19.,6,4,7,30,db,res, start_frame=101)            # Tracking    #### extra errors * 2 + start####
        db.db.close()
        try:
            Stitch(db_path,db_path2,3,0.4,18,30,1,100,100)    # First Stitching   #### extra changed ####
        except IndexError:
            pass
        try:
            Stitch(db_path2,db_path3,10,5,10,10,1,100,100)  # Second Stitching
        except IndexError:
            pass

        db = clickpoints.DataFile(db_path3)
        ### For Deleting Tracks above and below
        Tracks = load_tracks(db_path3, type)
        Z_posis = getZpos(Tracks, v_fac)
        Z_posis = np.asarray(Z_posis)
        tracks_to_delete = get_Tracks_to_delete(Z_posis, perc)
        ###
        list2 = create_list2(db)  # Create List for true dist
        drift,drift_list, missing_frame = Drift(Frames, list2, 5)  # Create List with offsets
        list2 = list2_with_drift(db,drift,tracks_to_delete,del_track=True) # Create list for true dist with drift_cor
        list = create_list(Frames, db, drift=drift, Drift=True, Missing=missing_frame)  # Create List for analysis
        ### For Deleting Tracks above and below
        list_copy = list[:]
        for l, m in enumerate(list):
            keys = m.keys()
            for k, j in enumerate(keys):
                if j in tracks_to_delete:
                    del list_copy[l][j]
        ###
        directions,velocities,dirt,alternative_vel, vel_mean, dir_mean, alt_vel_mean = measure(step,time_step,list)  # Calculate directions and velocities
        # gaussian_data, gaussian_dataset, gaussian_weights, gaussian_means = Gaussian_mix(directions, alternative_vel, 100, 10000)
        motile_percentage,mean_v,mean_dire,number,len_count,mo_p_al,me_v_al,me_d_al = values(directions,velocities,db,dirt,alternative_vel,tracks_to_delete,del_Tracks=True)
        motile_per_true_dist, real_dirt = motiletruedist(list2)
        # Vel_mean, Dir_mean = for_mean_plots(vel_mean,dir_mean)
        # alt_Vel_mean = for_mean_plots(alt_vel_mean, dir_mean, Vel_and_Dir=False)
        # mean_plots(Vel_mean,Dir_mean)
        # mean_plots(alt_Vel_mean,Dir_mean)
        # ind_max = np.argmax(gaussian_means[:,1])
        # ind_min = np.argmin(gaussian_means[:,1])
        # ind_b = 3-ind_max-ind_min
        # cov = gaussian_dataset.covariances_
        # std_vel = np.sqrt(cov[ind_max][0,0])
        # std_dir = np.sqrt(cov[ind_max][1,1])
        # rho = cov[ind_max][0,1]/np.sqrt(cov[ind_max][0,0]*cov[ind_max][0,0])
        if i.count('TCell'):
            Colorplot1(directions, velocities, Scatterplot_Name, path=pic_path_T_Drift, Save=True)
            # Colorplot1(directions,velocities,Scatterplot_Name, path=pic_path_T, Save=True)
            # Elip_plot(gaussian_data,gaussian_dataset, Scatterplot_Name, path=pic_path_T_Elip ,Save=True)
            # with open(File_Name_G, 'ab') as f:
            #     f.write('%s\t\t\t%34s\t\t\t%8f\t\t\t%12f\t\t\t%6f\t\t\t%13f\t\t\t%19f\t\t\t%22f\t\t\t%22f\t\t\t%6f\n'%(Day, Text_Name, gaussian_weights[ind_max], gaussian_weights[ind_b], gaussian_weights[ind_min], gaussian_means[:,1][ind_max], gaussian_means[:,0][ind_max], std_vel, std_dir, rho))
            with open(File_Name_Drift, 'ab') as f:
                f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n'%(Day, Text_Name,motile_per_true_dist, motile_percentage, mean_v,mean_dire, me_v_al, number, len_count, real_dirt))
            # with open(File_Name, 'ab') as f:
            #     f.write('%s\t\t\t%18s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n'%(Day, Text_Name,motile_per_true_dist, motile_percentage, mean_v,mean_dire, number, len_count, real_dirt))
        elif i.count('NKCell'):
            Colorplot1(directions, velocities, Scatterplot_Name, path=pic_path_NK_Drift, Save=True)
            # Colorplot1(directions,velocities,Scatterplot_Name, path=pic_path_NK, Save=True)
            # Elip_plot(gaussian_data, gaussian_dataset, Scatterplot_Name, path=pic_path_NK_Elip, Save=True)
            # with open(File_Name_G2, 'ab') as f:
            #     f.write('%s\t\t\t%34s\t\t\t%8f\t\t\t%12f\t\t\t%6f\t\t\t%13f\t\t\t%19f\t\t\t%22f\t\t\t%22f\t\t\t%6f\n'%(Day, Text_Name, gaussian_weights[ind_max], gaussian_weights[ind_b], gaussian_weights[ind_min], gaussian_means[:,1][ind_max], gaussian_means[:,0][ind_max], std_vel, std_dir, rho))
            with open(File_Name_Drift2, 'ab') as f:
                f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n'%(Day, Text_Name,motile_per_true_dist, motile_percentage, mean_v,mean_dire, me_v_al, number, len_count, real_dirt))
            # with open(File_Name2, 'ab') as f:
            #     f.write('%s\t\t\t%18s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n'%(Day, Text_Name,motile_per_true_dist, motile_percentage, mean_v,mean_dire, number, len_count, real_dirt))
        db.db.close()
        print("--- Done with db %s ---" %(i))
    print("-------Done-------")
    ########

    ####### Main
    # Scatterplot_Name = "SCA"
    # Frames = 119
    # # Name = "layers_2017_06_07_24h_SCA.cdb"# DB name
    # # pic_path = "/mnt/cmark2/T-Cell-Motility/2017-06-07/24hnachMACS/Kontrolle/SCA/*.tif"# Path for the pictures
    # # db_connection_path = "/mnt/cmark2/T-Cell-Motility/2017-06-07/24hnachMACS/Kontrolle/SCA/"
    # db_path = "/home/user/TCell/2017-05-30/layers_2017_05_30_24h_1hiG_BIS.cdb"
    # db_path2 = "/home/user/TCell/2017-05-30/layers_2017_05_30_24h_1hiG_BIS_stitched.cdb"
    # db_path3 = "/home/user/TCell/2017-05-30/layers_2017_05_30_24h_1hiG_BIS_stitched2.cdb"
    # # pic_pos = "pos01"
    # step = 20
    # # Create_DB(Name,pic_path,db_connection_path,pic_pos)       # Create DB with correct Layers
    # # Tracking fehlt                               # Tracking
    # Stitch(db_path,db_path2,5,2,20,30,1,100,10)    # First Stitching
    # Stitch(db_path2,db_path3,10,5,10,10,1,100,10)  # Second Stitching
    # db = clickpoints.DataFile(db_path3)
    # list = create_list(Frames, db)                 # Create List for analysis
    # directions,velocities,dirt = measure(step,15,list)  # Calculate directions and velocities
    # motile_percentage,mean_v,mean_dire,number,len_count = values(directions,velocities,db,dirt)
    # print(motile_percentage,mean_v,mean_dire,number,len_count,dirt)
    #######







