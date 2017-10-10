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
import os.path
import sys
from scipy.optimize import leastsq,curve_fit
from sklearn.mixture import GaussianMixture
import cv2
import peewee


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
        for i in range(min_frame, max_frame+1):
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


def measure(step, dt, list):
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
        PT_tracks = list[j].keys()
        PT_tracks_corrected = []
        for i in range(len(PT_tracks)):
            PT_end_marker = list[j + step].has_key(PT_tracks[i])
            if PT_end_marker == 1:
                PT_tracks_corrected.append(PT_tracks[i])
        track_distance = []
        track_distance_for_mean = []
        alternative_track_distance = []
        alternative_track_distance_for_mean = []
        Directions2 = []
        Directions2_for_mean = []
        for track in PT_tracks_corrected:
            PT_positions = np.asarray(
                [list[e].get(track) for e in range(j, j + step) if list[e].get(track) is not None])
            PT_positions1 = [list[e].get(track) for e in range(j, j + step)]
            for i,pos in enumerate(PT_positions1):
                if pos == None:
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
                if PT_positions1[i]!=[9999.9,9999.9] and PT_positions1[i-1]!=[9999.9,9999.9]:
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
                if PT_positions1[i - 1] != [9999.9, 9999.9] and PT_positions1[i] != [9999.9, 9999.9] and \
                                PT_positions1[i + 1] != [9999.9, 9999.9]:
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


def Drift(Frames,list2, percentile):
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
            if dri[i-1] != [9999.9,9999.9] and dri[i] != [9999.9,9999.9]:
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

def get_Tracks_to_delete(Z_posis,perc):
    Z_pos_ab = []
    Z_pos_be = []
    for i in Z_posis:
        if i[1]>=200 and i[3]>10:
            Z_pos_ab.append(i[1])
        if i[1]<=200 and i[3]>10:
            Z_pos_be.append(i[1])
    if len(Z_pos_ab)<=1:
        Z_pos_ab = np.asarray([9999,9999])
    else:
        Z_pos_ab = np.asarray(Z_pos_ab)
    if len(Z_pos_be)<=1:
        Z_pos_be = np.asarray([0,0])
    else:
        Z_pos_be = np.asarray(Z_pos_be)
    perc_ab = np.percentile(Z_pos_ab,perc) - 20
    perc_be = np.percentile(Z_pos_be,perc) + 20
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
    return mean_z

path = "/home/user/TCell/2017-08-15/layers_2017_08_15_24h_1hiG_Kon_RB_2_pos3_12Gel_stitched2.cdb"
type = "PT_Track_Marker"
dt = 15
step = 20
Frames = 119
v_fac = 0.645/(dt/60)
perc  = 10
# frames = 120
Tracks = load_tracks(path,type)
Z_posis = getZpos(Tracks,v_fac)
Z_posis = np.asarray(Z_posis)
tracks_to_delete = get_Tracks_to_delete(Z_posis,perc)
db = clickpoints.DataFile(path)
list2 = create_list2(db)  # Create List for true dist
drift,drift_list, missing_frame = Drift(Frames, list2, 5)  # Create List with offsets
list2 = list2_with_drift(db,drift,tracks_to_delete,del_track=True) # Create list for true dist with drift_cor
list = create_list(Frames, db, drift=drift, Drift=True, Missing=missing_frame)  # Create List for analysis
list_copy = list[:]
for l,i in enumerate(list):
    keys = i.keys()
    for k,j in enumerate(keys):
        if j in tracks_to_delete:
            del list_copy[l][j]
directions,velocities,dirt,alternative_vel, vel_mean, dir_mean, alt_vel_mean = measure(step,dt,list_copy)  # Calculate directions and velocities
# gaussian_data, gaussian_dataset, gaussian_weights, gaussian_means = Gaussian_mix(directions, alternative_vel, 100, 10000)
motile_percentage,mean_v,mean_dire,number,len_count,mo_p_al,me_v_al,me_d_al = values(directions,velocities,db,dirt,alternative_vel,tracks_to_delete,del_Tracks=True)
motile_per_true_dist, real_dirt = motiletruedist(list2)
print(motile_per_true_dist, motile_percentage, mean_v,mean_dire, me_v_al, number, len_count, real_dirt)

