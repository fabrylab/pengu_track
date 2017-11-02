from __future__ import division, print_function
import clickpoints
import datetime
from qtpy import QtWidgets, QtGui
import sys
import os
import fnmatch
import TCell_Analysis
import numpy as np
from PenguTrack.DataFileExtended import DataFileExtended
from glob import glob


from time import time

global START
START = time()

def timer(text=""):
    global START
    print(text, time() - START)
    START = time()


def Track(db, progress_bar=None):
    res = 6.45 / 10
    TCell_Analysis.run(-19., 6, 4, 7, 30, db, res, start_frame=1, progress_bar=progress_bar)

def getDateFromPath(path):
    path = os.path.normpath(path)
    for s in path.split(os.path.sep):
        try:
            return datetime.datetime.strptime(s, "%Y-%M-%d")
        except ValueError:
            pass
    return None

def Stitch(db_str):
    TCell_Analysis.Stitch(db_str, db_str[:-4] + "_stitched.cdb", 3, 0.4, 18, 30, 1, 100, 100)
    TCell_Analysis.Stitch(db_str[:-4] + "_stitched.cdb", db_str[:-4] + "_stitched2.cdb", 10, 5, 10, 10, 1, 100, 100)

def AnalyzeDB(db_str):
    db = DataFileExtended(db_str)
    time_step = 110
    v_fac = 0.645 / (time_step / 60.)
    perc = 30
    step = 20
    type = 'PT_Track_Marker'
    Frame_list = [f.sort_index for f in db.getImages()]
    Frames = np.amax(Frame_list)

    timer()
    Tracks = db.PT_tracks_from_db(type)
    timer("Normal Tracks")
    Tracks_woMeasurements = db.PT_tracks_from_db(type, get_measurements=False)
    timer("Tracks WO")

    Z_posis = Z_from_PT_Track(Tracks, v_fac)
    timer("Z_posis")


    tracks_to_delete = deletable_tracks(Z_posis, perc)
    timer("Trackstodelete")

    list2 = getXY(Tracks_woMeasurements)  # Create List for true dist
    timer("GETXY")

    drift, drift_list, missing_frame = TCell_Analysis.Drift(Frames, list2, 5)  # Create List with offsets
    timer("Drift")
    list2 = TCell_Analysis.list2_with_drift(db, drift, tracks_to_delete,
                                            del_track=True)  # Create list for true dist with drift_cor

    list2 = getXY_drift_corrected(Tracks_woMeasurements,np.vstack([[0,0],drift]))
    timer("CorrectXY")

    list1 = analysis_list_from_tracks(Frames, Tracks_woMeasurements, drift=drift, Drift=True, Missing=missing_frame)  # Create List for analysis

    ### For Deleting Tracks above and below
    list_copy = list1[:]
    for l, m in enumerate(list1):
        keys = m.keys()
        for k, j in enumerate(keys):
            if j in tracks_to_delete:
                del list_copy[l][j]

    directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = TCell_Analysis.measure(step,
                                                                                                             time_step,
                                                                                                             list1,
                                                                                                             Frames)  # Calculate directions and velocities
    timer("Measure")

    motile_percentage, mean_v, mean_dire, number, len_count, mo_p_al, me_v_al, me_d_al = TCell_Analysis.values(
        directions,
        velocities,
        db,
        dirt,
        alternative_vel,
        tracks_to_delete,
        del_Tracks=True)
    timer("Values")
    motile_per_true_dist, real_dirt = TCell_Analysis.motiletruedist(list2)
    timer("Motile True Dist")

def analysis_list_from_tracks(Frames,Tracks_woMeasurement,drift=None, Drift=False, Missing=False):
    Tracks = Tracks_woMeasurement
    list = []
    if Missing:
        frames = Frames
    else:
        frames = Frames+1
    for i in range(frames):
        if Drift==True:
            list.append(dict([[track_id, Tracks[track_id].X[i][:2]-drift[i-1]] for track_id in Tracks if i in Tracks[track_id].X]))
        else:
            list.append(dict([[track_id, Tracks[track_id].X[i][:2]] for track_id in Tracks if i in Tracks[track_id].X]))
    return list


def Z_from_PT_Track(Tracks, v_fac):
    output = []
    for track_id in Tracks:
        # get all frames with track points
        frames = sorted(Tracks[track_id].X.keys())
        start = min(frames)
        end = max(frames)

        # get the existing Positions (not all timestaps have points)
        existing_Positions = dict([[i,Tracks[track_id].Measurements.get(i).getPosition()] for i in frames])
        # get Positions and fill up empty slots with nan
        X, Y, Z = np.array([existing_Positions.get(i, np.nan*np.ones(3)) for i in range(start, end+1)]).T

        # calculate average path length in xy
        Average_XY_Path = np.nanmean(np.linalg.norm([X[1:]-X[:-1], Y[1:]-Y[:-1]], axis=0))*v_fac

        # pack together output: track_id, average path in Z, time in which track exists, average xy path
        output.append([track_id, np.nanmean(Z), len(Z), Average_XY_Path])
    return np.array(output)

def deletable_tracks(Z_posis, perc):
    track_id, Z, time, XY_dist = Z_posis.T

    # find positions above and below limits, with big enough xy-dislocation
    Z_above = [z for i, z in enumerate(Z) if (z>=200) and (XY_dist[i]>10)]
    Z_below = [z for i, z in enumerate(Z) if (z<=200) and (XY_dist[i]>10)]

    # Put dummy in there if list to short
    if len(Z_above) <= 10:
        Z_above = np.asarray([9999,9999])
    else:
        Z_above = np.array(Z_above)

    # Put dummy in there if list to short
    if len(Z_below) <= 10:
        Z_below = np.asarray([0, 0])
    else:
        Z_below = np.array(Z_below)

    # calculate percentile
    percentile_above = np.percentile(Z_above,perc) - 20
    percentile_below = np.percentile(Z_below,perc) + 20
    print("Percentiles ->",percentile_above,percentile_below)

    # Get the tracks that dont fit the percentiles (and add dummy value anyways....)
    tracks_to_delete = [j for i,j in enumerate(track_id) if
                        (Z[i]>=percentile_above) or (Z[i] <= percentile_below)]
    tracks_to_delete.append([99999])

    return tracks_to_delete

def getXY(Tracks_woMeasurements):
    Tracks = Tracks_woMeasurements

    list_XY = []
    for track_id in Tracks:
        # get all frames with track points
        frames = sorted(Tracks[track_id].X.keys())
        start = min(frames)
        end = max(frames)

        # get the existing Positions (not all timestaps have points)
        existing_Positions = dict([[i, Tracks[track_id].Measurements.get(i).getPosition()] for i in frames])
        # get Positions and fill up empty slots with dummy value
        X, Y, Z = np.array([existing_Positions.get(i, 9999.9 * np.ones(3)) for i in range(start, end + 1)]).T
        list_XY.append(np.array([X,Y]).T)

    return list_XY

def getXY_drift_corrected(Tracks_woMeasurements, drift):
    Tracks = Tracks_woMeasurements

    list_XY = []
    for track_id in Tracks:
        # get all frames with track points
        frames = sorted(Tracks[track_id].X.keys())
        start = min(frames)
        end = max(frames)

        # get the existing Positions (not all timestaps have points)
        existing_Positions = dict([[i, Tracks[track_id].Measurements.get(i).getPosition()[:2]-drift[i-1]] for i in frames])
        # get Positions and fill up empty slots with dummy value
        X, Y = np.array([existing_Positions.get(i, 9999.9 * np.ones(2)) for i in range(start, end + 1)]).T
        list_XY.append(np.array([X,Y]).T)

    return list_XY


if __name__ == "__main__":
    AnalyzeDB("/home/alex/2017-03-10_Tzellen_microwells_bestdata/1T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_24himGel_Kontrolle_RB_1.cdb")
