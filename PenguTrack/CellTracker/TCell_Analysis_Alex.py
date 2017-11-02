from __future__ import division, print_function
import datetime
import os
import numpy as np
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.CellTracker import Analysis_Tools



from time import time
global START
START = time()
def timer(text=""):
    global START
    print(text, time() - START)
    START = time()




def Track(db, progress_bar=None):
    res = 6.45 / 10
    Analysis_Tools.run(-19., 6, 4, 7, 30, db, res, start_frame=1, progress_bar=progress_bar)


def getDateFromPath(path):
    path = os.path.normpath(path)
    for s in path.split(os.path.sep):
        try:
            return datetime.datetime.strptime(s, "%Y-%M-%d")
        except ValueError:
            pass
    return None


def Stitch(db_str):
    Analysis_Tools.Stitch(db_str, db_str[:-4] + "_stitched.cdb", 3, 0.4, 18, 30, 1, 100, 100)
    Analysis_Tools.Stitch(db_str[:-4] + "_stitched.cdb", db_str[:-4] + "_stitched2.cdb", 10, 5, 10, 10, 1, 100, 100)


def AnalyzeDB(db_str):
    db = DataFileExtended(db_str)
    time_step = 110
    v_fac = 0.645 / (time_step / 60.)
    perc = 30
    step = 20
    type = 'PT_Track_Marker'
    Frame_list = []
    for f in db.getImageIterator():
        Frame_list.append(f.sort_index)
    try:
        Frames = np.amax(Frame_list)
    except ValueError:
        Frames = 0
    timer()
    Tracks = Analysis_Tools.load_tracks(db_str, type)
    timer("Loading Tracks")

    Z_posis = Analysis_Tools.getZpos(Tracks, v_fac)
    timer("Fetching Z")

    Z_posis = np.asarray(Z_posis)
    tracks_to_delete = Analysis_Tools.get_Tracks_to_delete(Z_posis, perc)
    timer("Getting Tracks to delete")

    ###
    list2 = Analysis_Tools.create_list2(db)  # Create List for true dist
    timer("Creating List2")

    drift, drift_list, missing_frame = Analysis_Tools.Drift(Frames, list2, 5)  # Create List with offsets
    timer("Getting Drift")

    list2 = Analysis_Tools.list2_with_drift(db, drift, tracks_to_delete,
                                            del_track=True)  # Create list for true dist with drift_cor
    timer("List2 with drift")

    list = Analysis_Tools.create_list(Frames, db, drift=drift, Drift=True,
                                      Missing=missing_frame)  # Create List for analysis
    timer("Create List")
    ### For Deleting Tracks above and below
    list_copy = list[:]
    for l, m in enumerate(list):
        keys = m.keys()
        for k, j in enumerate(keys):
            if j in tracks_to_delete:
                del list_copy[l][j]
    timer("Stuff")

    ###
    print("bla")
    directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = Analysis_Tools.measure(step,
                                                                                                             time_step,
                                                                                                             list,
                                                                                                             Frames)  # Calculate directions and velocities
    timer("Measure")

    motile_percentage, mean_v, mean_dire, number, len_count, mo_p_al, me_v_al, me_d_al = Analysis_Tools.values(
        directions,
        velocities,
        db,
        dirt,
        alternative_vel,
        tracks_to_delete,
        del_Tracks=True)
    timer("Values")
    motile_per_true_dist, real_dirt = Analysis_Tools.motiletruedist(list2)
    timer("Motile True Dist")


    if not os.path.exists(db_str[:-4] + "_analyzed.txt"):
        with open(db_str[:-4] + "_analyzed.txt", "w") as f:
            f.write(
                'Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')

    Day = getDateFromPath(db_str).strftime("%Y-%M-%d")
    if db_str.count('TCell') or db_str.count('T-Cell'):
        Analysis_Tools.Colorplot(directions, velocities, db_str,
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)  # Save the velocity vs directionality picture
        with open(db_str[:-4] + "_analyzed.txt", "ab") as f:
            f.write(
                '%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                    Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                    len_count, real_dirt))
    elif db_str.count('NKCell'):
        Analysis_Tools.Colorplot(directions, velocities, db_str,
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)
        with open(db_str[:-4] + "_analyzed.txt", 'ab') as f:
            f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                len_count, real_dirt))
    db.db.close()

    timer("Write")

if __name__ == "__main__":
    # AnalyzeDB("/home/alex/2017-03-10_Tzellen_microwells_bestdata/1T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_24himGel_Kontrolle_RB_1.cdb")
    AnalyzeDB(r"Z:\T-Cell-Motility\2017-10-17\1_2Gel\24hnachMACS\1himGel\Kontrolle\RB\4T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_1himGel_Kontrolle_RB_4_stitched2.cdb")
