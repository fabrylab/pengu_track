from __future__ import division, print_function
import datetime
import os
import numpy as np
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.CellTracker import Analysis_Tools
import fnmatch

global LAYER_DICT
LAYER_DICT = {"MinP": 0, "MinIndices": 1, "MaxP": 2, "MaxIndices": 3}

from time import time
global START
START = time()
def timer(text=""):
    global START
    print(text, time() - START)
    START = time()


def name_from_path(path):
    blocks = []
    path = os.path.normpath(path)
    for s in path.split(os.path.sep):
        blocks.append(s)
    return "_".join(blocks[1:])


def CreateDB(path, files, progress_bar = None):
    Folder = os.path.normpath(path)
    Files = [os.path.normpath(f) for f in files]
    db_path = os.path.sep.join(Folder.split(os.path.sep)[:-1]+[""])

    db_name = name_from_path(Folder)
    db = DataFileExtended(db_path+db_name+".cdb", "w")
    path = db.setPath(Folder)
    idx_dict = {}
    for file in Files:
        progress_bar.increase()
        layer = LAYER_DICT[[k for k in LAYER_DICT if file.count(k)][0]]
        time = datetime.datetime.strptime(file.split("_")[0], "%Y%m%d-%H%M%S")
        idx = int([k[3:] for k in file.split("_") if k.count("rep")][0])
        if len(idx_dict)<1:
            idx_dict.update({idx: 0})
        elif idx not in idx_dict:
            idx_dict.update({idx: max(idx_dict.values())+1})
        image = db.setImage(filename=file, path=path, layer=layer, timestamp=time)  # , frame=int(idx))
        image.sort_index = idx#idx_dict[idx]
        image.save()
    return db_path+db_name+".cdb"


def Crawl_Folder(folder, progress_bar = None):
    Matches = set()
    MatchedFiles = {}
    for root, dirnames, filenames in os.walk(folder):
        if progress_bar is not None:
            progress_bar.increase()
        filtered = fnmatch.filter(filenames, "*.tif")
        if len(filtered) > 0:
            root = os.path.normpath(root)
            Matches.update([root])
            MatchedFiles.update({root: filtered})
    return Matches, MatchedFiles

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
<<<<<<< local
    try:
        Frames = np.amax(Frame_list)
    except ValueError:
        Frames = 0
    timer()
=======
    Frames = np.max(Frame_list)

    # Fetch the tracks from the DB
>>>>>>> other
    Tracks = Analysis_Tools.load_tracks(db_str, type)
    Z_posis = Analysis_Tools.getZpos(Tracks, v_fac)
    Z_posis = np.asarray(Z_posis)
    tracks_to_delete = Analysis_Tools.get_Tracks_to_delete(Z_posis, perc)
    ###
    list2 = Analysis_Tools.create_list2(db)  # Create List for true dist

    drift, drift_list, missing_frame = Analysis_Tools.Drift(Frames, list2, 5)  # Create List with offsets

    list2 = Analysis_Tools.list2_with_drift(db, drift, tracks_to_delete,
                                            del_track=True)  # Create list for true dist with drift_cor

    list = Analysis_Tools.create_list(Frames, db, drift=drift, Drift=True,
                                      Missing=missing_frame)  # Create List for analysis
    ### For Deleting Tracks above and below
    list_copy = list[:]
    for l, m in enumerate(list):
        keys = m.keys()
        for k, j in enumerate(keys):
            if j in tracks_to_delete:
                del list_copy[l][j]

    ###
    directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = Analysis_Tools.measure(step,
                                                                                                             time_step,
                                                                                                             list,
                                                                                                             Frames)  # Calculate directions and velocities
    motile_percentage, mean_v, mean_dire, number, len_count, mo_p_al, me_v_al, me_d_al = Analysis_Tools.values(
        directions,
        velocities,
        db,
        dirt,
        alternative_vel,
        tracks_to_delete,
        del_Tracks=True)
    motile_per_true_dist, real_dirt = Analysis_Tools.motiletruedist(list2)

    if not os.path.exists(db_str[:-4] + "_analyzed.txt"):
        with open(db_str[:-4] + "_analyzed.txt", "w") as f:
            f.write(
                'Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
    try:
        Day = getDateFromPath(db_str).strftime("%Y-%M-%d")
    except AttributeError:
        Day = "XXXX-XX-XX"

    if db_str.count('TCell') or db_str.count('T-Cell'):
        Analysis_Tools.Colorplot(directions, velocities, db_str.split(os.path.sep)[-1][::-4],
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)  # Save the velocity vs directionality picture
        with open(db_str[:-4] + "_analyzed.txt", "a") as f:
            f.write(
                '%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                    Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                    len_count, real_dirt))
    elif db_str.count('NKCell'):
        Analysis_Tools.Colorplot(directions, velocities, db_str.split(os.path.sep)[-1][::-4],
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)
        with open(db_str[:-4] + "_analyzed.txt", 'a') as f:
            f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                len_count, real_dirt))
    db.db.close()

if __name__ == "__main__":

<<<<<<< local
if __name__ == "__main__":
    # AnalyzeDB("/home/alex/2017-03-10_Tzellen_microwells_bestdata/1T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_24himGel_Kontrolle_RB_1.cdb")
    AnalyzeDB(r"Z:\T-Cell-Motility\2017-10-17\1_2Gel\24hnachMACS\1himGel\Kontrolle\RB\4T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_1himGel_Kontrolle_RB_4_stitched2.cdb")
=======
    Matches, Matched_Files = Crawl_Folder("/mnt/cmark2/T-Cell-Motility/2017-10-17/1_2Gel/24hnachMACS/24himGel/Kontrolle/RB/")
    Database_paths =[]
    for m in Matches:
        Database_paths.append(CreateDB(m, Matched_Files[m]))
    for db_path in Database_paths:
        AnalyzeDB(db_path)>>>>>>> other
