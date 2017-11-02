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
    Frame_list = []
    for f in db.getImageIterator():
        Frame_list.append(f.sort_index)
    Frames = np.max(Frame_list)

    timer()
    Tracks = TCell_Analysis.load_tracks(db_str, type)
    timer("Loading Tracks")

    Z_posis = TCell_Analysis.getZpos(Tracks, v_fac)
    timer("Fetching Z")

    Z_posis = np.asarray(Z_posis)
    tracks_to_delete = TCell_Analysis.get_Tracks_to_delete(Z_posis, perc)
    timer("Getting Tracks to delete")

    ###
    list2 = TCell_Analysis.create_list2(db)  # Create List for true dist
    timer("Creating List2")

    drift, drift_list, missing_frame = TCell_Analysis.Drift(Frames, list2, 5)  # Create List with offsets
    timer("Getting Drift")

    list2 = TCell_Analysis.list2_with_drift(db, drift, tracks_to_delete,
                                            del_track=True)  # Create list for true dist with drift_cor
    timer("List2 with drift")

    list = TCell_Analysis.create_list(Frames, db, drift=drift, Drift=True,
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
    directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = TCell_Analysis.measure(step,
                                                                                                             time_step,
                                                                                                             list,
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


    if not os.path.exists(db_str[:-4] + "_analyzed.txt"):
        with open(db_str[:-4] + "_analyzed.txt", "w") as f:
            f.write(
                'Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')

    Day = getDateFromPath(db_str).strftime("%Y-%M-%d")
    if db_str.count('TCell') or db_str.count('T-Cell'):
        TCell_Analysis.Colorplot(directions, velocities, db_str,
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)  # Save the velocity vs directionality picture
        with open(db_str[:-4] + "_analyzed.txt", "ab") as f:
            f.write(
                '%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                    Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                    len_count, real_dirt))
    elif db_str.count('NKCell'):
        TCell_Analysis.Colorplot(directions, velocities, db_str,
                                 path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
                                 Save=True)
        with open(db_str[:-4] + "_analyzed.txt", 'ab') as f:
            f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
                Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
                len_count, real_dirt))
    db.db.close()

    timer("Write")



class Window(QtWidgets.QWidget):
# class Window(QtWidgets):
    def __init__(self):
        super(Window, self).__init__()
        self.resize(640, 360)
        # self.move(300,300)
        self.setWindowTitle("Database Creator")
        self.Layout = QtWidgets.QVBoxLayout()

        self.Matches = set()
        self.MatchedFiles = {}

        self.Tree = QtWidgets.QTreeWidget(self)
        self.Layout.addWidget(self.Tree)

        self.Tree.Header = QtWidgets.QTreeWidgetItem(["Directory", "Included", "Image-Count", "DataBases"])
        self.Tree.setHeaderItem(self.Tree.Header)
        self.Tree.itemClicked.connect(self.treeItemClicked)
        self.Dirs = {'': self.Tree}

        def increase(bar):
            QtGui.QGuiApplication.processEvents()
            bar.setValue(bar.value()+1)
            QtGui.QGuiApplication.processEvents()

        self.Button = QtWidgets.QPushButton("Add Folder")
        self.Button.clicked.connect(self.AddFolder)
        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(self.Button)
        self.Bar = QtWidgets.QProgressBar()
        self.Bar.increase = lambda : increase(self.Bar)
        sublayout.addWidget(self.Bar)
        # self.Layout.addWidget(sublayout)
        self.Layout.addLayout(sublayout)

        self.Button2 = QtWidgets.QPushButton("Create DataBases")
        self.Button2.clicked.connect(self.CreateDataBases)
        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(self.Button2)
        self.Bar2 = QtWidgets.QProgressBar()
        self.Bar2.increase = lambda : increase(self.Bar2)
        sublayout.addWidget(self.Bar2)
        self.Layout.addLayout(sublayout)

        self.Button3 = QtWidgets.QPushButton("Track")
        self.Button3.clicked.connect(self.TrackDataBases)
        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(self.Button3)
        self.Bar3 = QtWidgets.QProgressBar()
        self.Bar3.increase = lambda : increase(self.Bar3)
        sublayout.addWidget(self.Bar3)
        self.Layout.addLayout(sublayout)

        # def setValExt(bar, val):
        #     bar.setValue(val)
        #     # QtGui.QApplication.processEvents()
        #     QtGui.QGuiApplication.processEvents()
        # self.Bar3.setValueExtended = lambda x: setValExt(self.Bar3, x)

        self.Button4 = QtWidgets.QPushButton("Stitch")
        self.Button4.clicked.connect(self.StitchDataBases)
        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(self.Button4)
        self.Bar4 = QtWidgets.QProgressBar()
        self.Bar4.increase = lambda : increase(self.Bar4)
        sublayout.addWidget(self.Bar4)
        self.Layout.addLayout(sublayout)

        self.Button5 = QtWidgets.QPushButton("Analyze")
        self.Button5.clicked.connect(self.AnalyzeDataBases)
        sublayout = QtWidgets.QHBoxLayout()
        sublayout.addWidget(self.Button5)
        self.Bar5 = QtWidgets.QProgressBar()
        self.Bar5.increase = lambda : increase(self.Bar5)
        sublayout.addWidget(self.Bar5)
        self.Layout.addLayout(sublayout)

        self.TruthDict = {"True": True, "False": False, "": False}

        self.setLayout(self.Layout)

        self.layer_dict = {"MinP": 0, "MinIndices": 1, "MaxP": 2, "MaxIndices": 3}


    def treeItemClicked(self, elem, col):
        if col == 1:
            self.ToggleInclusion(elem)

    def ToggleInclusion(self, elem):
        if elem.text(1) == "True":
            self.falsify(elem)
        elif elem.text(1) == "False":
            self.trueify(elem)

    def falsify(self, elem):
        j = [i for i in self.Dirs if self.Dirs[i] == elem][0]
        self.Dirs[j].setText(1, "False")
        if j in self.Matches:
            self.Matches.discard(j)
        try:
            children = self.Dirs[j].children()
        except AttributeError:
            children = []
        for c in children:
            self.falsify(c)

    def trueify(self, elem):
        j = [i for i in self.Dirs if self.Dirs[i] == elem][0]
        self.Dirs[j].setText(1, "True")
        self.Matches.update([j])
        try:
            children = self.Dirs[j].children()
        except AttributeError:
            children = []
        for c in children:
            self.trueify(c)

    def AddFolder(self, *args, **kwargs):
        print(args)
        print(kwargs)
        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        walker = [x for x in os.walk(folder)]
        t = 0
        self.Bar.setValue(0)
        self.Bar.setRange(0, len(walker))
        self.Bar.setFormat("Walking Directories")
        for root, dirnames, filenames in walker:
            self.Bar.increase()
            filtered = fnmatch.filter(filenames, "*.tif")
            if len(filtered) > 0:
                root=os.path.normpath(root)
                self.Matches.update([root])
                self.MatchedFiles.update({root: filtered})

        t = 0
        self.Bar.setValue(0)
        self.Bar.setFormat("Finding Images")
        self.Bar.setRange(0, len(self.Matches))
        for m in self.Matches:
            self.Bar.increase()
            for i, mm in enumerate(m.split(os.path.sep)):
                part = os.path.sep.join(m.split(os.path.sep)[:i])
                part_m1 = os.path.sep.join(m.split(os.path.sep)[:i - 1])
                if part not in self.Dirs:
                    self.Dirs.update({part: QtWidgets.QTreeWidgetItem(self.Dirs[part_m1],
                                                                      [part.split(os.path.sep)[-1]])})
            if not m in self.Dirs:
                self.Dirs.update({m: QtWidgets.QTreeWidgetItem(self.Dirs[os.path.sep.join(m.split(os.path.sep)[:-1])],
                                                           [m.split(os.path.sep)[-1],
                                                            "True",
                                                            str(len(self.MatchedFiles[m])),
                                                            str([g.split(os.path.sep)[-1] for g in glob(m+"*.cdb")])])})
        self.Tree.expandAll()
        self.Bar.setFormat("Done")

    def CreateDataBases(self):
        overwriting = []
        for item in self.Dirs.values():
            try:
                included = self.TruthDict[item.text(1)]
                exists =  False #self.TruthDict[item.text(3)]
            except AttributeError:
                continue
            if exists and included:
                overwriting.append(item.text(0))

        if self.Overwriting_Dialog(overwriting):
            self.Bar2.setFormat("Creating DataBases")
            self.Bar2.setValue(0)
            self.Bar2.setRange(0, np.sum([len(self.MatchedFiles[m]) for m in self.Matches]))
            t=0
            for m in self.Matches:
                # t+=1
                # self.Bar2.setValue(t)
                self.CreateDB(m)
            self.Bar2.setFormat("Done")


    def Overwriting_Dialog(self, overwriting):
        if len(overwriting) < 1:
            return True
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Existing Database")
        msg.setInformativeText("You are going to overwrite existing DataBases %s. Proceed?" % ", ".join(overwriting))
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        retval = msg.exec_()
        print("value of pressed message box button:", retval)
        if retval == 16384:
            return True
        else:
            return False

    def name_from_path(self, path):
        blocks = []
        for s in path.split("\\"):
            for ss in s.split("/"):
                blocks.append(ss)
        return "_".join(blocks[1:])


    def CreateDB(self, Folder):
        Files = self.MatchedFiles[Folder]
        db_path = os.path.sep.join(Folder.split(os.path.sep)[:-1]+[""])
        db_name = self.name_from_path(Folder)
        db = DataFileExtended(db_path+db_name+".cdb", "w")
        path = db.setPath(Folder)
        idx_dict = {}
        for file in Files:
            # self.Bar2.setValue(self.Bar2.value()+1)
            self.Bar2.increase()
            # QtGui.QApplication.processEvents()
            QtGui.QGuiApplication.processEvents()
            layer = self.layer_dict[[k for k in self.layer_dict if file.count(k)][0]]
            time = datetime.datetime.strptime(file.split("_")[0], "%Y%m%d-%H%M%S")
            idx = int([k[3:] for k in file.split("_") if k.count("rep")][0])
            if len(idx_dict)<1:
                idx_dict.update({idx: 0})
            elif idx not in idx_dict:
                idx_dict.update({idx: max(idx_dict.values())+1})
            image = db.setImage(filename=file, path=path, layer=layer, timestamp=time)  # , frame=int(idx))
            image.sort_index = idx#idx_dict[idx]
            image.save()

    def TrackDataBases(self):
        self.Bar3.setValue(0)
        self.Bar3.setRange(0, np.sum([len(self.MatchedFiles[m]) for m in self.Matches])/4)
        for m in self.Matches:
            self.Bar3.setFormat("Tracking %s"%m)
            self.Track(m)
        self.Bar3.setFormat("Done")

    def Track(self, Folder):
        Track(Folder)
        db_name = self.name_from_path(Folder)
        db_path = os.path.sep.join(Folder.split(os.path.sep)[:-1]+[""])
        if not os.path.exists(db_path + db_name + ".cdb"):
            self.CreateDB(Folder)
        db = DataFileExtended(db_path+ db_name +".cdb")
        print([i.sort_index for i in db.getImageIterator()])
        Track(db, progress_bar=self.Bar3)

    def StitchDataBases(self):
        self.Bar4.setValue(0)
        self.Bar4.setRange(0, len(self.Matches))
        for m in self.Matches:
            self.Bar4.increase()
            self.Stitch(m)

    def Stitch(self, m):
        db_name = self.name_from_path(m)
        Stitch(m+db_name+".cdb")
        # TCell_Analysis.Stitch(m+db_name+".cdb",m+db_name+"_stitched.cdb",3,0.4,18,30,1,100,100)
        # TCell_Analysis.Stitch(m+db_name+"_stitched.cdb",m+db_name+"_stitched2.cdb",10,5,10,10,1,100,100)

    def getDateFromPath(self, path):
        return getDateFromPath(path)
        # for s in path.split(os.path.sep):
        #     if s.count("/"):
        #         for ss in s.split("/"):
        #             try:
        #                 return datetime.datetime.strptime(ss, "%Y-%M-%d")
        #             except ValueError:
        #                 pass
        #     else:
        #         try:
        #             return datetime.datetime.strptime(s, "%Y-%M-%d")
        #         except ValueError:
        #             pass
        # return None

    def AnalyzeDB(self, db_str):
        AnalyzeDB(db_str)
        # db = DataFileExtended(db_str)
        #
        # time_step = 110
        # v_fac = 0.645 / (time_step / 60.)
        # perc = 30
        # step = 20
        # type = 'PT_Track_Marker'
        # Frame_list = []
        # for f in db.getImageIterator():
        #     Frame_list.append(f.sort_index)
        # Frames = np.max(Frame_list)
        #
        # Tracks = TCell_Analysis.load_tracks(db_str, type)
        # Z_posis = TCell_Analysis.getZpos(Tracks, v_fac)
        # Z_posis = np.asarray(Z_posis)
        # tracks_to_delete = TCell_Analysis.get_Tracks_to_delete(Z_posis, perc)
        # ###
        # list2 = TCell_Analysis.create_list2(db)  # Create List for true dist
        # drift, drift_list, missing_frame = TCell_Analysis.Drift(Frames, list2, 5)  # Create List with offsets
        # list2 = TCell_Analysis.list2_with_drift(db, drift, tracks_to_delete,
        #                                         del_track=True)  # Create list for true dist with drift_cor
        # list = TCell_Analysis.create_list(Frames, db, drift=drift, Drift=True,
        #                                   Missing=missing_frame)  # Create List for analysis
        # ### For Deleting Tracks above and below
        # list_copy = list[:]
        # for l, m in enumerate(list):
        #     keys = m.keys()
        #     for k, j in enumerate(keys):
        #         if j in tracks_to_delete:
        #             del list_copy[l][j]
        # ###
        # print("bla")
        # directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = TCell_Analysis.measure(step,
        #                                                                                                          time_step,
        #                                                                                                          list,
        #                                                                                                          Frames)  # Calculate directions and velocities
        # motile_percentage, mean_v, mean_dire, number, len_count, mo_p_al, me_v_al, me_d_al = TCell_Analysis.values(directions,
        #                                                                                                            velocities,
        #                                                                                                            db,
        #                                                                                                            dirt,
        #                                                                                                            alternative_vel,
        #                                                                                                            tracks_to_delete,
        #                                                                                                            del_Tracks=True)
        # motile_per_true_dist, real_dirt = TCell_Analysis.motiletruedist(list2)
        #
        # if not os.path.exists(db_str[:-4] + "_analyzed.txt"):
        #     with open(db_str[:-4] + "_analyzed.txt", "w") as f:
        #         f.write(
        #             'Day       \t\t\tData                              \t\t\tMotile % true dist\t\t\tMotile in %\t\t\tMean velocity\t\t\tMean Directionality\t\t\tMean vel dt1\t\t\t#Tracks\t\t\t#Evaluated Tracks\t\t\t#Dirt\n')
        #
        # Day = self.getDateFromPath(db_str).strftime("%Y-%M-%d")
        # if db_str.count('TCell') or db_str.count('T-Cell'):
        #     TCell_Analysis.Colorplot(directions, velocities, db_str,
        #                              path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
        #                              Save=True)  # Save the velocity vs directionality picture
        #     with open(db_str[:-4] + "_analyzed.txt", "ab") as f:
        #         f.write(
        #             '%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
        #                 Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
        #                 len_count, real_dirt))
        # elif db_str.count('NKCell'):
        #     TCell_Analysis.Colorplot(directions, velocities, db_str,
        #                              path=os.path.sep.join(db_str.split(os.path.sep)[:-1]),
        #                              Save=True)
        #     with open(db_str[:-4] + "_analyzed.txt", 'ab') as f:
        #         f.write('%s\t\t\t%34s\t\t\t%18f\t\t\t%11f\t\t\t%13f\t\t\t%19f\t\t\t%12f\t\t\t%7d\t\t\t%17d\t\t\t%5d\n' % (
        #             Day, db_str, motile_per_true_dist, motile_percentage, mean_v, mean_dire, me_v_al, number,
        #             len_count, real_dirt))
        # db.db.close()

    def AnalyzeFolder(self, Folder):
        m=Folder
        db_name = self.name_from_path(m)
        db_path = os.path.sep.join(m.split(os.path.sep)[:-1] + [""])

        if os.path.exists(db_path + db_name + "_stitched2.cdb"):
            # db_path = m+db_name+"_stitched2.cdb"
            db_str = db_path + db_name + "_stitched2.cdb"
        elif os.path.exists(db_path + db_name + "_stitched.cdb"):
            # db_path = m+db_name+"_stitched.cdb"
            db_str = db_path + db_name + "_stitched.cdb"
        elif os.path.exists(db_path + db_name + ".cdb"):
            db_str = db_path + db_name + ".cdb"
            # db_path = m+db_name + ".cdb"
        else:
            self.CreateDB(m)
            self.Track(m)
            db_str = db_path + db_name + ".cdb"
            # db_path = m +db_name + ".cdb"
        self.AnalyzeDB(db_str)



    def AnalyzeDataBases(self):
        self.Bar5.setValue(0)
        self.Bar5.setRange(0, len(self.Matches))
        for m in self.Matches:
            self.Bar5.increase()
            self.AnalyzeFolder(m)



def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # main()
    AnalyzeDB("/home/alex/2017-03-10_Tzellen_microwells_bestdata/1T-Cell-Motility_2017-10-17_1_2Gel_24hnachMACS_24himGel_Kontrolle_RB_1.cdb")
