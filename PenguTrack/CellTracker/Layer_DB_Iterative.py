from __future__ import division, print_function
import clickpoints
import datetime
from qtpy import QtWidgets
import sys
import os
import fnmatch
import TCell_Analysis
import numpy as np
from PenguTrack.DataFileExtended import DataFileExtended
from glob import glob

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.resize(640, 360)
        self.resize(640, 360)
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

        self.Button = QtWidgets.QPushButton("Add Folder")
        self.Button.clicked.connect(self.AddFolder)
        self.Layout.addWidget(self.Button)

        self.Button2 = QtWidgets.QPushButton("Create DataBases")
        self.Button2.clicked.connect(self.CreateDataBases)
        self.Layout.addWidget(self.Button2)

        self.Button3 = QtWidgets.QPushButton("Track")
        self.Button3.clicked.connect(self.TrackDataBases)
        self.Layout.addWidget(self.Button3)

        self.Button4 = QtWidgets.QPushButton("Stitch")
        self.Button4.clicked.connect(self.StitchDataBases)
        self.Layout.addWidget(self.Button4)

        self.Button5 = QtWidgets.QPushButton("Analyze")
        self.Button5.clicked.connect(self.AnalyzeDataBases)
        self.Layout.addWidget(self.Button5)

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

        for root, dirnames, filenames in os.walk(folder):
            filtered = fnmatch.filter(filenames, "*.tif")
            if len(filtered) > 0:
                self.Matches.update([root])
                self.MatchedFiles.update({root: filtered})
        for m in self.Matches:
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

    def CreateDataBases(self):
        overwriting = []
        for item in self.Dirs.values():
            try:
                included = self.TruthDict[item.text(1)]
                exists = self.TruthDict[item.text(3)]
            except AttributeError:
                continue
            if exists and included:
                overwriting.append(item.text(0))

        if self.Overwriting_Dialog(overwriting):
            for m in self.Matches:
                self.CreateDB(m)

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

    def CreateDB(self, Folder):
        Files = self.MatchedFiles[Folder]
        db_path = Folder + ".cdb"
        db = DataFileExtended(db_path, "w")
        path = db.setPath(Folder)
        idx_dict = {}
        for file in Files:
            layer = self.layer_dict[[k for k in self.layer_dict if file.count(k)][0]]
            time = datetime.datetime.strptime(file.split("_")[0], "%Y%m%d-%H%M%S")
            idx = int([k[3:] for k in file.split("_") if k.count("rep")][0])
            if len(idx_dict)<1:
                idx_dict.update({idx: 0})
            elif idx not in idx_dict:
                idx_dict.update({idx: max(idx_dict.values())+1})
            image = db.setImage(filename=file, path=path, layer=layer, timestamp=time)  # , frame=int(idx))
            image.sort_index = idx_dict[idx]
            image.save()

    def TrackDataBases(self):
        for m in self.Matches:
            self.Track(m)

    def Track(self, Folder):
        if not os.path.exists(Folder + ".cdb"):
            self.CreateDB(Folder)
        db = DataFileExtended(Folder+".cdb")
        print([i.sort_index for i in db.getImageIterator()])
        res = 6.45 / 10
        TCell_Analysis.run(-19.,6,4,7,30,db,res, start_frame=1)

    def StitchDataBases(self):
        for m in self.Matches:
            self.Stitch(m)

    def Stitch(self, m):
        TCell_Analysis.Stitch(m+".cdb",m+"_stitched.cdb",3,0.4,18,30,1,100,100)
        TCell_Analysis.Stitch(m+"_stitched.cdb",m+"_stitched2.cdb",10,5,10,10,1,100,100)

    def AnalyzeDataBases(self):
        for m in self.Matches:
            if os.path.exists(m+"_stitched2.cdb"):
                db_path = m+"_stitched2.cdb"
            elif os.path.exists(m+"_stitched.cdb"):
                db_path = m+"_stitched.cdb"
            elif os.path.exists(m+".cdb"):
                db_path = m + ".cdb"
            else:
                self.CreateDB(m)
                self.Track(m)
                db_path = m + ".cdb"

            db = DataFileExtended(m+".cdb")
            time_step = 110
            v_fac = 0.645 / (time_step / 60.)
            perc = 30
            step = 20
            type = 'PT_Track_Marker'
            Frame_list = []
            for f in db.getImageIterator():
                Frame_list.append(f.sort_index)
            Frames = np.max(Frame_list)

            Tracks = TCell_Analysis.load_tracks(db_path, type)
            Z_posis = TCell_Analysis.getZpos(Tracks, v_fac)
            Z_posis = np.asarray(Z_posis)
            tracks_to_delete = TCell_Analysis.get_Tracks_to_delete(Z_posis, perc)
            ###
            list2 = TCell_Analysis.create_list2(db)  # Create List for true dist
            drift, drift_list, missing_frame = TCell_Analysis.Drift(Frames, list2, 5)  # Create List with offsets
            list2 = TCell_Analysis.list2_with_drift(db, drift, tracks_to_delete,
                                     del_track=True)  # Create list for true dist with drift_cor
            list = TCell_Analysis.create_list(Frames, db, drift=drift, Drift=True, Missing=missing_frame)  # Create List for analysis
            ### For Deleting Tracks above and below
            list_copy = list[:]
            for l, m in enumerate(list):
                keys = m.keys()
                for k, j in enumerate(keys):
                    if j in tracks_to_delete:
                        del list_copy[l][j]
            ###
            directions, velocities, dirt, alternative_vel, vel_mean, dir_mean, alt_vel_mean = TCell_Analysis.measure(step, time_step,
                                                                                                      list, Frames)  # Calculate directions and velocities
            motile_percentage, mean_v, mean_dire, number, len_count, mo_p_al, me_v_al, me_d_al = TCell_Analysis.values(directions,
                                                                                                        velocities, db,
                                                                                                        dirt,
                                                                                                        alternative_vel,
                                                                                                        tracks_to_delete,
                                                                                                        del_Tracks=True)
            motile_per_true_dist, real_dirt = TCell_Analysis.motiletruedist(list2)
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
