from qtpy import QtWidgets, QtGui
from glob import glob
import fnmatch
import sys
import os
import numpy as np
from datetime import datetime
from PenguTrack.DataFileExtended import DataFileExtended
from PenguTrack.CellTracker import TCell_Analysis_Alex as TCell_Analysis


from time import time
global START
START = time()
def timer(text=""):
    global START
    print(text, time() - START)
    START = time()



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
            time = datetime.strptime(file.split("_")[0], "%Y%m%d-%H%M%S")
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
        db_name = self.name_from_path(Folder)
        db_path = os.path.sep.join(Folder.split(os.path.sep)[:-1]+[""])
        if not os.path.exists(db_path + db_name + ".cdb"):
            self.CreateDB(Folder)
        db = DataFileExtended(db_path+ db_name +".cdb")
        print([i.sort_index for i in db.getImageIterator()])
        TCell_Analysis.Track(db, progress_bar=self.Bar3)

    def StitchDataBases(self):
        self.Bar4.setValue(0)
        self.Bar4.setRange(0, len(self.Matches))
        for m in self.Matches:
            self.Bar4.increase()
            self.Stitch(m)

    def Stitch(self, m):
        db_name = self.name_from_path(m)
        TCell_Analysis.Stitch(m+db_name+".cdb")

    def getDateFromPath(self, path):
        return TCell_Analysis.getDateFromPath(path)


    def AnalyzeDB(self, db_str):
        TCell_Analysis.AnalyzeDB(db_str)


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
    main()

