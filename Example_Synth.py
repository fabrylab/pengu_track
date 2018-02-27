#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example_Synth.py

# Copyright (c) 2016-2017, Alexander Winterl
#
# This file is part of PenguTrack
#
# PenguTrack is free software: you can redistribute and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the Licensem, or
# (at your option) any later version.
#
# PenguTrack is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PenguTrack. If not, see <http://www.gnu.org/licenses/>.

from PenguTrack.Detectors import Measurement
import numpy as np

class SyntheticDataGenerator(object):
    def __init__(self, object_number, position_variation, speed_variation, model, loose=False):
        self.N = int(object_number)
        self.R = float(position_variation)
        self.Q = float(speed_variation)
        self.Model = model
        self.Loose = bool(loose)
        vec = np.random.rand(self.N, self.Model.State_dim, 1)
        vec [:,::2] *= self.R
        vec [1:,::2] *= self.Q
        self.Objects = {0: dict(zip(range(self.N), vec))}

        for i in range(100):
            self.step()

    def step(self):
        i = max(self.Objects.keys())
        tracks = list(self.Objects[i].keys())
        positions = [self.Objects[i][k] for k in tracks]
        self.Objects[i+1] = dict(zip(tracks,np.array([
            self.Model.predict(
            self.Model.evolute(
                self.Q*np.random.randn(
                    self.Model.Evolution_dim, 1),
                pos),
                np.zeros((self.Model.Control_dim, 1)))
            for pos in positions])))
        out = [Measurement(1.0, self.Model.measure(pos)) for pos in self.Objects[i+1].values()]
        if self.Loose:
            for k in set(np.random.randint(low=0, high=self.N, size=int(self.N*0.1))):
                try:
                    out.pop(k)
                except IndexError:
                    pass
        return out

if __name__ == '__main__':

    def track(filter, gen):
        # Extended Clickpoints Database for usage with pengutack
        from PenguTrack.DataFileExtended import DataFileExtended
        # Open ClickPoints Database
        db = DataFileExtended("./synth_data.cdb", "w")

        gt_type = db.setMarkerType(name="Ground_Truth", mode=db.TYPE_Track, color="#FFFFFF")
        db_model, db_tracker = db.init_tracker(filter.Model, filter)

        # Start Iteration over Images
        print('Starting Iteration')
        for i in range(10):
            image = db.setImage(filename="%s.png"%i, frame=i)
            # Prediction step, without applied control(vector of zeros)
            filter.predict(i=i)

            # Detection step
            Positions = gen.step()
            j = max(gen.Objects)
            i_x = gen.Model.Measured_Variables.index("PositionX")
            i_y = gen.Model.Measured_Variables.index("PositionY")
            for t in gen.Objects[j]:
                if not db.getTrack(t):
                    db.setTrack(type=gt_type, id=t)
                state = gen.Model.measure(gen.Objects[j][t])
                db.setMarker(x=state[i_y],
                             y=state[i_x],
                             type=gt_type, track=t, image=image)

            print("Found %s Objects!"%len(Positions))

            if len(Positions)>0:
                # Update Filter with new Detections
                filter.update(z=Positions, i=i, verbose=False)
                # Write everything to a DataBase
                db.write_to_DB(filter, image, i=i, db_tracker=db_tracker, db_model=db_model)


        print('done with Tracking')
        return filter

    all_params = []

    from PenguTrack.Trackers import VariableSpeedTracker

    MultiKal = VariableSpeedTracker(r=0.1, no_dist=False, prob_update=False)
    # MultiKal.LogProbabilityThreshold = -3.
    # Physical Model (used for predictions)
    from PenguTrack.Models import VariableSpeed
    from PenguTrack.DataFileExtended import DataFileExtended

    Generator = SyntheticDataGenerator(10, 10., 1., VariableSpeed(dim=2, timeconst=0.5, damping=1.), loose=True)

    MultiKal = track(MultiKal, Generator)


    db = DataFileExtended("./synth_data.cdb", "r")
    Tracker = db.tracker_from_db()[0]
    # def retrieve_db():
    #     # Extended Clickpoints Database for usage with pengutack
    #     from PenguTrack.DataFileExtended import DataFileExtended
    #     # Open ClickPoints Database
    #     db = DataFileExtended("./synth_data.cdb", "r")
    #     Trackers = db.tracker_from_db()
    #     return db.table_tracker.get(id=1)#Trackers[0]
    # tracker = retrieve_db()
    # print(tracker.Probability_Gains)
    # print(retrieve_db().Probability_Gains)
        # break
        #
        # state_dict = dict([[k, np.array([MultiKal.Filters[k].X[i] for i in MultiKal.Filters[k].X])] for k in MultiKal.Filters])
        # params = ["damping", "timeconst"]
        # # params = ["timeconst"]
        # # params = ["damping"]
        #
        # print("Real Params: ",[Generator.Model.Initial_KWArgs[o] for o in sorted(params)])
        # new_mod = Generator.Model#VariableSpeed(dim=2, damping=1., timeconst=2)
        # print("Easy Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
        # print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
        #
        # new_mod = VariableSpeed(dim=2, damping=1., timeconst=1)
        # print("Complex Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
        # print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
        # all_params.append(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
        # p_opt = new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params)
        # new_mod = VariableSpeed(dim=2, damping=p_opt[0], timeconst=p_opt[1])
        # MultiKal.Model = new_mod

    # SimStitch = DistanceStitcher(0.89)
    # SimStitch.add_PT_Tracks_from_Tracker(MultiKal.Filters)
    # SimStitch.stitch()
    #
    # from PenguTrack.DataFileExtended import DataFileExtended
    # db = DataFileExtended("./synth_data.cdb")
    # stiched_type = db.setMarkerType(name="Stitched", color="F0F0FF", mode=db.TYPE_Track)
    # db.deleteMarkers(type=stiched_type)
    # SimStitch.save_tracks_to_db("./synth_data.cdb", stiched_type)
