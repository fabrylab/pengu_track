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

from PenguTrack.Tools.SyntheticDataGenerator import SyntheticDataGenerator

if __name__ == '__main__':

    def track(filters, gen):
        # Extended Clickpoints Database for usage with pengutack
        from PenguTrack.DataFileExtended import DataFileExtended
        # Open ClickPoints Database
        db = DataFileExtended("./synth_data.cdb", "w")

        gt_type = db.setMarkerType(name="Ground_Truth", mode=db.TYPE_Track, color="#FFFFFF")
        db_models = {}
        db_trackers = {}
        db_track_marker_types = {}
        for filter in filters:
            db_model, db_tracker = db.init_tracker(filter.Model, filter)
            db_models[filter.name()] = db_model
            db_trackers[filter.name()] = db_tracker
            db_track_marker_types[filter.name()] = db.setMarkerType("%s_Track_Marker"%filter.name(),
                                                                    color="#00FF00",
                                                                    mode=db.TYPE_Track)

        # Start Iteration over Images
        print('Starting Iteration')
        for i in range(11):
            image = db.setImage(filename="%s.png"%i, frame=i)

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
                for filter in filters:
                    # Prediction step, without applied control(vector of zeros)
                    filter.predict(i=i)
                    # Update Filter with new Detections
                    filter.update(z=Positions, i=i, verbose=False)
                    # Write everything to a DataBase
                    db.track_marker_type = db_track_marker_types[filter.name()]
                    db.write_to_DB(filter, image, i=i, debug_mode=0b00000,
                                   db_tracker=db_trackers[filter.name()],
                                   db_model=db_models[filter.name()])


        print('done with Tracking')
        return filters

    all_params = []

    from PenguTrack.Trackers import VariableSpeedTracker, GreedyVariableSpeedTracker, NetworkVariableSpeedTracker

    MultiKal = VariableSpeedTracker(q=1., no_dist=False, prob_update=False)
    MultiKal.LogProbabilityThreshold = -10.
    MultiKal2 = GreedyVariableSpeedTracker(q=1., no_dist=False, prob_update=False)
    MultiKal2.LogProbabilityThreshold = -10.
    MultiKal3 = NetworkVariableSpeedTracker(q=1., no_dist=False, prob_update=False)
    MultiKal3.LogProbabilityThreshold = -10.
    Kals = [MultiKal, MultiKal2, MultiKal3]
    # MultiKal.LogProbabilityThreshold = -3.
    # Physical Model (used for predictions)
    from PenguTrack.Models import VariableSpeed
    from PenguTrack.DataFileExtended import DataFileExtended

    mult = 1024
    Generator = SyntheticDataGenerator(int(1.1*mult), 0.5*mult, 1., VariableSpeed(dim=2, timeconst=0.5, damping=1.), loose=True)

    Kals = track(Kals, Generator)


    # db = DataFileExtended("./synth_data.cdb", "r")
    # Tracker = db.tracker_from_db()[0]
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
