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
    def __init__(self, object_number, position_variation, speed_variation, model):
        self.N = int(object_number)
        self.R = float(position_variation)
        self.Q = float(speed_variation)
        self.Model = model
        vec = np.random.rand(self.N, self.Model.State_dim, 1)
        vec [:,::2] *= self.R
        vec [1:,::2] *= self.Q
        self.Objects = {0: vec}

        for i in range(100):
            self.step()

    def step(self):
        i = max(self.Objects.keys())
        positions = self.Objects[i]
        self.Objects[i+1] = np.array([self.Model.predict(self.Model.evolute(self.Q*np.random.randn(self.Model.Evolution_dim, 1),
                                                                            pos),
                                                         np.zeros((self.Model.Control_dim, 1))) for pos in positions])
        return [Measurement(1.0, self.Model.measure(pos)) for pos in self.Objects[i+1]]

if __name__ == '__main__':

    def track(filter, gen):
        # Extended Clickpoints Database for usage with pengutack
        from PenguTrack.DataFileExtended import DataFileExtended
        # Open ClickPoints Database
        # db = DataFileExtended("./synth_data.cdb", "w")

        # Start Iteration over Images
        print('Starting Iteration')
        for i in range(30):
            # image = db.setImage(filename="%s.png"%i, frame=i)
            # Prediction step, without applied control(vector of zeros)
            filter.predict(i=i)

            # Detection step
            Positions = gen.step()
            print("Found %s Objects!"%len(Positions))

            if len(Positions)>0:
                # Update Filter with new Detections
                filter.update(z=Positions, i=i)
                # Write everything to a DataBase
                # db.write_to_DB(Filter, image)

        print('done with Tracking')
        return filter

    all_params = []

    for i in range(20):
        from PenguTrack.Trackers import VariableSpeedTracker

        MultiKal = VariableSpeedTracker()
        # Physical Model (used for predictions)
        from PenguTrack.Models import VariableSpeed

        Generator = SyntheticDataGenerator(10, 10., 1., VariableSpeed(dim=2, timeconst=0.5, damping=1.))

        MultiKal = track(MultiKal, Generator)

        state_dict = dict([[k, np.array([MultiKal.Filters[k].X[i] for i in MultiKal.Filters[k].X])] for k in MultiKal.Filters])
        params = ["damping", "timeconst"]
        # params = ["timeconst"]
        # params = ["damping"]

        print("Real Params: ",[Generator.Model.Initial_KWArgs[o] for o in sorted(params)])
        new_mod = Generator.Model#VariableSpeed(dim=2, damping=1., timeconst=2)
        print("Easy Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
        print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))

        new_mod = VariableSpeed(dim=2, damping=1., timeconst=1)
        print("Complex Start: ", [new_mod.Initial_KWArgs[o] for o in sorted(params)])
        print(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
        all_params.append(new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params))
        p_opt = new_mod.__unflatparams__(new_mod.optimize_mult([s for s in state_dict.values() if len(s)>2], params=params)[0], params=params)
        new_mod = VariableSpeed(dim=2, damping=p_opt[0], timeconst=p_opt[1])
        MultiKal.Model = new_mod



