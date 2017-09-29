#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DataFileExtended.py

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


import peewee
import clickpoints

class DataFileExtended(clickpoints.DataFile):

    def __init__(self, *args, **kwargs):
        clickpoints.DataFile.__init__(self, *args, **kwargs)

        class Measurement(self.base_model):
            # full definition here - no need to use migrate
            marker = peewee.ForeignKeyField(self.table_marker, unique=True, related_name="measurement",
                                            on_delete='CASCADE')  # reference to frame and track via marker!
            log = peewee.FloatField(default=0)
            x = peewee.FloatField()
            y = peewee.FloatField(default=0)
            z = peewee.FloatField(default=0)

        if "measurement" not in self.db.get_tables():
            Measurement.create_table()  # important to respect unique constraint

        self.table_measurement = Measurement  # for consistency

    def setMeasurement(self, marker=None, log=None, x=None, y=None, z=None):
        assert not (marker is None), "Measurement must refer to a marker."
        try:
            item = self.table_measurement.get(marker=marker)
        except peewee.DoesNotExist:
            item = self.table_measurement()

        dictionary = dict(marker=marker, log=log, x=x, y=y, z=z)
        for key in dictionary:
            if dictionary[key] is not None:
                setattr(item, key, dictionary[key])
        item.save()
        return item

