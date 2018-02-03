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
from clickpoints.DataFile import noNoneDict, setFields, addFilter, PY3, packToDictList
import numpy as np
import PenguTrack
from PenguTrack.Filters import Filter as PT_Filter
from PenguTrack.Detectors import Measurement as PT_Measurement
from PenguTrack.Models import VariableSpeed
import ast
import scipy.stats as ss
try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        import io

def parse_dict(dictionary):
    out = dict(dictionary)
    for entry in dictionary:
        if type(dictionary[entry]) not in [str, int, float, tuple, list, dict, bool, type(None)]:
            if str(type(dictionary[entry])).count("scipy.stats"):
                dist = dictionary[entry]
                out[entry] = "ss."+str(dist.__class__).split("'")[1].split(".")[-1].replace(
                    "_frozen", "") + "(mean=%s,cov=%s)" % (dist.mean, dist.cov)
        else:
            out[entry]=str(dictionary[entry])
            # print(dictionary[entry])
            # print("----------dict------------")
    return str(out)


def parse_list(input):
    out = list(input)
    for i,entry in enumerate(input):
        if type(entry) not in [str, int, float, tuple, list, dict, bool, type(None)]:
            if str(type(entry)).count("scipy.stats"):
                out[i] = parse_dist(entry)
            # print(entry)
            # print("--------- list-------------")
    return str(out)


def parse_dist(dist):
    out = str(dist.__class__).split("'")[1].split(".")[-1].replace(
        "_frozen", "")
    # out += "(mean=%s,cov=%s)" % (dist.mean, dist.cov)
    return out


def parse_dist_dict(dist_dict):
    # print(dist_dict)
    return dict(mean=dist_dict.get("mean",None),
                cov=dist_dict.get("cov", None),
                lower=dist_dict.get("lower", None),
                upper=dist_dict.get("upper", None))


def parse_filter_class(filter_class):
    try:
        return str(filter_class).split("'")[1].split(".")[-1]
    except IndexError:
        return "Filter"


def parse_tracker_class(filter_class):
    try:
        return str(filter_class).split("'")[1].split(".")[-1]
    except IndexError:
        return "MultiFilter"

class MatrixField(peewee.BlobField):
    """ A database field, that """
    def db_value(self, value):
        if np.all(value == np.nan):
            value = np.array([None])
        value_b = np.array(value).astype(np.float64).tobytes()
        shape_b = np.array(np.array(value).shape).astype(np.int64).tobytes()
        len_shape_b = np.array(len(np.array(value).shape)).astype(np.int64).tobytes()
        value = len_shape_b+shape_b+value_b
        if not PY3:
            return value
        else:
            return peewee.binary_construct(value)

    def python_value(self, value):
        if not PY3:
            pass
        else:
            value = io.BytesIO(value)
        if value is not None and not len(value) == 0:
            l = np.frombuffer(value, dtype=np.int64, count=1, offset=0)
            if l == 0:
                return None
            shape = np.frombuffer(value, dtype=np.int64, count=l, offset=8)
            array = np.frombuffer(value, dtype=np.float64, count=int(np.prod(shape)*8), offset=8+l*8)
            return array.reshape(shape)
        else:
            return None

class ListField(peewee.TextField):
    """A database field, that"""
    def db_value(self, value):
        value=str(value)
        if PY3:
            return value
        return peewee.binary_construct(value)
    def python_value(self, value):
        value = self.parse2python(value)
        if not PY3:
            value = str(value)
            value = value.replace("array","np.array")
            return eval(value)
        return eval(io.BytesIO(value))

    def parse2python(self,value):
        return value

    def parse2db(self, value):
        # for v in value:
        #     #strings, numbers, tuples, lists, dicts, booleans, and None
        #     if type(v) not in [str, int, float, tuple, list, dict, bool, type(None)]:
        #         if type(v) == np.ndarray:
        #             pass
        #         if str(v).contains("PenguTrack.Filters"):
        #             v = str(v).split("PenguTrack.Filters")[1].split(" ")[0]
        #     else:
        #         v = str(v)
        # print(value)
        return value

class DictField(peewee.TextField):
    """A database field, that"""
    def db_value(self, value):
        value = self.parse2db(value)
        # value.pop("state_dist", None)
        # value.pop("meas_dist", None)
        value=str(value)
        if PY3:
            return value
        return peewee.binary_construct(value)
    def python_value(self, value):
        value = str(value)
        value = value.replace("array","np.array")
        out = eval(value)
        return dict([[v, eval(out[v])] for v in out])

    def parse2db(self, value):
        # for v in value:
        #     if type(value[v]) not in [list, str, int, float, np.ndarray]:
        #         print(type(value[v]), type(v))
        return value

class NumDictField(peewee.BlobField):
    """ A database field, that """
    def db_value(self, value):
        value = np.array([[v,value[v]] for v in value]).tobytes()
        if PY3:
            return value
        return peewee.binary_construct(value)

    def python_value(self, value):
        if not PY3:
            return dict([[v[0],v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1,2))])
        return dict([[v[0], v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1, 2))])

class DataFileExtended(clickpoints.DataFile):
    def __init__(self, *args, **kwargs):
        clickpoints.DataFile.__init__(self, *args, **kwargs)
        # Define ClickPoints Marker
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track)
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF")

        db = self

        """State Entry"""
        class State(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='state_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(default=0)
            state_vector = MatrixField()
            state_error = MatrixField()
        if "state" not in db.db.get_tables():
            State.create_table()
        self.table_state = State


        """Measurement Entry"""
        class Measurement(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='measurement_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(default=0)
            measurement_vector = MatrixField()
            measurement_error = MatrixField(null=True)
        if "measurement" not in db.db.get_tables():
            Measurement.create_table()
        self.table_measurement = Measurement

        """Prediction Entry"""
        class Prediction(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='prediction_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(default=0)
            prediction_vector = MatrixField()
            prediction_error = MatrixField()
        if "prediction" not in db.db.get_tables():
            Prediction.create_table()
        self.table_prediction = Prediction

        # """Marker Entry"""
        # class Marker(db.table_marker):
        #     state = peewee.ForeignKeyField(State, null=True, related_name='state_marker', on_delete='CASCADE')
        #     measurement = peewee.ForeignKeyField(Measurement, null=True, related_name='measurement_marker', on_delete='CASCADE')
        #     prediction = peewee.ForeignKeyField(State, null=True, related_name='prediction_marker', on_delete='CASCADE')
        # Marker.create_table()
        # self.table_marker = Marker


        """Tracker Entry"""
        class Tracker(db.base_model):
            # track = peewee.ForeignKeyField(self.table_track, related_name="tracker", on_delete="CASCADE")
            tracker_class = peewee.TextField()
            filter_class = peewee.TextField()
            # model = peewee.ForeignKeyField(self.table_model, related_name="tracker", on_delete='CASCADE')
            filter_threshold = peewee.IntegerField()
            log_probability_threshold = peewee.FloatField()
            measurement_probability_threshold = peewee.FloatField()
            assignment_probability_threshold = peewee.FloatField()
            filter_args = ListField()
            filter_kwargs = DictField()

        if "tracker" not in db.db.get_tables():
            Tracker.create_table()
        self.table_tracker = Tracker

        """Dist Entry"""
        class Distribution(db.base_model):
            name = peewee.TextField()
            mean = MatrixField(null=True,default=None)
            cov = MatrixField(null=True,default=None)
            lower = MatrixField(null=True,default=None)
            upper = MatrixField(null=True,default=None)
        if "distribution" not in db.db.get_tables():
            Distribution.create_table()
        self.table_distribution = Distribution

        """Model Entry"""
        class Model(self.base_model):
            state_dim = peewee.IntegerField(default=1)
            control_dim = peewee.IntegerField(default=1)
            meas_dim = peewee.IntegerField(default=1)
            evolution_dim = peewee.IntegerField(default=1)

            # opt_params = peewee.TextField(default="")
            # opt_params_shape = ListField()
            # opt_params_borders = ListField()

            state_matrix = MatrixField()
            control_matrix = MatrixField()
            measurement_matrix = MatrixField()
            evolution_matrix = MatrixField()
        if "model" not in db.db.get_tables():
            Model.create_table()
        self.table_model = Model

        """Filter Entry"""
        class Filter(db.base_model):
            track = peewee.ForeignKeyField(db.table_track,
                                           unique=True,
                                           related_name='filter_track', on_delete='CASCADE')
            tracker = peewee.ForeignKeyField(db.table_tracker,
                                             related_name='filter_tracker', on_delete='CASCADE')
            measurement_distribution = peewee.ForeignKeyField(db.table_distribution,
                                                              unique=True,
                                                              related_name='filter_meas_dist',
                                                              on_delete='CASCADE')
            state_distribution = peewee.ForeignKeyField(db.table_distribution,
                                                        unique=True,
                                                        related_name='filter_state_dist',
                                                        on_delete='CASCADE')
            model = peewee.ForeignKeyField(db.table_model,
                                           unique=True,
                                           related_name='filter_model',
                                           on_delete='CASCADE')
        if "filter" not in db.db.get_tables():
            Filter.create_table()
        self.table_filter = Filter

        """Probability Gain Entry"""
        class Probability_Gain(db.base_model):
            image = peewee.ForeignKeyField(db.table_image, related_name="probability_gains", on_delete='CASCADE')
            tracker = peewee.ForeignKeyField(db.table_tracker, related_name="tracker_prob_gain", on_delete='CASCADE')
            probability_gain = MatrixField()
            probability_gain_dict = NumDictField()
        if "probability_gain" not in db.db.get_tables():
            Probability_Gain.create_table()
        self.table_probability_gain = Probability_Gain


    def setState(self, id=None, marker=None, log_prob=None, state_vector=None, state_error=None):
        dictionary = dict(id=id, marker=marker)
        if id is not None:
            try:
                item = self.table_state.get(**dictionary)
            except peewee.DoesNotExist:
                item = self.table_state()
        else:
            item = self.table_state()
        dictionary.update(dict(log_prob=log_prob,
                               state_vector=state_vector,
                               state_error=state_error))
        setFields(item, dictionary)
        item.save()
        return item


    def setMeasurement(self, id=None, marker=None, log_prob=None, measurement_vector=None, measurement_error=None):
        dictionary = dict(id=id, marker=marker)
        if id is not None:
            try:
                item = self.table_measurement.get(**dictionary)
            except peewee.DoesNotExist:
                item = self.table_measurement()
        else:
            item = self.table_measurement()
        dictionary.update(dict(log_prob=log_prob,
                               measurement_vector=measurement_vector,
                               measurement_error=measurement_error))
        setFields(item, dictionary)
        item.save()
        return item


    def setPrediction(self, id=None, marker=None, log_prob=None, prediction_vector=None, prediction_error=None):
        dictionary = dict(id=id, marker=marker)
        if id is not None:
            try:
                item = self.table_prediction.get(**dictionary)
            except peewee.DoesNotExist:
                item = self.table_prediction()
        else:
            item = self.table_prediction()
        dictionary.update(dict(log_prob=log_prob,
                               prediction_vector=prediction_vector,
                               prediction_error=prediction_error))
        setFields(item, dictionary)
        item.save()
        return item


    # def setMarker(self, state=None, measurement=None, prediction=None, **kwargs):
    #     item = super(DataFileExtended, self).setMarker(**kwargs)
    #     kwargs.update(dict(state=state, measurement=measurement, prediction=prediction))
    #     setFields(item, **kwargs)
    #     return item


    def setTracker(self,tracker_class="", filter_class="",# model=None,
               filter_threshold=3,log_probability_threshold=0.,
               measurement_probability_threshold=0.0,
               assignment_probability_threshold=0.0,
               filter_args=[], filter_kwargs={}, id=None):
        dictionary = dict(id=id)
        if id is not None:
            try:
                item = self.table_tracker.get(**dictionary)
            except peewee.DoesNotExist:
                item = self.table_tracker()
        else:
            item = self.table_tracker()
        dictionary.update(dict(tracker_class=tracker_class,
                               filter_class=filter_class,
                               filter_threshold=filter_threshold,
                               log_probability_threshold=log_probability_threshold,
                               measurement_probability_threshold=measurement_probability_threshold,
                               assignment_probability_threshold=assignment_probability_threshold,
                               filter_args=parse_list(filter_args),
                               filter_kwargs=parse_dict(filter_kwargs)))
        setFields(item, dictionary)
        item.save()
        return item


    def getTrackers(self, tracker_class="", filter_class="",
                    filter_threshold=3,log_probability_threshold=0.,
                    measurement_probability_threshold=0.0,
                    assignment_probability_threshold=0.0,
                    id=None):
        query = self.table_tracker.select()
        query = addFilter(query, id, self.table_tracker.id)
        query = addFilter(query, tracker_class, self.table_tracker.tracker_class)
        query = addFilter(query, filter_class, self.table_tracker.filter_class)
        query = addFilter(query, filter_threshold, self.table_tracker.filter_threshold)
        query = addFilter(query, log_probability_threshold, self.table_tracker.log_probability_threshold)
        query = addFilter(query, measurement_probability_threshold, self.table_tracker.measurement_probability_threshold)
        query = addFilter(query, assignment_probability_threshold, self.table_tracker.assignment_probability_threshold)
        return query


    def setDistribution(self, name=None, mean=None, cov=None, lower=None, upper=None, id=None):
        dictionary = dict(id=id)
        if id is not None:
            try:
                item = self.table_distribution.get(**dictionary)
            except peewee.DoesNotExist:
                item = self.table_distribution()
        else:
            item = self.table_distribution()
        dictionary.update(dict(name=name,
                               mean=mean,
                               cov=cov,
                               lower=lower,
                               upper=upper))
        setFields(item, dictionary)
        item.save()
        return item

    def setModel(self, id=None, state_dim=1, control_dim=1, meas_dim=1, evolution_dim=1,
                 state_matrix=None, control_matrix=None, measurement_matrix=None, evolution_matrix=None):

        dictionary = dict(id=id)
        try:
            item = self.table_model.get(**dictionary)
        except peewee.DoesNotExist:
            item = self.table_model()

        dictionary.update(dict(state_dim=state_dim, control_dim=control_dim,
                               meas_dim=meas_dim, evolution_dim=evolution_dim))
        if state_matrix is None:
            state_matrix = np.identity(state_dim)
        if control_matrix is None:
            control_matrix = np.identity(max(state_dim, control_dim))[:state_dim, :control_dim]
        if measurement_matrix is None:
            measurement_matrix = np.identity(max(state_dim, meas_dim))[: meas_dim, :state_dim]
        if evolution_matrix is None:
            evolution_matrix = np.identity(max(evolution_dim, state_dim))[:state_dim, :evolution_dim]
        dictionary.update(dict(state_matrix=state_matrix, control_matrix=control_matrix,
                               measurement_matrix=measurement_matrix, evolution_matrix=evolution_matrix))
        setFields(item, dictionary)
        item.save()
        return item

    def setFilter(self, track=None, tracker=None,
                  measurement_distribution=None,
                  state_distribution=None,
                  model=None):
        dictionary = dict(track=track)
        try:
            item = self.table_filter.get(**dictionary)
        except peewee.DoesNotExist:
            item = self.table_filter()
            # except peewee.OperationalError:
            #     model = self.table_model()

        dictionary.update(dict(track=track,
                               tracker=tracker,
                               measurement_distribution=measurement_distribution,
                               state_distribution=state_distribution,
                               model=model))
        setFields(item, dictionary)
        item.save()
        return item

    def getFilter(self, track=None):
        query = self.table_filter.select()
        query = addFilter(query, track, self.table_track)
        # query = addFilter(query, id, self.table_filter.id)
        if len(query) > 0:
            return query[0]
        else:
            return None


    def setProbabilityGain(self, id=None, image=None, tracker=None,
                           probability_gain=None, probability_gain_dict=None):
        dictionary = dict(id=id, image=image, tracker=tracker)
        if id is not None:
            try:
                item = self.table_probability_gain.get(**noNoneDict(**dictionary))
            except peewee.DoesNotExist:
                item = self.table_probability_gain()
        else:
            item = self.table_probability_gain()
        dictionary.update(dict(image=image,
                               tracker=tracker,
                               probability_gain=probability_gain,
                               probability_gain_dict=probability_gain_dict))
        setFields(item, dictionary)
        item.save()
        return item

    def setMeasurements(self, marker=None, log=None, x=None, y=None, z=None):
        data = packToDictList(self.table_measurement, marker=marker, log=log,x=x, y=y, z=z)
        return self.saveUpsertMany(self.table_measurement, data)

    def deletetOld(self):
        # Define ClickPoints Marker
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.deleteMarkers(type=self.detection_marker_type)
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track)
        self.deleteMarkers(type=self.track_marker_type)
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF")
        self.deleteMarkers(type=self.prediction_marker_type)
        # Delete Old Tracks
        self.deleteTracks(type=self.track_marker_type)


    def init_tracker(self, model, tracker):
        model_item = self.setModel(state_dim=model.State_dim, control_dim=model.Control_dim,
                      meas_dim=model.Meas_dim, evolution_dim=model.Evolution_dim,
                      state_matrix=model.State_Matrix, control_matrix=model.Control_Matrix,
                      measurement_matrix=model.Measurement_Matrix, evolution_matrix=model.Evolution_Matrix)
        tracker_item = self.setTracker(tracker_class=tracker.__class__,
                                       filter_class=tracker.Filter_Class,
                                       # model=model_item,
                        filter_threshold=tracker.FilterThreshold,
                        log_probability_threshold=tracker.LogProbabilityThreshold,
                        measurement_probability_threshold=tracker.MeasurementProbabilityThreshold,
                        filter_args=tracker.filter_args,
                        filter_kwargs=tracker.filter_kwargs)
        return model_item, tracker_item

    def write_to_DB(self, Tracker, image, i=None, text=None, cam_values=False, db_tracker=None, db_model=None):
        if text is None:
            set_text = True
        else:
            set_text = False

        if i is None:
            i = image.sort_index


        with self.db.atomic() as transaction:
            markerset = []
            measurement_set = []
            pred_markerset = []
            # Get Tracks from Filters
            for k in Tracker.Filters.keys():
                x = y = np.nan
                if cam_values:
                    x = meas.PositionX_Cam
                    y = meas.PositionY_Cam
                else:
                    x = meas.PositionX
                    y = meas.PositionY

                if self.getTrack(id=100 + k):
                    db_track = self.getTrack(id=100 + k)
                else:
                    db_track = self.setTrack(self.track_marker_type, id=100 + k)
                if len(db_track.filter_track)>0:
                    print('Setting Track(%s)-Marker at %s, %s' % ((100 + k), x, y))
                    db_filter = db_track.filter_track[0]
                else:
                    db_dist_s = self.setDistribution(name=parse_dist(Tracker.Filters[k].State_Distribution),
                                                     **parse_dist_dict(Tracker.Filters[k].State_Distribution.__dict__))
                    db_dist_m = self.setDistribution(name=parse_dist(Tracker.Filters[k].Measurement_Distribution),
                                                     **parse_dist_dict(Tracker.Filters[k].Measurement_Distribution.__dict__))
                    db_filter_model = self.setModel(state_dim=db_model.state_dim, control_dim=db_model.control_dim,
                                                    meas_dim=db_model.meas_dim, evolution_dim=db_model.evolution_dim,
                                                    state_matrix=db_model.state_matrix, control_matrix=db_model.control_matrix,
                                                    measurement_matrix=db_model.measurement_matrix, evolution_matrix=db_model.evolution_matrix)
                    db_filter = self.setFilter(model=db_filter_model, tracker=db_tracker,
                                               # id=100 + k,
                                               track=db_track,
                                               measurement_distribution=db_dist_m,
                                               state_distribution=db_dist_s)

        self.write_to_DB(Tracker, image, i=i, text=text, cam_values=True)



    def write_to_DB_cam(self, Tracker, image, i=None, text=None, db_tracker=None, db_model=None):
        return self.write_to_DB(Tracker,image, i=i, text=text, cam_values=True,db_tracker=db_tracker, db_model=db_model)

    def PT_tracks_from_db(self, type, get_measurements=True):
        Tracks = {}
        if self.getTracks(type=type)[0].markers[0].measurement is not None and get_measurements:
            for track in self.getTracks(type=type):
                Tracks.update({track.id: PT_Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    meas = PT_Measurement(1., [m.measurement[0].x,
                                            m.measurement[0].y,
                                            m.measurement[0].z])
                    Tracks[track.id].update(z=meas, i=m.image.sort_index)
        else:
            for track in self.getTracks(type=type):
                Tracks.update({track.id: PT_Filter(VariableSpeed(dim=3))})
                for m in track.markers:
                    x, y = m.correctedXY()
                    meas = PT_Measurement(1., [x,
                                            y,
                                            0])
                    Tracks[track.id].update(z=meas, i=m.image.sort_index)
        return Tracks


    def model_from_db(self, db_model):
        out=PenguTrack.Models.Model(state_dim=db_model.state_dim,
                                control_dim=db_model.control_dim,
                                meas_dim=db_model.meas_dim,
                                evo_dim=db_model.evolution_dim)
        out.State_Matrix = db_model.state_matrix.reshape((db_model.state_dim,db_model.state_dim))
        out.Control_Matrix = db_model.control_matrix.reshape((db_model.state_dim,db_model.control_dim))
        out.Measurement_Matrix = db_model.measurement_matrix.reshape((db_model.meas_dim,db_model.state_dim))
        out.Evolution_Matrix = db_model.evolution_matrix.reshape((db_model.state_dim,db_model.evolution_dim))
        return out


    def dist_from_db(self, db_dist):
        dist_dict = dict(mean=db_dist.mean,
                         cov=db_dist.cov,
                         lower=db_dist.lower,
                         upper=db_dist.upper)
        dist_class = eval("ss."+db_dist.name)
        return dist_class(**noNoneDict(**dist_dict))


    def filter_from_db(self, db_filter, filter_class=None, filter_args=None, filter_kwargs=None):
        model = self.model_from_db(db_filter.model)
        meas_dist = self.dist_from_db(db_filter.measurement_distribution)
        state_dist = self.dist_from_db(db_filter.state_distribution)
        if filter_class is None or filter_args is None or filter_kwargs is None:
            filter = PenguTrack.Filters.Filter(model, meas_dist=meas_dist, state_dist=state_dist)
        else:
            filter = filter_class(model, *filter_args, **filter_kwargs)

        filter.X.update(dict([[state.marker.image.sort_index, state.state_vector] for state in db_filter.filter_states]))
        filter.Predicted_X.update(dict([[pred.marker.image.sort_index, pred.prediction_vector] for pred in db_filter.filter_predictions]))
        filter.Measurements.update(dict([[meas.marker.image.sort_index, meas.measurement_vector] for meas in db_filter.filter_measurements]))

        filter.X_error.update(dict([[state.marker.image.sort_index, state.state_error] for state in db_filter.filter_states]))
        filter.Predicted_X_error.update(dict([[pred.marker.image.sort_index, pred.prediction_error] for pred in db_filter.filter_predictions]))

        return db_filter.id, filter


    def tracker_from_db(self):
        db_trackers = self.getTrackers()
        out = []
        for db_tracker in db_trackers:
            tracker_class = eval("PenguTrack.Filters."+db_tracker.tracker_class)
            filter_class = eval("PenguTrack.Filters."+db_tracker.filter_class)
            model = self.model_from_db(db_tracker.model)
            # print(db_tracker.filter_args)
            # print(db_tracker.filter_kwargs)
            tracker = tracker_class(filter_class, model, *db_tracker.filter_args, **db_tracker.filter_kwargs)
            tracker.Filters.update(dict([self.filter_from_db(db_filter, filter_class=filter_class) for db_filter in db_tracker.tracks]))
            tracker.LogProbabilityThreshold = db_tracker.log_probability_threshold
            tracker.FilterThreshold = db_tracker.filter_threshold
            tracker.AssignmentProbabilityThreshold = db_tracker.assignment_probability_threshold
            tracker.MeasurementProbabilityThreshold = db_tracker.measurement_probability_threshold
            tracker.Probability_Gain.update(dict([[pg.image.sort_index, pg.gain.reshape((len(pg.gain_dict), -1))] for pg in db_tracker.tracker_probability_gains]))
            tracker.Probability_Gain_Dicts.update(dict([[pg.image.sort_index, pg.gain_dict] for pg in db_tracker.tracker_probability_gains]))
            out.append(tracker)
        return out