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
from clickpoints.DataFile import noNoneDict, setFields, addFilter, PY3, packToDictList, dict_factory
import numpy as np
import PenguTrack
from PenguTrack.Filters import Filter as PT_Filter
import PenguTrack.Filters
from PenguTrack.Detectors import Measurement as PT_Measurement
from PenguTrack.Models import VariableSpeed
import ast
import scipy.stats as ss
import json
import inspect

try:
    from cStringIO import StringIO
except ImportError:
    try:
        from StringIO import StringIO
    except ImportError:
        import io

def parse_list(x):
    if listable(x):
        return [parse(xx) for xx in x]
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def parse_dict(x):
    if listable(x.keys()):
        return dict([[xx,parse(x[xx])] for xx in x])
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def parse(x):
    if type(x) in [str, float, int, bool]:
        return x
    elif type(x) == np.ndarray:
        return float(x)
    elif type(x) == list:
        return parse_list(x)
    elif type(x) == dict:
        return parse_dict(x)
    elif listable(x):
        return [parse(xx) for xx in x]
    elif x is None:
        return None
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def listable(x):
    try:
        list(x)
    except TypeError:
        return False
    else:
        return len(list(x))>0

def is_dist(x):
    return str(type(x)).count("scipy.stats")


def parse_dist(dist):
    return [parse_dist_name(dist), parse_dist_dict(dist.__dict__)]

def parse_dist_name(dist):
    return str(dist.__class__).split("'")[1].split(".")[-1].replace("_frozen", "")


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
            return super(MatrixField, self).db_value(value)
        else:
            return super(MatrixField, self).db_value(value)#peewee.binary_construct(value)

    def python_value(self, value):
        value = super(MatrixField, self).python_value(value)
        # if not PY3:
        #     pass
        # else:
        #     value = io.BytesIO(value)
        if value is not None and not len(value) == 0:
            l = np.frombuffer(value, dtype=np.int64, count=1, offset=0)
            if l == 0:
                return None
            shape = np.frombuffer(value, dtype=np.int64, count=int(l), offset=8)
            array = np.frombuffer(value, dtype=np.float64, count=int(np.prod(shape)), offset=8+int(l)*8)
            return array.reshape(shape)
        else:
            return None

class ListField(peewee.TextField):
    """A database field, that"""
    def db_value(self, value):
        value=json.dumps(parse_list(value))
        if PY3:
            return super(ListField, self).db_value(value)
        return super(ListField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        return json.loads(super(ListField, self).python_value(value))


class DictField(peewee.TextField):
    """A database field, that"""
    def db_value(self, value):
        value = json.dumps(parse_dict(value))
        if PY3:
            return super(DictField, self).db_value(value)
        return super(DictField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        return json.loads(super(DictField, self).python_value(value))


class NumDictField(peewee.BlobField):
    """ A database field, that """
    def db_value(self, value):
        value = np.array([[v,value[v]] for v in value]).tobytes()
        if PY3:
            return super(NumDictField, self).db_value(value)
        return super(NumDictField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        value=super(NumDictField, self).python_value(value)
        if not PY3:
            return dict([[v[0],v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1,2))])
        return dict([[v[0], v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1, 2))])

class DataFileExtended(clickpoints.DataFile):
    def __init__(self, *args, **kwargs):
        clickpoints.DataFile.__init__(self, *args, **kwargs)
        # Define ClickPoints Marker
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track,
                                                    style='{"shape":"circle","transform": "image","color":"hsv"}')
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF",
                                                         style='{"shape": "circle", "transform": "image"}')

        db = self

        self._current_ext_version = 1
        ext_version = self._CheckExtVersion()


        """State Entry"""
        class State(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='state_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(null=True, default=0)
            state_vector = MatrixField()
            state_error = MatrixField(null=True)
        if "state" not in db.db.get_tables():
            State.create_table()
        self.table_state = State


        """Measurement Entry"""
        class Measurement(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='measurement_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(null=True, default=0)
            measurement_vector = MatrixField()
            measurement_error = MatrixField(null=True)
        if "measurement" not in db.db.get_tables():
            Measurement.create_table()

        self.table_measurement = Measurement

        """Prediction Entry"""
        class Prediction(db.base_model):
            marker = peewee.ForeignKeyField(db.table_marker, unique=True, related_name='prediction_marker', on_delete='CASCADE')
            log_prob = peewee.FloatField(null=True, default=0)
            prediction_vector = MatrixField()
            prediction_error = MatrixField(null=True)
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


        """Tracker Entry"""
        class Tracker(db.base_model):
            # track = peewee.ForeignKeyField(self.table_track, related_name="tracker", on_delete="CASCADE")
            tracker_class = peewee.TextField()
            filter_class = peewee.TextField()
            # model = peewee.ForeignKeyField(self.table_model, related_name="tracker", on_delete='CASCADE')
            model = peewee.ForeignKeyField(db.table_model,
                                           related_name='tracker_model',
                                           on_delete='CASCADE')
            filter_threshold = peewee.IntegerField()
            log_probability_threshold = peewee.FloatField()
            measurement_probability_threshold = peewee.FloatField()
            assignment_probability_threshold = peewee.FloatField()
            filter_args = ListField()
            filter_kwargs = DictField()

            def __getattribute__(self, item):
                if item == "tracks":
                    return .select().join(self.table_filter).where(self.table_filter.tracker_id==self.id)
                return super(Tracker, self).__getattribute__(item)

        if "tracker" not in db.db.get_tables():
            Tracker.create_table()
        self.table_tracker = Tracker


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


    def setTracker(self,tracker_class="", filter_class="", model=None,
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

        for k in filter_kwargs:
            if is_dist(filter_kwargs[k]):
                filter_kwargs[k] = parse_dist(filter_kwargs[k])

        dictionary.update(dict(tracker_class=tracker_class,
                               filter_class=filter_class,
                               model=model,
                               filter_threshold=filter_threshold,
                               log_probability_threshold=log_probability_threshold,
                               measurement_probability_threshold=measurement_probability_threshold,
                               assignment_probability_threshold=assignment_probability_threshold,
                               filter_args=parse_list(filter_args),
                               filter_kwargs=parse_dict(filter_kwargs)))
        setFields(item, dictionary)
        item.save()
        return item


    def getTrackers(self, tracker_class=None, filter_class=None,
                    filter_threshold=None,log_probability_threshold=None,
                    measurement_probability_threshold=None,
                    assignment_probability_threshold=None,
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

    def setMeasurements(self, marker=None, log_prob=None, measurement_vector=None, measurement_error=None):
        data = packToDictList(self.table_measurement, marker=marker, measurement_vector=measurement_vector,
                              measurement_error=measurement_error)
        return self.saveUpsertMany(self.table_measurement, data)

    def setPredictions(self, marker=None, log_prob=None, prediction_vector=None, prediction_error=None):
        data = packToDictList(self.table_prediction, marker=marker, log_prob=log_prob,
                              prediction_vector=prediction_vector,
                              prediction_error=prediction_error)
        return self.saveUpsertMany(self.table_prediction, data)

    def setStates(self, marker=None, log_prob=None, state_vector=None, state_error=None):
        data = packToDictList(self.table_state, marker=marker, log_prob=log_prob,
                              state_vector=state_vector,
                              state_error=state_error)
        return self.saveUpsertMany(self.table_state, data)

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
        tracker_item = self.setTracker(tracker_class=parse_tracker_class(tracker.__class__),
                                       filter_class=parse_filter_class(tracker.Filter_Class),
                                       model=model_item,
                        filter_threshold=tracker.FilterThreshold,
                        log_probability_threshold=tracker.LogProbabilityThreshold,
                        measurement_probability_threshold=tracker.MeasurementProbabilityThreshold,
                        filter_args=tracker.filter_args,
                        filter_kwargs=tracker.filter_kwargs)
        return model_item, tracker_item

    def write_to_DB(self, Tracker, image, i=None, text=None, cam_values=False, db_tracker=None, db_model=None,
                    debug_mode=0b111):
        if text is None:
            set_text = True
        else:
            set_text = False

        if i is None:
            i = image.sort_index

        with self.db.atomic() as transaction:
            track_markerset = []
            stateset = []
            measurement_markerset = []
            measurementset = []
            prediction_markerset = []
            predictionset = []
            # Get Tracks from Filters
            for k in Tracker.Filters.keys():
                x = y = np.nan
                prob = Tracker.Filters[k].log_prob(keys=[i], update=Tracker.Filters[k].ProbUpdate)

                if self.getTrack(id=100 + k):
                    db_track = self.getTrack(id=100 + k)
                else:
                    db_track = self.setTrack(self.track_marker_type, id=100 + k)
                if len(db_track.filter_track)>0:
                    new = ""
                    db_filter = db_track.filter_track[0]
                else:
                    new = "new "
                    db_dist_s = self.setDistribution(name=parse_dist_name(Tracker.Filters[k].State_Distribution),
                                                     **parse_dist_dict(Tracker.Filters[k].State_Distribution.__dict__))
                    db_dist_m = self.setDistribution(name=parse_dist_name(Tracker.Filters[k].Measurement_Distribution),
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

                # Case 1: we tracked something in this filter
                if i in Tracker.Filters[k].Measurements.keys() and (debug_mode&0b001):
                    meas = Tracker.Filters[k].Measurements[i]
                    state = Tracker.Filters[k].X[i]
                    try:
                        state_err = Tracker.Model.measure(Tracker.Filters[k].X_error[i])
                    except KeyError:
                        state_err = np.zeros((len(state), len(state)))
                    i_x = Tracker.Model.Measured_Variables.index("PositionX")
                    i_y = Tracker.Model.Measured_Variables.index("PositionY")
                    state_err = (state_err[i_x,i_x]**2+state_err[i_y,i_y]**2)**0.5

                    if cam_values:
                        x = meas.PositionX_Cam
                        y = meas.PositionY_Cam
                    else:
                        x = Tracker.Model.measure(state)[i_x]
                        y = Tracker.Model.measure(state)[i_y]
                    print('Setting %sTrack(%s)-Marker at %s, %s' % (new, (100 + k), x, y))
                    if set_text:
                        text = 'Track %s, Prob %.2f' % ((100 + k), prob)
                    track_markerset.append(dict(image=image, type=self.track_marker_type, track=100 + k, x=y, y=x,
                                                text=text,
                                                style='{"scale":%.2f}'%(2*state_err)))
                    stateset.append(dict(log_prob=prob,
                                         state_vector=Tracker.Filters[k].X[i],
                                         state_error=Tracker.Filters[k].X_error.get(i, None)))
                    # track_marker = self.setMarker(image=image, type=self.track_marker_type, track=100 + k, x=y, y=x,
                    #                               text=text)
                    # db_state = self.setState(marker=track_marker,
                    #                          log_prob=prob,
                    #                          state_vector=Tracker.Filters[k].X[i],
                    #                          state_error=Tracker.Filters[k].X_error.get(i, None))

                # Case 2: we want to see the prediction markers
                if i in Tracker.Filters[k].Predicted_X.keys() and (debug_mode&0b010):
                    prediction = Tracker.Model.measure(Tracker.Filters[k].Predicted_X[i])
                    try:
                        prediction_err = Tracker.Model.measure(Tracker.Filters[k].Predicted_X_error[i])
                    except KeyError:
                        prediction_err = np.zeros((len(prediction), len(prediction)))
                    i_x = Tracker.Model.Measured_Variables.index("PositionX")
                    i_y = Tracker.Model.Measured_Variables.index("PositionY")
                    pred_x = prediction[i_x]
                    pred_y = prediction[i_y]

                    pred_err = (prediction_err[i_x,i_x]**2+prediction_err[i_y,i_y]**2)**0.5

                    # pred_marker = self.setMarker(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                    #                            type=self.prediction_marker_type)
                    prediction_markerset.append(dict(image=image, x=pred_y, y=pred_x, text="Track %s" % (100 + k),
                                                     type=self.prediction_marker_type,
                                                     style='{"scale":%.2f}'%(2*pred_err)))
                    predictionset.append(dict(log_prob=prob,
                                              prediction_vector=Tracker.Filters[k].Predicted_X[i],
                                              prediction_error=Tracker.Filters[k].Predicted_X_error.get(i, None)))
                    # db_pred = self.setPrediction(marker=pred_marker,
                    #                              log_prob=prob,
                    #                              prediction_vector=Tracker.Filters[k].Predicted_X[i],
                    #                              prediction_error=Tracker.Filters[k].Predicted_X_error.get(i, None))


                # Case 3: we want to see the measurement markers
                if i in Tracker.Filters[k].Measurements.keys() and (debug_mode&0b100):
                    meas = Tracker.Filters[k].Measurements[i]
                    meas_x = meas.PositionX
                    meas_y = meas.PositionY
                    # meas_marker = self.setMarker(image=image, x=meas_y, y=meas_x, text="Track %s" % (100 + k),
                    #                            type=self.detection_marker_type)
                    measurement_markerset.append(dict(image=image, x=meas_y, y=meas_x, text="Track %s" % (100 + k),
                                                      type=self.detection_marker_type))
                    measurementset.append(dict(log_prob=prob,
                                               measurement_vector=np.array([meas_x, meas_y])))
                    # db_meas = self.setMeasurement(marker=meas_marker,
                    #                               log_prob=prob,
                    #                               measurement_vector=np.array([meas_x, meas_y]))#,
                    #                               measurement_error=db_filter.measurement_distribution.cov)

        try:
            prob_gain = Tracker.Probability_Gain[i]
            prob_gain_dict = Tracker.Probability_Gain_Dicts[i]
            self.setProbabilityGain(image=image,tracker=db_tracker,
                                    probability_gain=prob_gain,
                                    probability_gain_dict=prob_gain_dict)
        except KeyError:
            pass

        if (debug_mode&0b001):
            self.setMarkers(image=[m["image"] for m in track_markerset],
                            type=[m["type"] for m in track_markerset],
                            track=[m["track"] for m in track_markerset],
                            x=[m["x"] for m in track_markerset],
                            y=[m["y"] for m in track_markerset],
                            text=[m["text"] for m in track_markerset],
                            style=[m["style"] for m in track_markerset])
            state_markers = self.getMarkers(image=image,
                                           x=[m["x"] for m in track_markerset],
                                           y=[m["y"] for m in track_markerset],
                                            type= [m["type"] for m in track_markerset])
            self.setStates(marker=[m.id for m in state_markers],
                                 log_prob=[m["log_prob"] for m in stateset],
                                 state_vector=[m["state_vector"] for m in stateset],
                                 state_error=[m["state_error"] for m in stateset])
        if (debug_mode&0b010):
            self.setMarkers(image=[m["image"] for m in prediction_markerset],
                            type=[m["type"] for m in prediction_markerset],
                            x=[m["x"] for m in prediction_markerset],
                            y=[m["y"] for m in prediction_markerset],
                            text=[m["text"] for m in prediction_markerset],
                            style=[m["style"] for m in prediction_markerset])
            pred_markers = self.getMarkers(image=image,
                                           x=[m["x"] for m in prediction_markerset],
                                           y=[m["y"] for m in prediction_markerset],
                                          type=[m["type"] for m in prediction_markerset])
            self.setPredictions(marker=[m.id for m in pred_markers],
                                 log_prob=[m["log_prob"] for m in predictionset],
                                 prediction_vector=[m["prediction_vector"] for m in predictionset],
                                 prediction_error=[m["prediction_error"] for m in predictionset])
        if (debug_mode&0b100):
            self.setMarkers(image=[m["image"] for m in measurement_markerset],
                            type=[m["type"] for m in measurement_markerset],
                            x=[m["x"] for m in measurement_markerset],
                            y=[m["y"] for m in measurement_markerset],
                            text=[m["text"] for m in measurement_markerset])
            meas_markers = self.getMarkers(image=image,
                                           x=[m["x"] for m in measurement_markerset],
                                           y=[m["y"] for m in measurement_markerset],
                                           type=[m["type"] for m in measurement_markerset])
            self.setMeasurements(marker=[m.id for m in meas_markers],
                                 log_prob=[m["log_prob"] for m in measurementset],
                                 measurement_vector=[m["measurement_vector"] for m in measurementset],
                                 measurement_error=[m.get("measurement_error", None) for m in measurementset])

        print("Got %s Filters" % len(Tracker.ActiveFilters.keys()))


    def write_to_DB_cam(self, *args, **kwargs):
        return self.write_to_DB(*args, cam_values=True, **kwargs)


    def _CheckExtVersion(self):
        try:
            ext_version = self.db.execute_sql('SELECT value FROM meta WHERE key = "DataFileExtendedVersion"').fetchone()[0]
        except (KeyError, peewee.DoesNotExist, TypeError):
            ext_version = "0"
        print("Open extended database with version", ext_version)
        if int(ext_version) < int(self._current_ext_version):
            self._migrateDBExtFrom(ext_version)
        elif int(ext_version) > int(self._current_ext_version):
            print("Warning Extended Database version %d is newer than PenguTrack version %d "
                  "- please get an updated Version!"
                  % (int(ext_version), int(self._current_ext_version)))
            print("Proceeding on own risk!")
        return ext_version

    def _migrateDBExtFrom(self, version):
        print("Migrating Ext-DB from version %s" % version)
        nr_version = int(version)
        self.db.get_conn().row_factory = dict_factory

        if nr_version < 1:
            print("\tto 1")
            with self.db.transaction():
                try:
                    self.db.execute_sql(
                        'CREATE TABLE "measurement_tmp" ("id" INTEGER NOT NULL PRIMARY KEY, "marker_id" INTEGER NOT NULL, "log_prob" REAL, "measurement_vector" BLOB NOT NULL, "measurement_error" BLOB, FOREIGN KEY ("marker_id") REFERENCES "marker" ("id") ON DELETE CASCADE)')
                    # self.db.execute_sql(
                    #     'CREATE TABLE "measurement_tmp" ("id" INTEGER NOT NULL PRIMARY KEY, "marker_id" INTEGER NOT NULL, "log" REAL NOT NULL, "x" REAL NOT NULL, "y" REAL NOT NULL, "z" REAL NOT NULL, FOREIGN KEY ("marker_id") REFERENCES "marker" ("id") ON DELETE CASCADE)')
                    try:
                        measurements = self.db.execute_sql('SELECT * from measurement').fetchall()
                        for meas in measurements:
                            self.db.execute_sql(
                                'INSERT INTO measurement_tmp ("id", "marker_id", "log_prob", "measurement_vector", measurement_error"") VALUES(?, ?, ?, ?, ?)',
                                [meas["id"], meas["marker_id"], meas["log"], np.array([meas.get("x",0),
                                                                                               meas.get("y", 0),
                                                                                               meas.get("z", 0)]), None])
                        self.db.execute_sql('DROP TABLE measurement')
                    except peewee.OperationalError:
                        pass
                    self.db.execute_sql('ALTER TABLE measurement_tmp RENAME TO measurement')
                    self.db.execute_sql('CREATE INDEX "measurement_marker_id" ON "measurement" ("marker_id");')
                except peewee.OperationalError:
                    raise
                pass
            self._SetExtVersion(1)

        self.db.get_conn().row_factory = None


    def _SetExtVersion(self, nr_new_version):
        self.db.execute_sql("INSERT OR REPLACE INTO meta (id,key,value) VALUES ( \
                                            (SELECT id FROM meta WHERE key='DataFileExtendedVersion'),'DataFileExtendedVersion',%s)" % str(
            nr_new_version))

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
            tracker_class = dict(inspect.getmembers(PenguTrack.Filters))[db_tracker.tracker_class]
            filter_class = dict(inspect.getmembers(PenguTrack.Filters))[db_tracker.filter_class]
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