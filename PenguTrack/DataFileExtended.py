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
import scipy.stats
import json
import inspect


""" Parse Functions """

def parse(x):
    """
    Function to parse any object into a json-encodable form.
    iterates over lists, tuples and dicts.
    raises TypeError if object is not parsable.
    :param x: any object
    :return: object with json-encodable type
    """
    if type(x) in [str, float, int, bool]:
        return x
    elif type(x) == list:
        return parse_list(x)
    elif type(x) == tuple:
        return parse_list(x)
    elif type(x) == dict:
        return parse_dict(x)
    elif listable(x):
        return [parse(xx) for xx in x]
    elif type(x) == np.ndarray:
        return float(x)
    # handling of numpy data types
    elif str(type(x)).split("'")[1].split(".")[-1] in dict(inspect.getmembers(np)):
        return float(x)
    elif x is None:
        return None
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def parse_list(x):
    """
    Iterates over a list and returns a json-encodable list.
    :param x: list-like object
    :return: list with json-encodable objects
    """
    if listable(x):
        if len(x)>0:
            return [parse(xx) for xx in x]
        else:
            return parse(None)
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def parse_dict(x):
    """
    Iterates over a dict and returns a json-encodable dict.
    :param x: dict-like object
    :return: dict with json-encodable objects
    """
    if listable(x.keys()):
        if len(list(x))>0:
            return dict([[xx,parse(x[xx])] for xx in x])
        else:
            return parse(None)
    else:
        raise TypeError("Type %s can not be parsed!"%type(x))

def listable(x):
    """
    Auxiliary function to determine if object can be parsed as list.
    :param x: potential list like object
    :return: bool
    """
    try:
        list(x)
    except TypeError:
        return False
    else:
        return True#len(list(x))>0

def parse_model_name(model):
    """
    returns name of model class from class string
    :param model: pengu-track model
    :return: string
    """
    return str(model.__class__).split("'")[1].split(".")[-1]


def is_dist(x):
    """
    Checks if object is a scipy stats distribution
    :param x: potential distribution
    :return: bool
    """
    return str(type(x)).count("scipy.stats")

def parse_dist(dist):
    """
    Parses scipy stats distributions into a json-encodable form.
    WARNING: not all dists will be recoverable from parsed object.
    :param dist: scipy stats dist
    :return: list of dist name and initialisation arguments.
    """
    return [parse_dist_name(dist), parse_dist_dict(dist.__dict__)]

def parse_dist_name(dist):
    """
    Returns name of scipy stats dist from class string.
    :param dist: scipy stats dist
    :return: string
    """
    return str(dist.__class__).split("'")[1].split(".")[-1].replace("_frozen", "")


def parse_dist_dict(dist_dict):
    """Auxiliar function"""
    return dict(mean=dist_dict.get("mean",None),
                cov=dist_dict.get("cov", None),
                lower=dist_dict.get("lower", None),
                upper=dist_dict.get("upper", None))


def parse_filter_class(filter_class):
    """
    Parses Pengu Track Filter class string to human readable string
    :param filter_class: PenguTrack Filter class
    :return: string
    """
    try:
        return str(filter_class).split("'")[1].split(".")[-1]
    except IndexError:
        return "Filter"


def parse_tracker_class(filter_class):
    """
    Parses Pengu Track Tracker class string to human readable string
    :param filter_class: PenguTrack Tracker class
    :return: string
    """
    try:
        return str(filter_class).split("'")[1].split(".")[-1]
    except IndexError:
        return "MultiFilter"


def reverse_dict(D):
    """
    Auxiliar function to switch dict entries with keys.
    :param D: dictionary
    :return: dictionary
    """
    return dict([[D[d],d] for d in D])

class MatrixField(peewee.BlobField):
    """
    DataBase field for numpy arrays. Encoding and decoding into and from binary object.
    """
    @staticmethod
    def encode(value):
        """
        Encodes numpy array into binary object.
        :param value: numpy array
        :return: binary object
        """
        if np.all(value == np.nan):
            value = np.array([None])
        value_b = np.array(value).astype(np.float64).tobytes()
        shape_b = np.array(np.array(value).shape).astype(np.int64).tobytes()
        len_shape_b = np.array(len(np.array(value).shape)).astype(np.int64).tobytes()
        value = len_shape_b+shape_b+value_b
        return value

    def db_value(self, value):
        """
        Wraps encoding of numpy arrays into binary objects.
        :param value: numpy array
        :return: DataBase entry
        """
        value = self.encode(value)
        if not PY3:
            return super(MatrixField, self).db_value(value)
        else:
            return super(MatrixField, self).db_value(value)

    @staticmethod
    def decode(value):
        """
        Decodes previously encoded binary object to numpy array.
        :param value: binary object
        :return: numpy array
        """
        if value is not None and not len(value) == 0:
            l = np.frombuffer(value, dtype=np.int64, count=1, offset=0)
            if l == 0:
                return None
            shape = np.frombuffer(value, dtype=np.int64, count=int(l), offset=8)
            array = np.frombuffer(value, dtype=np.float64, count=int(np.prod(shape)), offset=8+int(l)*8)
            return array.reshape(shape)
        else:
            return None

    def python_value(self, value):
        """
        Wrapper for decoding function. Returns numpy array from binary object.
        :param value: DataBase entry
        :return: numpy array
        """
        value = super(MatrixField, self).python_value(value)
        return self.decode(value)


class ListField(peewee.TextField):
    """
    A generic field for python list. Uses json encoding.
    """
    def db_value(self, value):
        value=json.dumps(parse_list(value))
        if PY3:
            return super(ListField, self).db_value(value)
        return super(ListField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        value = json.loads(super(ListField, self).python_value(value))
        if value is None:
            return []
        else:
            return value


class DictField(peewee.TextField):
    """
    A generic field for python dict. Uses json encoding.
    """
    def db_value(self, value):
        value = json.dumps(parse_dict(value))
        if PY3:
            return super(DictField, self).db_value(value)
        return super(DictField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        value = json.loads(super(DictField, self).python_value(value))
        if value is None:
            return {}
        else:
            return value


class NumDictField(peewee.BlobField):
    """ A database field, that encodes purely numerical dicts to database entries."""
    def db_value(self, value):
        value = np.array([[v,value[v]] for v in value]).tobytes()
        if PY3:
            return super(NumDictField, self).db_value(value)
        return super(NumDictField, self).db_value(peewee.binary_construct(value))

    def python_value(self, value):
        value=super(NumDictField, self).python_value(value)
        if not PY3:
            value = dict([[v[0],v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1,2))])
        else:
            value = dict([[v[0], v[1]] for v in np.frombuffer(value, dtype=int).reshape((-1, 2))])
        if value is None:
            return {}
        else:
            return value

class DataFileExtended(clickpoints.DataFile):
    """
    Extended DataBase Class to save PenguTrack objects generic and rebuilt them from databases.
    """
    TYPE_BELIEVE = 0
    TYPE_PREDICTION = 1
    TYPE_MEASUREMENT = 2
    def __init__(self, *args, **kwargs):
        """
        Extended DataBase Class to save PenguTrack objects generically and rebuilt them from databases.
        """
        clickpoints.DataFile.__init__(self, *args, **kwargs)
        # Define ClickPoints Marker
        self.track_marker_type = self.setMarkerType(name="Track_Marker", color="#00FF00", mode=self.TYPE_Track,
                                                    style='{"transform": "image","color":"hsv"}')
        self.detection_marker_type = self.setMarkerType(name="Detection_Marker", color="#FF0000", style='{"scale":1.2}')
        self.prediction_marker_type = self.setMarkerType(name="Prediction_Marker", color="#0000FF",
                                                         style='{"shape": "circle", "transform": "image"}')
        #Namespace
        db = self

        # Version Handling
        self._current_ext_version = 2
        ext_version = self._CheckExtVersion()

        """Dist Entry"""
        class Distribution(db.base_model):
            """ Entry for scipy stats distributions """
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
            """ Entry for pengu track Models"""
            # model name
            name = peewee.TextField()

            # Optimisation lists.
            opt_params = ListField()
            opt_params_shape = DictField()
            opt_params_borders = DictField()

            # Inititialisation Argmuents. Used later to restore the original object from database.
            initial_args = ListField()
            initial_kwargs = DictField()

            # Possible extensions (untypical measured variables like object size or posture) are also stored
            extensions = ListField()

            measured_variables = ListField()

            # Dimension Values are needed in any case
            state_dim = peewee.IntegerField(default=1)
            control_dim = peewee.IntegerField(default=1)
            meas_dim = peewee.IntegerField(default=1)
            evolution_dim = peewee.IntegerField(default=1)

            # Mathematical description of model according to Kalman nomenclature
            state_matrix = MatrixField()
            control_matrix = MatrixField()
            measurement_matrix = MatrixField()
            evolution_matrix = MatrixField()
        # Built table
        if "model" not in db.db.get_tables():
            Model.create_table()
        self.table_model = Model


        """Tracker Entry"""
        class Tracker(db.base_model):
            """Tracker object. In most cases a table with single entry."""
            # e.g. Hungarian Tracker
            tracker_class = peewee.TextField()
            # Class to built new Filters from
            filter_class = peewee.TextField()
            # Model for new Filters
            model = peewee.ForeignKeyField(db.table_model,
                                           related_name='tracker_model',
                                           on_delete='CASCADE')
            # Tracker Parameters
            filter_threshold = peewee.IntegerField()
            log_probability_threshold = peewee.FloatField()
            measurement_probability_threshold = peewee.FloatField()
            assignment_probability_threshold = peewee.FloatField()

            # Inititialisation Arguments for new Filter
            filter_args = ListField()
            filter_kwargs = DictField()

            def __getattribute__(self, item):
                # Easy access to tracks
                if item == "tracks":
                    return [f.track for f in self.tracker_filters]
                return super(Tracker, self).__getattribute__(item)
        # built table
        if "tracker" not in db.db.get_tables():
            Tracker.create_table()
        self.table_tracker = Tracker


        """Filter Entry"""
        class Filter(db.base_model):
            """ Table to store (Kalman-)Filter Objects. Mostly holds junctions between other database entries."""
            # clickpoints track (unique)
            track = peewee.ForeignKeyField(db.table_track,
                                           unique=True,
                                           related_name='track_filter', on_delete='CASCADE')
            # tracker (e.g. Hungarian)
            tracker = peewee.ForeignKeyField(db.table_tracker,
                                             related_name='tracker_filters', on_delete='CASCADE')
            # distributions from dist table
            measurement_distribution = peewee.ForeignKeyField(db.table_distribution,
                                                              unique=True,
                                                              related_name='mdist_filter',
                                                              on_delete='CASCADE')
            state_distribution = peewee.ForeignKeyField(db.table_distribution,
                                                        unique=True,
                                                        related_name='sdist_filter',
                                                        on_delete='CASCADE')
            # model from model table
            model = peewee.ForeignKeyField(db.table_model,
                                           unique=True,
                                           related_name='model_filter',
                                           on_delete='CASCADE')
        # built table
        if "filter" not in db.db.get_tables():
            Filter.create_table()
        self.table_filter = Filter

        """State Entry"""
        class PT_State(db.base_model):
            """ Table storing all kinds of states: Measurements, Predictions, Believes.
             Including their covariance matrices and log_probability."""
            # The filter this state belongs to
            filter = peewee.ForeignKeyField(db.table_filter,
                                           related_name='filter_states', on_delete='CASCADE')
            # Corresponding image
            image = peewee.ForeignKeyField(db.table_image,
                                           related_name='image_states', on_delete='CASCADE')
            # Type: Believe, Prediction, Measurement
            type = peewee.IntegerField(default=0)
            # Log_Probability assuming filter model
            log_prob = peewee.FloatField(null=True, default=0)
            # Actual state vector and covariance matrix
            state_vector = MatrixField()
            state_error = MatrixField(null=True)
        # built table
        if "pt_state" not in db.db.get_tables():
            PT_State.create_table()
        self.table_state = PT_State

        """Probability Gain Entry"""
        class Probability_Gain(db.base_model):
            """ Table holding the cost matrices for each frame """
            # corresponding image
            image = peewee.ForeignKeyField(db.table_image, related_name="probability_gains", on_delete='CASCADE')
            # corresponding tracker
            tracker = peewee.ForeignKeyField(db.table_tracker, related_name="tracker_prob_gain", on_delete='CASCADE')
            # cost matrix
            probability_gain = MatrixField()
            # dictionary linking filter ids with cost_matrix rows
            probability_gain_dict = NumDictField()
            # dictionary linking filter ids and assigned measurements
            probability_assignment_dict = NumDictField()
        # built table
        if "probability_gain" not in db.db.get_tables():
            Probability_Gain.create_table()
        self.table_probability_gain = Probability_Gain


    def setState(self, id=None, filter=None, image=None, type=None, log_prob=None, state_vector=None, state_error=None):
        """
        Update or create State entry.
        :param id: int, optional
            the id of the state entry
        :param filter: entry of filter table
            filter must be specified.
        :param image: entry of image table
            image must be specified
        :param type: int, optional
            defines if state is Believe, Measurement or Prediction
        :param log_prob: float, optional
            defines the probability of this state
        :param state_vector: array-like
            the mathematical state
        :param state_error: array-like, optional
            covariance matrix of the state, used to determine error
        :return: State entry
        """
        dictionary = dict(id=id, filter=filter, image=image, type=type)
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

    def getState(self, id=None, filter_id=None, image_id=None, type=None, log_prob=None):
        query = self.table_state.select()
        query = addFilter(query, id, self.table_state.id)
        query = addFilter(query, filter_id, self.table_state.filter_id)
        query = addFilter(query, image_id, self.table_state.image_id)
        query = addFilter(query, type, self.table_state.type)
        query = addFilter(query, log_prob, self.table_state.log_prob)
        return query

    def deleteState(self, id=None, filter_id=None, image_id=None, type=None, log_prob=None):
        query = self.table_state.delete()
        query = addFilter(query, id, self.table_state.id)
        query = addFilter(query, filter_id, self.table_state.filter_id)
        query = addFilter(query, image_id, self.table_state.image_id)
        query = addFilter(query, type, self.table_state.type)
        query = addFilter(query, log_prob, self.table_state.log_prob)
        return query.execute()

    def setTracker(self,tracker_class="", filter_class="", model=None,
               filter_threshold=3,log_probability_threshold=0.,
               measurement_probability_threshold=0.0,
               assignment_probability_threshold=0.0,
               filter_args=[], filter_kwargs={}, id=None):
        """
        Update or create Tracker entry.
        :param tracker_class: PenguTrack MultiFilter Class
        :param filter_class: PenguTrack Filter Class
        :param model: PenguTrack Model Class
        :param filter_threshold: int, optional
            number of lag frames over which tracks will be deprecated
        :param log_probability_threshold: float, optional
            threshold for cost matrix
        :param measurement_probability_threshold: float, optional
            threshold for allowing assignment of measurements
        :param assignment_probability_threshold: float, optional
            threshold for assigning tracks
        :param filter_args: list
            initialisation list to create new filters with this tracker
        :param filter_kwargs: dict
            initialisation dict to create new filters with this tracker
        :param id: int, optional
            id of the tracker
        :return: Tracker entry
        """
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
        """
        Load Tracker entry from database
        :param tracker_class: PenguTrack MultiFilter Class
        :param filter_class: PenguTrack Filter Class
        :param model: PenguTrack Model Class
        :param filter_threshold: int, optional
            number of lag frames over which tracks will be deprecated
        :param log_probability_threshold: float, optional
            threshold for cost matrix
        :param measurement_probability_threshold: float, optional
            threshold for allowing assignment of measurements
        :param assignment_probability_threshold: float, optional
            threshold for assigning tracks
        :param filter_args: list
            initialisation list to create new filters with this tracker
        :param filter_kwargs: dict
            initialisation dict to create new filters with this tracker
        :param id: int, optional
            id of the tracker
        :return: Tracker entry
        """
        query = self.table_tracker.select()
        query = addFilter(query, id, self.table_tracker.id)
        query = addFilter(query, tracker_class, self.table_tracker.tracker_class)
        query = addFilter(query, filter_class, self.table_tracker.filter_class)
        query = addFilter(query, filter_threshold, self.table_tracker.filter_threshold)
        query = addFilter(query, log_probability_threshold, self.table_tracker.log_probability_threshold)
        query = addFilter(query, measurement_probability_threshold, self.table_tracker.measurement_probability_threshold)
        query = addFilter(query, assignment_probability_threshold, self.table_tracker.assignment_probability_threshold)
        return query

    def deleteTracker(self, id=None, tracker_class=None, filter_class=None):
        query = self.table_tracker.delete()
        query = addFilter(query, id, self.table_tracker.id)
        query = addFilter(query, tracker_class, self.table_tracker.tracker_class)
        query = addFilter(query, filter_class, self.table_tracker.filter_class)
        return query.execute()


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

    def getDistribution(self, id=None, name=None):
        query = self.table_distribution.select()
        query = addFilter(query, id, self.table_distribution.id)
        query = addFilter(query, name, self.table_distribution.name)
        return query

    def deleteDistribution(self, id=None, name=None):
        query = self.table_distribution.delete()
        query = addFilter(query, id, self.table_distribution.id)
        query = addFilter(query, name, self.table_distribution.name)
        return query.execute()

    def setModel(self, id=None, name="Model", state_dim=1, control_dim=1, meas_dim=1, evolution_dim=1,
                 state_matrix=None, control_matrix=None, measurement_matrix=None, evolution_matrix=None,
                 opt_params=[], opt_params_shape={}, opt_params_borders={}, initial_args=[], initial_kwargs={},
                 extensions=[], measured_variables=[]):

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
        dictionary.update(dict(name=name,
                               state_matrix=state_matrix, control_matrix=control_matrix,
                               measurement_matrix=measurement_matrix, evolution_matrix=evolution_matrix,
                               opt_params=opt_params, opt_params_shape=opt_params_shape,
                               opt_params_borders=opt_params_borders, initial_args=initial_args,
                               initial_kwargs=initial_kwargs,extensions=extensions,
                               measured_variables=measured_variables))
        setFields(item, dictionary)
        item.save()
        return item

    def getModel(self, id=None, name=None):
        query = self.table_model.select()
        query = addFilter(query, id, self.table_model.id)
        query = addFilter(query, name, self.table_model.name)
        return query

    def deleteModel(self, id=None, name=None):
        query = self.table_model.delete()
        query = addFilter(query, id, self.table_model.id)
        query = addFilter(query, name, self.table_model.name)
        return query.execute()

    def setFilter(self, id=None, track=None, tracker=None,
                  measurement_distribution=None,
                  state_distribution=None,
                  model=None):
        dictionary = dict(id=id, track=track)
        try:
            item = self.table_filter.get(**dictionary)
        except peewee.DoesNotExist:
            item = self.table_filter()
            # except peewee.OperationalError:
            #     model = self.table_model()

        dictionary.update(dict(tracker=tracker,
                               measurement_distribution=measurement_distribution,
                               state_distribution=state_distribution,
                               model=model))
        setFields(item, dictionary)
        item.save()
        return item

    def setFilters(self, id=None, track=None, tracker=None, measurement_distribution=None,
                  state_distribution=None, model=None):
        data = packToDictList(self.table_filter, id=id, track=track, tracker=tracker,
                              measurement_distribution=measurement_distribution,
                              state_distribution=state_distribution,
                              model=model)
        return self.saveUpsertMany(self.table_filter, data)

    def getFilter(self, id=None, track_id=None):
        query = self.table_filter.select()
        query = addFilter(query, track_id, self.table_filter.track_id)
        query = addFilter(query, id, self.table_filter.id)
        if len(query) > 0:
            return query[0]
        else:
            return None

    def getFilters(self, id=None, track_id=None):
        query = self.table_filter.select()
        query = addFilter(query, track_id, self.table_filter.track_id)
        query = addFilter(query, id, self.table_filter.id)
        return query

    def deleteFilters(self, id=None, track_id=None, tracker_id=None, measurement_distribution_id=None,
                      state_distribution_id=None, model_id=None):
        query = self.table_filter.delete()
        query = addFilter(query, id, self.table_filter.id)
        query = addFilter(query, track_id, self.table_filter.track_id)
        query = addFilter(query, tracker_id, self.table_filter.tracker_id)
        query = addFilter(query, measurement_distribution_id, self.table_filter.measurement_distribution_id)
        query = addFilter(query, state_distribution_id, self.table_filter.state_distribution_id)
        query = addFilter(query, model_id, self.table_filter.model_id)
        return query.execute()


    def setProbabilityGain(self, id=None, image=None, tracker=None,
                           probability_gain=None, probability_gain_dict=None, probability_assignment_dict=None):
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
                               probability_gain_dict=probability_gain_dict,
                               probability_assignment_dict=probability_assignment_dict))
        setFields(item, dictionary)
        item.save()
        return item

    def getProbabilityGain(self, id=None, image_id=None, tracker_id=None):
        query = self.table_probability_gain.select()
        query = addFilter(query, id, self.table_probability_gain.id)
        query = addFilter(query, image_id, self.table_probability_gain.image_id)
        query = addFilter(query, tracker_id, self.table_probability_gain.tracker_id)
        return query

    def deleteProbabilityGain(self, id=None, image_id=None, tracker_id=None):
        query = self.table_probability_gain.delete()
        query = addFilter(query, id, self.table_probability_gain.id)
        query = addFilter(query, image_id, self.table_probability_gain.image_id)
        query = addFilter(query, tracker_id, self.table_probability_gain.tracker_id)
        return query.execute()

    def setStates(self, image=None, filter=None, type=None, log_prob=None, state_vector=None, state_error=None):
        data = packToDictList(self.table_state, image=image, filter=filter, type=type,
                              log_prob=log_prob,
                              state_vector=state_vector,
                              state_error=state_error)
        return self.saveUpsertMany(self.table_state, data)

    def deletetOld(self):
        """Deletes old Clickpoints marker entries"""
        self.deleteMarkers(type=self.track_marker_type)
        self.deleteMarkers(type=self.prediction_marker_type)
        self.deleteMarkers(type=self.detection_marker_type)
        # Delete Old Tracks
        self.deleteTracks(type=self.track_marker_type)
        # Delete Distributions
        self.deleteState()
        # Delete Distributions
        self.deleteDistribution()
        # Delete Model
        self.deleteModel()
        # Delete Filter
        self.deleteFilters()
        # Delete Tracker
        self.deleteTracker()



    def init_tracker(self, model, tracker):
        """
        Initialise Database entries for tracker using PenguTrack model and tracker object
        :param model: PenguTrack Model
        :param tracker: PenguTrack Tracker
        :return: model_db, tracker_db
            entries of the extended Database. Use these to set new Filters.
        """
        model_item = self.setModel(name=parse_model_name(model), state_dim=model.State_dim, control_dim=model.Control_dim,
                      meas_dim=model.Meas_dim, evolution_dim=model.Evolution_dim,
                      state_matrix=model.State_Matrix, control_matrix=model.Control_Matrix,
                      measurement_matrix=model.Measurement_Matrix, evolution_matrix=model.Evolution_Matrix,
                               opt_params=model.Opt_Params, opt_params_shape=model.Opt_Params_Shape,
                               opt_params_borders=model.Opt_Params_Borders, initial_args=model.Initial_Args,
                               initial_kwargs=model.Initial_KWArgs,extensions=model.Extensions,
                               measured_variables=model.Measured_Variables)
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
                    debug_mode=0b1111, image_transform=None, verbose=True):
        """
        Writes a tracking step to the extended DataBase
        :param Tracker: PenguTrack Tracker (e.g. Hungarian Tracker) holding multiple
        :param image:
        :param i:
        :param text:
        :param cam_values:
        :param db_tracker:
        :param db_model:
        :param debug_mode:
        :return:
        """
        if db_tracker is None or db_model is None:
            if len(self.getTrackers())>0:
                db_tracker = self.getTrackers()[0]
                db_model = db_tracker.model
            else:
                db_model, db_tracker = self.init_tracker(Tracker.Filters[0].Model, Tracker)

        if text is None:
            set_text = True
        else:
            set_text = False

        if i is None:
            i = image.sort_index

        if image_transform is not None and cam_values:
            cam_states = image_transform([Tracker.Filters[k].X[i] for k in Tracker.Filters if i in Tracker.Filters[k].X])
            cam_states = dict(zip([k for k in Tracker.Filters if i in Tracker.Filters[k].X], cam_states))
            cam_errors = image_transform([np.diag(Tracker.Filters[k].X_error[i]) for k in Tracker.Filters if i in Tracker.Filters[k].X_error])
            cam_errors = dict(zip([k for k in Tracker.Filters if i in Tracker.Filters[k].X_error], [np.diag(c) for c in cam_errors]))

            if debug_mode & 0b1010:
                cam_pred = image_transform([Tracker.Filters[k].Predicted_X[i] for k in Tracker.Filters if i in Tracker.Filters[k].Predicted_X])
                cam_pred = dict(zip([k for k in Tracker.Filters if i in Tracker.Filters[k].Predicted_X], cam_pred))
                cam_pred_err = image_transform([np.diag(Tracker.Filters[k].Predicted_X_error[i]) for k in Tracker.Filters if i in Tracker.Filters[k].Predicted_X_error])
                cam_pred_err = dict(zip([k for k in Tracker.Filters if i in Tracker.Filters[k].Predicted_X_error], [np.diag(c) for c in cam_pred_err]))

        with self.db.atomic() as transaction:
            markerset = []
            stateset = []
            prediction_markerset = []
            measurement_markerset = []
            filter_list = []
            db_filters = dict([[f.id, f] for f in self.getFilters()])
            # Get Tracks from Filters
            for k in Tracker.Filters.keys():
                x = y = np.nan
                if i not in Tracker.Probability_Gain:
                    prob = Tracker.Filters[k].log_prob(keys=[i], update=Tracker.Filters[k].ProbUpdate)
                else:
                    if k not in Tracker.Probability_Assignment_Dicts[i]:
                        prob = Tracker.Filters[k].log_prob(keys=[i], update=Tracker.Filters[k].ProbUpdate)
                    else:
                        prob = Tracker.Probability_Gain[i][reverse_dict(Tracker.Probability_Gain_Dicts[i])[k], Tracker.Probability_Assignment_Dicts[i][k]]

                if k in db_filters:
                    new = ""
                    db_filter = db_filters[k]
                    db_track = db_filter.track
                else:
                    db_track = self.setTrack(self.track_marker_type)
                    new = "new "
                    db_dist_s = self.setDistribution(name=parse_dist_name(Tracker.Filters[k].State_Distribution),
                                                     **parse_dist_dict(Tracker.Filters[k].State_Distribution.__dict__))
                    db_dist_m = self.setDistribution(name=parse_dist_name(Tracker.Filters[k].Measurement_Distribution),
                                                     **parse_dist_dict(Tracker.Filters[k].Measurement_Distribution.__dict__))
                    db_filter_model = self.setModel(name=db_model.name,
                                                    state_dim=db_model.state_dim,
                                                    control_dim=db_model.control_dim,
                                                    meas_dim=db_model.meas_dim,
                                                    evolution_dim=db_model.evolution_dim,
                                                    state_matrix=db_model.state_matrix,
                                                    control_matrix=db_model.control_matrix,
                                                    measurement_matrix=db_model.measurement_matrix,
                                                    evolution_matrix=db_model.evolution_matrix,
                                                    opt_params=db_model.opt_params,
                                                    opt_params_shape=db_model.opt_params_shape,
                                                    opt_params_borders=db_model.opt_params_borders,
                                                    initial_args=db_model.initial_args,
                                                    initial_kwargs=db_model.initial_kwargs,
                                                    extensions=db_model.extensions,
                                                    measured_variables=db_model.measured_variables)

                    filter_list.append(dict(id=k,
                                            model=db_filter_model.id, tracker=db_tracker.id,
                                            track=db_track.id,
                                            measurement_distribution=db_dist_m.id,
                                            state_distribution=db_dist_s.id))

                # Case 1: we tracked something in this filter
                if i in Tracker.Filters[k].X.keys():
                    state = Tracker.Filters[k].X[i]
                    try:
                        state_err = Tracker.Model.measure(Tracker.Filters[k].X_error[i])
                    except KeyError:
                        state_err = np.zeros((len(state), len(state)))
                    i_x = Tracker.Model.Measured_Variables.index("PositionX")
                    i_y = Tracker.Model.Measured_Variables.index("PositionY")

                    if cam_values and image_transform is not None:
                        # state_image = image_transform(state)
                        x = Tracker.Model.measure(cam_states[k])[i_x]
                        y = Tracker.Model.measure(cam_states[k])[i_y]
                        error_image = cam_errors.get(k, np.diag(np.zeros((len(state), len(state)))))
                        # error_image = image_transform(np.diag(state_err))
                        state_err = (error_image[i_x]+error_image[i_y])*0.5
                    else:
                        x = Tracker.Model.measure(state)[i_x]
                        y = Tracker.Model.measure(state)[i_y]
                        state_err = (state_err[i_x, i_x]+state_err[i_y,i_y])*0.5

                    if verbose:
                        print('Setting %sTrack(%s)-Marker at %s, %s' % (new, k, x, y))
                    if set_text:
                        text = 'Filter %s, Prob %.2f' % (k, prob)
                    markerset.append(dict(image=image, type=self.track_marker_type, track=db_track, x=y, y=x,
                                                text=text,
                                                style = '{}'))
                                                # style='{"scale":%.2f}'%(state_err)))
                    if (debug_mode & 0b001):
                        stateset.append(dict(log_prob=prob,
                                             filter=k,
                                             image=image,
                                             type=self.TYPE_BELIEVE,
                                             state_vector=Tracker.Filters[k].X[i],
                                             state_error=Tracker.Filters[k].X_error.get(i, None)))

                # Case 2: we want to see the prediction markers
                if i in Tracker.Filters[k].Predicted_X.keys() and (debug_mode&0b010):
                    stateset.append(dict(log_prob=prob,
                                         filter=k,
                                         image=image,
                                         type=self.TYPE_PREDICTION,
                                         state_vector=Tracker.Filters[k].Predicted_X[i],
                                         state_error=Tracker.Filters[k].Predicted_X_error.get(i, None)))
                    if debug_mode&0b1010:
                        prediction = Tracker.Filters[k].Predicted_X[i]
                        try:
                            prediction_err = Tracker.Model.measure(Tracker.Filters[k].Predicted_X_error[i])
                        except KeyError:
                            prediction_err = np.zeros((len(prediction), len(prediction)))
                        i_x = Tracker.Model.Measured_Variables.index("PositionX")
                        i_y = Tracker.Model.Measured_Variables.index("PositionY")

                        if cam_values and image_transform is not None:
                            # prediction_image = image_transform(prediction)
                            # pred_x = Tracker.Model.measure(prediction_image)[i_x]
                            # pred_y = Tracker.Model.measure(prediction_image)[i_y]
                            pred_x = Tracker.Model.measure(cam_pred[k])[i_x]
                            pred_y = Tracker.Model.measure(cam_pred[k])[i_y]
                            # error_image = image_transform(np.diag(prediction_err))
                            error_image = cam_pred_err.get(k, np.diag(np.zeros((len(state), len(state)))))
                            pred_err = (error_image[i_x]+error_image[i_y])*0.5
                        else:
                            pred_x = Tracker.Model.measure(prediction)[i_x]
                            pred_y = Tracker.Model.measure(prediction)[i_y]
                            pred_err = (prediction_err[i_x, i_x]+prediction_err[i_y, i_y])*0.5

                        prediction_markerset.append(dict(image=image, x=pred_y, y=pred_x,
                                                         text="Filter %s" % k,
                                                         type=self.prediction_marker_type,
                                                         style='{"scale":%.2f}'%(pred_err)))

                # Case 3: we want to see the measurement markers
                if i in Tracker.Filters[k].Measurements.keys() and (debug_mode&0b100):
                    meas = Tracker.Filters[k].Measurements[i]
                    meas_x = meas.PositionX
                    meas_y = meas.PositionY

                    meas_err = meas.Covariance

                    stateset.append(dict(filter=k,
                                         log_prob=prob,
                                         image=image,
                                         type=self.TYPE_MEASUREMENT,
                                         state_vector=np.array([meas_x, meas_y]),
                                         state_error=meas_err))
                    if debug_mode&0b1100:
                        measurement_markerset.append(dict(image=image, x=meas_y, y=meas_x,
                                                          text="Track %s" % (db_track.id),
                                                          type=self.detection_marker_type))


        try:
            prob_gain = Tracker.Probability_Gain[i]
            prob_gain_dict = Tracker.Probability_Gain_Dicts[i]
            prob_assign_dict = Tracker.Probability_Assignment_Dicts[i]
            self.setProbabilityGain(image=image, tracker=db_tracker,
                                    probability_gain=prob_gain,
                                    probability_gain_dict=prob_gain_dict,
                                    probability_assignment_dict=prob_assign_dict)
        except KeyError:
            pass

        if len(filter_list)>0:
            self.setFilters(id=[f["id"] for f in filter_list],
                            track=[f["track"] for f in filter_list],
                            tracker=[f["tracker"] for f in filter_list],
                            measurement_distribution=[f["measurement_distribution"] for f in filter_list],
                            state_distribution=[f["state_distribution"] for f in filter_list],
                            model=[f["model"] for f in filter_list])

        if (debug_mode&0b001):
            self.setMarkers(image=[m["image"] for m in markerset],
                            type=[m["type"] for m in markerset],
                            track=[m["track"] for m in markerset],
                            x=[m["x"] for m in markerset],
                            y=[m["y"] for m in markerset],
                            text=[m["text"] for m in markerset],
                            style=[m["style"] for m in markerset])
        self.setStates(image=[m["image"] for m in stateset],
                       filter=[m["filter"] for m in stateset],
                       type=[m["type"] for m in stateset],
                       log_prob=[m["log_prob"] for m in stateset],
                       state_vector=[m["state_vector"] for m in stateset],
                       state_error=[m["state_error"] for m in stateset])

        if (debug_mode&0b1010):
            self.setMarkers(image=[m["image"] for m in prediction_markerset],
                            type=[m["type"] for m in prediction_markerset],
                            x=[m["x"] for m in prediction_markerset],
                            y=[m["y"] for m in prediction_markerset],
                            text=[m["text"] for m in prediction_markerset],
                            style=[m["style"] for m in prediction_markerset])
        if (debug_mode&0b1100):
            self.setMarkers(image=[m["image"] for m in measurement_markerset],
                            type=[m["type"] for m in measurement_markerset],
                            x=[m["x"] for m in measurement_markerset],
                            y=[m["y"] for m in measurement_markerset],
                            text=[m["text"] for m in measurement_markerset])
        if verbose:
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
                    # self.db.execute_sql(
                    #     'CREATE TABLE `measurement_tmp` ( `id`	INTEGER NOT NULL, `filter_id`	INTEGER NOT NULL, `marker_id`	INTEGER NOT NULL, `log_prob`	REAL, `measurement_vector`	BLOB NOT NULL, `measurement_error`	BLOB, PRIMARY KEY(id), FOREIGN KEY(`filter_id`) REFERENCES "filter" ( "id" ) ON DELETE CASCADE, FOREIGN KEY(`marker_id`) REFERENCES "marker" ( "id" ) ON DELETE CASCADE)')
                    self.db.execute_sql(
                        'CREATE TABLE `state_tmp` (`id`	INTEGER NOT NULL,`filter_id`	INTEGER NOT NULL,`image_id`	INTEGER NOT NULL, `type`	INTEGER NOT NULL, `log_prob`	REAL,`state_vector`	BLOB NOT NULL, `state_error`	BLOB, FOREIGN KEY(`image_id`) REFERENCES `image`(`id`) ON DELETE CASCADE, FOREIGN KEY(`filter_id`) REFERENCES `filter`(`id`) ON DELETE CASCADE, PRIMARY KEY(`id`))')
                    try:
                        measurements = self.db.execute_sql('select * from measurement m inner join (select track_id, image_id, id as marker_id from marker)mm on mm.marker_id==m.marker_id').fetchall()
                        for meas in measurements:
                            self.db.execute_sql(
                                'INSERT INTO state_tmp ("id", "filter_id", "image_id", "type", "log_prob","state_vector", "state_error") VALUES(?, ?, ?, ?, ?, ?, ?)',
                                [meas["id"], meas["track_id"], meas["image_id"], self.TYPE_MEASUREMENT , meas["log"], np.array([meas.get("x",0),
                                                                                               meas.get("y", 0),
                                                                                               meas.get("z", 0)]), None])
                        self.db.execute_sql('DROP TABLE measurement')
                    except peewee.OperationalError:
                        pass
                    self.db.execute_sql('ALTER TABLE state_tmp RENAME TO state')
                    # self.db.execute_sql('CREATE INDEX "measurement_marker_id" ON "state" ("marker_id");')
                    # self.db.execute_sql('CREATE INDEX "measurement_marker_id" ON "measurement" ("marker_id");')
                except peewee.OperationalError:
                    raise
                pass
            self._SetExtVersion(1)

        if nr_version < 2:
            print("\tto 2")
            with self.db.transaction():
                try:
                    self.db.execute_sql('ALTER TABLE state RENAME TO pt_state')
                except peewee.OperationalError:
                    if "pt_state" not in self.db.get_tables():
                        pass
                    else:
                        raise
            self._SetExtVersion(2)


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
        args = db_model.initial_args
        args = [] if args is None else args
        kwargs = db_model.initial_kwargs
        kwargs = {} if kwargs is None else kwargs
        Model_class = dict(inspect.getmembers(PenguTrack.Models))[db_model.name]
        out = Model_class(*args, **kwargs)
        out.State_dim = db_model.state_dim
        out.Control_dim = db_model.control_dim
        out.Meas_dim = db_model.meas_dim
        out.Evolution_dim = db_model.evolution_dim

        out.State_Matrix = db_model.state_matrix.reshape((db_model.state_dim,db_model.state_dim))
        out.Control_Matrix = db_model.control_matrix.reshape((db_model.state_dim,db_model.control_dim))
        out.Measurement_Matrix = db_model.measurement_matrix.reshape((db_model.meas_dim,db_model.state_dim))
        out.Evolution_Matrix = db_model.evolution_matrix.reshape((db_model.state_dim,db_model.evolution_dim))

        out.Opt_Params = db_model.opt_params
        out.Opt_Params_Shape = db_model.opt_params_shape
        out.Opt_Params_Borders = db_model.opt_params_borders

        out.Initial_Args = args
        out.Initial_KWArgs = kwargs
        out.Extensions = db_model.extensions
        out.Measured_Variables = db_model.measured_variables
        return out


    def dist_from_db(self, db_dist):
        dist_dict = dict(mean=db_dist.mean,
                         cov=db_dist.cov,
                         lower=db_dist.lower,
                         upper=db_dist.upper)
        dist_class = dict(inspect.getmembers(scipy.stats))[db_dist.name]
        # dist_class = eval("ss."+db_dist.name)
        return dist_class(**noNoneDict(**dist_dict))


    def filter_from_db(self, db_filter, filter_class=None, filter_args=None, filter_kwargs=None):
        model = self.model_from_db(db_filter.model)
        meas_dist = self.dist_from_db(db_filter.measurement_distribution)
        state_dist = self.dist_from_db(db_filter.state_distribution)
        if filter_class is None or filter_args is None or filter_kwargs is None:
            filter = PenguTrack.Filters.Filter(model, meas_dist=meas_dist, state_dist=state_dist)
        else:
            filter = filter_class(model, *filter_args, **filter_kwargs)
        X_raw = self.db.execute_sql('select sort_index, state_vector from (select * from pt_state where filter_id = ? and type=?) s inner join image i on s.image_id == i.id',
                                    params=[db_filter.id, self.TYPE_BELIEVE]).fetchall()
        Pred_raw = self.db.execute_sql('select sort_index, state_vector from (select * from pt_state where filter_id = ? and type=?) s inner join image i on s.image_id == i.id',
                                       params=[db_filter.id, self.TYPE_PREDICTION]).fetchall()
        Meas_raw = self.db.execute_sql('select sort_index, state_vector, state_error from (select * from pt_state where filter_id = ? and type=?) s inner join image i on s.image_id == i.id',
                                       params=[db_filter.id, self.TYPE_MEASUREMENT]).fetchall()

        X_err_raw = self.db.execute_sql('select sort_index, state_error from (select * from pt_state where filter_id = ? and type=?) s inner join image i on s.image_id == i.id',
                                        params=[db_filter.id, self.TYPE_BELIEVE]).fetchall()
        Pred_err_raw = self.db.execute_sql('select sort_index, state_error from (select * from pt_state where filter_id = ? and type=?) s inner join image i on s.image_id == i.id',
                                           params=[db_filter.id, self.TYPE_PREDICTION]).fetchall()

        filter.X.update(dict([[v[0], MatrixField.decode(v[1])] for v in X_raw]))
        filter.Predicted_X.update(dict([[v[0], MatrixField.decode(v[1])] for v in Pred_raw]))
        filter.Measurements.update(dict([[v[0], PT_Measurement(0.,
                                                               MatrixField.decode(v[1]),
                                                               cov=MatrixField.decode(v[2]))] for v in Meas_raw]))



        filter.X_error.update(dict([[v[0], MatrixField.decode(v[1])] for v in X_err_raw]))
        filter.Predicted_X_error.update(dict([[v[0], MatrixField.decode(v[1])] for v in Pred_err_raw]))

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
            tracker.Filters.update(dict([self.filter_from_db(db_filter,
                                                             filter_class=filter_class,
                                                             filter_args=db_tracker.filter_args,
                                                             filter_kwargs=db_tracker.filter_kwargs) for db_filter in db_tracker.tracker_filters]))
            tracker.LogProbabilityThreshold = db_tracker.log_probability_threshold
            tracker.FilterThreshold = db_tracker.filter_threshold
            tracker.AssignmentProbabilityThreshold = db_tracker.assignment_probability_threshold
            tracker.MeasurementProbabilityThreshold = db_tracker.measurement_probability_threshold
            p_gain_raw = self.db.execute_sql('select sort_index, probability_gain from (select * from probability_gain where tracker_id == ?) p inner join image i on p.image_id == i.id',
                                              [db_tracker.id])
            tracker.Probability_Gain.update(dict([[v[0], MatrixField.decode(v[1])] for v in p_gain_raw]))
            p_dict_raw = self.db.execute_sql('select sort_index, probability_gain_dict from (select * from probability_gain where tracker_id == ?) p inner join image i on p.image_id == i.id',
                                              [db_tracker.id])
            tracker.Probability_Gain_Dicts.update(dict([[v[0], MatrixField.decode(v[1])] for v in p_dict_raw]))
            out.append(tracker)
        return out

    def split(self, marker):
        new_track = super(DataFileExtended, self).split(marker)
        old_filter = self.getFilter(track_id=marker.track_id)
        new_filter = self.setFilter(track=new_track, tracker=old_filter.tracker,
                                    measurement_distribution=old_filter.measurement_distribution,
                                    state_distribution=old_filter.state_distribution,
                                    model=old_filter.model)
        self.db.execute_sql('update pt_state set filter_id=? where filter_id==? and image_id>?',
                            [new_filter.id, old_filter.id, marker.image_id])
        # pgains = self.db.execute_sql('',
        #                     [new_filter.id, old_filter.id, marker.image_id])

    def merge(self, track0, track1):
        # if we are not given a track..
        if not isinstance(track0, self.table_track):
            # interpret it as a track id and get the track entry
            track0 = self.table_track.get(id=track0)
            # if we don't get it, complain
            if track0 is None:
                raise ValueError("No valid track given.")
        if not isinstance(track1, self.table_track):
            # interpret it as a track id and get the track entry
            track1 = self.table_track.get(id=track1)
            # if we don't get it, complain
            if track1 is None:
                raise ValueError("No valid track given.")

        # find the image ids from this track and the other track
        image_ids0 = [m.image_id for m in track0.markers]
        image_ids1 = [m.image_id for m in track1.markers]
        self.db.execute_sql()


