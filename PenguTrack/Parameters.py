from qtpy import QtCore, QtGui, QtWidgets
import numpy as np


class QParameterEdit(QtWidgets.QWidget):
    """ base class for an widget to change the value of a parameter. """

    # flag that is set while the values are updated to prevent endless loops during update
    no_edit = False

    def __init__(self, parameter, layout):
        """
        Initialize the widget

        :param parameter: The parameter object for which the widget should be created.
        :param layout: The layout into which the widget should be inserted.
        """
        # call the super class initialization
        super().__init__()

        # if a layout is given add the widget to the layout
        if layout is not None:
            layout.addWidget(self)

        # store the parameter object
        self.parameter = parameter

        # store the type of the parameter
        self.type = parameter.type

        # add a layout for this widget (with 0 margins)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        """ value selector """
        # add subwidget for value selection of this widget (with 0 margins)
        self.widget_parameter = QtWidgets.QWidget()
        self.main_layout.addWidget(self.widget_parameter)

        # add a layout
        self.layout = QtWidgets.QHBoxLayout(self.widget_parameter)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # add a label containing the parameters name
        self.label = QtWidgets.QLabel(parameter.name)
        self.layout.addWidget(self.label)

        # add a stretch
        self.layout.addStretch()

        # if the parameter has a description, add a tooltip
        if self.parameter.desc is not None:
            self.setToolTip(self.parameter.desc)

        """ range selector """
        # add a subwidget for the range selector
        self.widget_range = QtWidgets.QWidget()
        self.main_layout.addWidget(self.widget_range)

        # add a layout
        self.layoutRange = QtWidgets.QHBoxLayout(self.widget_range)
        self.layoutRange.setContentsMargins(0, 0, 0, 0)

        # add a label containing the parameters name
        self.label = QtWidgets.QLabel(parameter.name)
        self.layoutRange.addWidget(self.label)
        self.layoutRange.addStretch()

        # add a checkbox to specify if the value should be fixed and not optimized
        self.checkbox_fixed = QtWidgets.QCheckBox("fixed")
        self.checkbox_fixed.stateChanged.connect(self.rangeChanged)
        self.layoutRange.addWidget(self.checkbox_fixed)

        self.displayRanges(False)

    def setValue(self, x):
        """ set the value of the parameter """

        # if the value is currently set, do nothing to prevent an endless loop
        if self.no_edit:
            return
        # set edit to true (subsequent calls to this function will do nothing)
        self.no_edit = True

        # call the setter method
        self.doSetValue(x)

        # change the value of the parameter
        if x is None:
            self.parameter.value = x
        else:
            self.parameter.value = self.type(x)
        # notify the parameter of it's change
        self.parameter.valueChanged()

        # set edit back to false
        self.no_edit = False

    def doSetValue(self, x):
        """ method to overload for the implementation. The widget should set here the value to its slider/edit widgets. """
        pass

    def rangeChanged(self):
        pass

    def displayRanges(self, do_display):
        """ display the value selector or the range selector. """
        # true means the range selector
        if do_display:
            self.widget_parameter.setHidden(True)
            self.widget_range.setHidden(False)
        # false means, show the value selector
        else:
            self.widget_parameter.setHidden(False)
            self.widget_range.setHidden(True)


class QNumberChooser(QParameterEdit):
    """ input widget to choose an integer or float value for a parameter. """

    # the slider input widget
    slider = None
    nan_checkbox = None

    check_log = None

    def __init__(self, parameter, layout, type_int):
        # call the super method's initializer
        super().__init__(parameter, layout)

        # if the parameter has a minimum and maximum, we can add a slider to display it
        if self.parameter.min is not None and self.parameter.max is not None:
            # create a slider object
            self.slider = QtWidgets.QSlider()
            # add it to the widget
            self.layout.addWidget(self.slider)
            # set the orientation of the slider to horizontal
            self.slider.setOrientation(QtCore.Qt.Horizontal)
            # set the range
            self.slider.setRange(parameter.min, parameter.max)
            # and connect the valueChanged signal
            self.slider.valueChanged.connect(self.setValue)

        # if it is an integer, use a QSpinBox
        if type_int:
            self.spinBox = QtWidgets.QSpinBox()
        # if it is a float, use a QDoubleSpinBox
        else:
            self.spinBox = QtWidgets.QDoubleSpinBox()
        # add the spinbox to the layout
        self.layout.addWidget(self.spinBox)

        # set the minimum of the spinbox
        if parameter.min is not None:
            self.spinBox.setMinimum(parameter.min)
        else:
            self.spinBox.setMinimum(-2 ** 16)

        # set the maximum of the spinbox
        if parameter.max is not None:
            self.spinBox.setMaximum(parameter.max)
        else:
            self.spinBox.setMaximum(2 ** 16)

        # connect the valueChanged signal of the spinbox
        self.spinBox.valueChanged.connect(self.setValue)

        # set the value of the parameter
        self.setValue(parameter.value)

        """ range selection """
        self.range_edits = []
        for name in ["min", "max"]:
            # the name of the range end, e.g. min or max
            self.layoutRange.addWidget(QtWidgets.QLabel(name))

            # the selector
            edit = QtWidgets.QLineEdit()
            # a validator for the input
            if type_int:
                edit.setValidator(QtGui.QIntValidator())
            else:
                edit.setValidator(QtGui.QDoubleValidator())
            # connect the signal
            edit.textEdited.connect(self.rangeChanged)
            # style
            edit.setMaximumWidth(50)
            edit.setPlaceholderText(name+"imum")

            # set the value if already a value is given
            if self.parameter.min is not None:
                edit.setText(str(self.parameter.min))

            # add it to the layout and the list
            self.layoutRange.addWidget(edit)
            self.range_edits.append(edit)

        # if the type is a float add a checkbox to sample from a log normal distribution for optimisation
        if not type_int:
            self.check_log = QtWidgets.QCheckBox("log-uniform")
            self.layoutRange.addWidget(self.check_log)

    def doSetValue(self, x):
        """ method, that displays the value of the parameter in the widget"""
        if x is None or np.isnan(x):
            if self.nan_checkbox is None:
                self.nan_checkbox = QtWidgets.QCheckBox("nan")
                self.layout.addWidget(self.nan_checkbox)
            self.nan_checkbox.setChecked(True)
        else:
            if self.nan_checkbox is not None:
                self.nan_checkbox.setChecked(False)
            # set the value in the spinbox
            self.spinBox.setValue(x)
            # and if existent in the slider
            if self.slider is not None:
                self.slider.setValue(x)

    def rangeChanged(self):
        if self.checkbox_fixed.isChecked():
            # set the range to invalid
            self.parameter.current_range = None
            # hide the input fields
            self.range_edits[0].setEnabled(False)
            self.range_edits[1].setEnabled(False)
        else:
            # get the input of both fields
            range_values = []
            for i in range(2):
                # try to convert them to the type (e.g. int or float)
                try:
                    range_values.append(self.parameter.type(self.range_edits[i].text()))
                # if not, the range is invalid
                except ValueError:
                    self.parameter.current_range = None
                    break
            else:
                # set the range if both ends are valid
                if self.check_log is not None and self.check_log.isChecked():
                    self.parameter.current_range = (range_values[0], range_values[1], "log-uniform")
                else:
                    self.parameter.current_range = tuple(range_values)

            # show the input fields
            self.range_edits[0].setEnabled(True)
            self.range_edits[1].setEnabled(True)


class QStringChooser(QParameterEdit):
    """ input widget to choose a string for a parameter. """

    def __init__(self, parameter, layout):
        # call the super method's initializer
        super().__init__(parameter, layout)

        # create the QLineEdit
        self.edit = QtWidgets.QLineEdit()
        self.layout.addWidget(self.edit)

        # and connect its textChange signal
        self.edit.textChanged.connect(self.setValue)

        # set the value of the parameter
        self.setValue(parameter.value)

        # strings cannot be optimized
        self.checkbox_fixed.setChecked(True)
        self.checkbox_fixed.setEnabled(False)

    def doSetValue(self, x):
        """ method, that displays the value of the parameter in the widget"""
        # set the value in the text edit
        self.edit.setText(x)


class QBoolChooser(QParameterEdit):
    """ input widget to choose a bool value for a parameter. """

    def __init__(self, parameter, layout):
        # call the super method's initializer
        super().__init__(parameter, layout)

        # create the QCheckBox
        self.check = QtWidgets.QCheckBox("")
        self.layout.addWidget(self.check)

        # and connect its stateChanged signal
        self.check.stateChanged.connect(self.setValue)

        # set the value of the parameter
        self.setValue(parameter.value)

    def doSetValue(self, x):
        """ method, that displays the value of the parameter in the widget"""
        # set the value in the check box
        self.check.setChecked(x)

    def rangeChanged(self):
        # set the range to invalid
        if self.checkbox_fixed.isChecked():
            self.parameter.current_range = None
        # or from 0 to 1
        else:
            self.parameter.current_range = (0, 1)


class QChoiceChooser(QParameterEdit):
    """ input widget to choose a category value for a parameter. """

    def __init__(self, parameter, layout):
        # call the super method's initializer
        super().__init__(parameter, layout)

        # create the QComboBox
        self.comboBox = QtWidgets.QComboBox()
        self.layout.addWidget(self.comboBox)

        # add the possible values
        for value in parameter.values:
            self.comboBox.addItem(str(value))

        # and connect its stateChanged signal
        self.comboBox.currentTextChanged.connect(self.setValue)

        # set the value of the parameter
        self.setValue(parameter.value)

    def doSetValue(self, x):
        """ method, that displays the value of the parameter in the widget"""
        # set the value in the combo box
        self.comboBox.setCurrentText(str(x))

    def rangeChanged(self):
        # set the range to invalid
        if self.checkbox_fixed.isChecked():
            self.parameter.current_range = None
        # or to the allowed values
        else:
            self.parameter.current_range = tuple(self.parameter.values)


class Parameter(object):
    """ a class the represents a parameter with its meta information."""
    parent = None
    current_range = None
    fixed = False

    def __init__(self, key, default, name=None, min=None, max=None, range=None, values=None, value_type=None,
                 desc=None):
        """
        Creates a new parameter object.

        :param key: the key which refers to the parameter. It's "script-name".
        :param default: the default value of the parameter. If not explicit type is given, the type is inferred from this value.
        :param name: the name of this parameter that is displayed, i.e. it's "display-name". Defaults to the key.
        :param min: the minimum of the parameter, for int/float parameters.
        :param max: the maximum of the parameter, for int/float parameters.
        :param range: specify a tuple of (min, max), shortcut for min and max parameters.
        :param values: the possible values, if it is a categorical parameter. These values are displayed in the ComboBox.
        :param value_type: the type of the parameter, can usually be inferred from the type of the default value.
        :param desc: a description of the parameter, is displayed as a tooltip on the input widget.
        """
        # store the key of the parameter
        self.key = key

        # store the default value
        self.default = default

        # store the name
        if name is None:
            # if no name is given, use the key as the parameter's name
            self.name = self.key
        else:
            self.name = name

        # store min and max
        self.min = min
        self.max = max
        # of if they are provided as a range, split the range to min and max
        if range is not None:
            self.min = range[0]
            self.max = range[1]

        # store the categories
        self.values = values

        # store the type of the value
        if value_type is None:
            # if no type is provided, infer it from the type of the default value
            self.type = type(default)
        else:
            self.type = value_type
        # if the value is a bool, adjust the min and max
        if self.type == bool:
            self.min = 0
            self.max = 1

        # store the description of the parameter
        self.desc = desc

        # start with the default value
        self.value = self.default

        # set the current range
        if self.values is not None:
            self.current_range = tuple(self.values)
        else:
            self.current_range = (self.min, self.max)

    def addWidget(self, layout):
        """ creates a widget to control the parameter. Optionally add it to the provided layout. """

        # if the parameter's type is categorical, use a QChoiceChooser
        if self.values is not None:
            return QChoiceChooser(self, layout)

        # if the parameter's type is an integer, use a QNumberChooser for ints
        if self.type == int:
            return QNumberChooser(self, layout, type_int=True)

        # if the parameter's type is a float, use a QNumberChooser for floats
        if self.type == float:
            return QNumberChooser(self, layout, type_int=False)

        # if the parameter's type is a string, use a QStringChooser
        if self.type == str:
            return QStringChooser(self, layout)

        # if the parameter's type is a bool, use a QBoolChooser
        if self.type == bool:
            return QBoolChooser(self, layout)

        # if it is none of the above, warn
        print("type", self.type, "not recognized")

    def valueChanged(self):
        """ callback when the parameter's value has changed (will be overloaded by the ParameterList). """
        if self.parent is not None:
            self.parent.valueChanged(self)

    def setParent(self, parent):
        self.parent = parent


class ParameterList(object):
    """ Parameters should be organized in a ParameterList, that groups them and enables dictionary-like access."""

    # the list of parameter objects
    parameters = None
    # the parameters organized in a dictionary
    parameter_dict = None
    # the list of widgets
    widgets = None

    # a list of parameters that can be optimized (e.g. are int, float, bool or categorical)
    optimizable_parameters = None

    def __init__(self, *parameters):
        # store the parameters
        self.parameters = parameters
        # initialize a new parameter dict
        self.parameter_dict = {}

        # iterate over all parameters
        for parameter in self.parameters:
            # set this ParameterList as the parent of the parameter
            parameter.setParent(self)
            # store the parameter in the parameter_dict
            self.parameter_dict[parameter.key] = parameter

    def addWidgets(self, layout):
        """ create the widgets to change the parameters of this parameter list. """

        # create a QGroupBox widget
        group = QtWidgets.QGroupBox("Parameters")
        layout.addWidget(group)
        # add a layout to the group
        layout = QtWidgets.QVBoxLayout(group)

        # create a new list of widgets
        self.widgets = {}
        # iterate over the parameters and let them add their widgets
        for parameter in self.parameters:
            self.widgets[parameter.key] = parameter.addWidget(layout)

        # return the group widget
        return group

    def __getitem__(self, key):
        """ parameter values can be accessed using dictionary-like syntax. """
        # if the ParameterList contains a parameter with this key, return it's value
        if key in self.parameter_dict:
            return self.parameter_dict[key].value

    def __setitem__(self, item, value):
        """ parameter values can be set using dictionary-like syntax. """
        # if the ParameterList contains a parameter with this key, set it's value
        if item in self.parameter_dict:
            # set the value
            self.parameter_dict[item].value = value
            # notify the widget
            if self.widgets is not None:
                self.widgets[item].setValue(value)
            return True

    def keys(self):
        """ return all keys """
        return self.parameter_dict.keys()

    def valueChanged(self, parameter):
        """ """
        self.valueChangedEvent()

    def valueChangedEvent(self):
        pass

    def displayRanges(self, do_display):
        for parameter in self.parameters:
            self.widgets[parameter.key].displayRanges(do_display)

    def getRanges(self):
        """ return a list of tuples indicating the ranges of all parameters. """
        # remember which parameters have been set
        self.optimizable_parameters = []

        # iterate over all parameters
        ranges = []
        for parameter in self.parameters:
            # if th parameter has a min and max set, we can give a range
            if parameter.current_range is not None:
                # add the range and add it to the optimizable parameters
                ranges.append(parameter.current_range)
                self.optimizable_parameters.append(parameter)

        # return all the ranges
        return ranges

    def setOptimisationValues(self, values, update_widgets=False):
        """ set the values for all optimizable parameters."""
        # iterate over all optimizable parameters and the provided values
        for parameter, value in zip(self.optimizable_parameters, values):
            # set the parameter with the given value
            parameter.value = value
            # and if desired, update the widgets
            if update_widgets and self.widgets is not None:
                self.widgets[parameter.key].setValue(value)

    def setValues(self, values, update_widgets=False):
        """ set the values for all parameters. Either from a list of from a dictionary. """

        # if the function is given a dictionary, iterate over the keys
        if isinstance(values, dict):
            for key in values:
                # set the parameter with the given value
                self.parameter_dict[key].value = values[key]
                # and if desired, update the widgets
                if update_widgets and self.widgets is not None:
                    self.widgets[parameter.key].setValue(values[key])
        # if not, iterate over all parameters
        else:
            for parameter, value in zip(self.parameters, values):
                # set the parameter with the given value
                parameter.value = value
                # and if desired, update the widgets
                if update_widgets and self.widgets is not None:
                    self.widgets[parameter.key].setValue(value)

    def updateWidgets(self):
        """ update the parameter values in the widget displays """
        for parameter in self.parameters:
            self.widgets[parameter.key].setValue(parameter.value)
