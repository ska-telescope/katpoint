################################################################################
# Copyright (c) 2009-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Model base class.

This provides a base class for pointing and delay models, handling the loading,
saving and display of parameters.
"""

import configparser
from collections import OrderedDict

import numpy as np


class Parameter:
    """Generic model parameter.

    This represents a single model parameter, bundling together its attributes
    and enabling it to be read from a string and output to a string by getting
    and setting the :attr:`value_str` property.

    Parameters
    ----------
    name : string
        Parameter name
    units : string
        Physical unit of parameter value
    doc : string
        Documentation string describing parameter
    from_str : function, signature float = f(string), optional
        Conversion function to extract parameter from string
    to_str : function, signature string = f(float), optional
        Conversion function to express parameter as string
    value : float, optional
        Parameter value (*default_value* by default, of course)
    default_value : float, optional
        Parameter default value

    Attributes
    ----------
    value_str
    """

    def __init__(
        self,
        name,
        units,
        doc,
        from_str=float,
        to_str=str,
        value=None,
        default_value=0.0,
    ):
        self.name = name
        self.units = units
        self.__doc__ = doc
        # These functions are underscored to encourage use of value_str instead
        self._from_str = from_str
        self._to_str = to_str
        self.value = value if value is not None else default_value
        self.default_value = default_value

    def __bool__(self):
        """Return True if parameter is active, i.e. its value differs from default."""
        # Do explicit cast to bool, as value can be a NumPy type, resulting in
        # an np.bool_ type for the expression (not allowed for __bool__)
        return bool(self.value != self.default_value)

    @property
    def value_str(self):
        """String form of parameter value used to convert it to/from a string."""
        return self._to_str(self.value)

    @value_str.setter
    def value_str(self, valstr):
        self.value = self._from_str(valstr)

    def __repr__(self):
        """Short human-friendly string representation of parameter object."""
        return (
            f"<katpoint.Parameter {self.name} = {self.value_str} {self.units} "
            f"at {id(self):#x}>"
        )


class BadModelFile(Exception):
    """Unable to load model from config file (unrecognised format)."""


class Model:
    """Base class for models (e.g. pointing and delay models).

    The base class handles the construction / loading, saving, display and
    comparison of models. A Model consists of a sequence of Parameters and
    an optional header dict. A number of these parameters may be *active*,
    i.e. not equal to their default values.

    Models can be constructed from description strings (:meth:`fromstring`),
    sequences of parameter values (:meth:`fromlist`), configuration files
    (:meth:`fromfile`) or other similar models. The :meth:`set` method
    automatically picks the correct constructor based on the input.

    Parameter names and values may be accessed and modified via a dict-like
    interface mapping names to values.

    Parameters
    ----------
    params : sequence of :class:`Parameter` objects
        Full set of model parameters in the expected order
    """

    def __init__(self, params):
        self.header = {}
        self.params = OrderedDict((p.name, p) for p in params)

    def __len__(self):
        """Return number of parameters in full model."""
        return len(self.params)

    def __bool__(self):
        """Return True if model contains any active (non-default) parameters."""
        return any(p for p in self)

    def __iter__(self):
        """Iterate over parameter objects."""
        return self.params.values().__iter__()

    def param_strs(self):
        """Justified (name, value, units, doc) strings for active parameters."""
        name_len = max(len(p.name) for p in self)
        value_len = max(len(p.value_str) for p in self.params.values())
        units_len = max(len(p.units) for p in self.params.values())
        return [
            (
                p.name.ljust(name_len),
                p.value_str.ljust(value_len),
                p.units.ljust(units_len),
                p.__doc__,
            )
            for p in self.params.values()
            if p
        ]

    def __repr__(self):
        """Short human-friendly string representation of model object."""
        class_name = self.__class__.__name__
        num_active = len([p for p in self if p])
        return (
            f"<katpoint.{class_name} active_params={num_active}/{len(self)} "
            f"at {id(self):#x}>"
        )

    def __str__(self):
        """Verbose human-friendly string representation of model object."""
        class_name = self.__class__.__name__
        num_active = len([p for p in self if p])
        summary = (
            f"{class_name} has {len(self)} parameters with "
            f"{num_active} active (non-default)"
        )
        if num_active == 0:
            return summary
        param_strs = "\n".join(
            # pylint: disable=consider-using-f-string
            "{} = {} {} ({})".format(*ps)
            for ps in self.param_strs()
        )
        return f"{summary}:\n{param_strs}"

    def __eq__(self, other):
        """Equality comparison operator (parameter values only)."""
        return self.description == (
            other.description if isinstance(other, self.__class__) else other
        )

    def __hash__(self):
        """Compute hash on description string, just like equality operator."""
        return hash(self.description)

    def __getitem__(self, key):
        """Access parameter value by name."""
        return self.params[key].value

    def __setitem__(self, key, value):
        """Modify parameter value by name."""
        self.params[key].value = value

    def keys(self):
        """List of parameter names in the expected order."""
        return self.params.keys()

    def values(self):
        """List of parameter values in the expected order ('tolist')."""
        return [p.value for p in self]

    def fromlist(self, floats):
        """Load model from sequence of floats."""
        self.header = {}
        params = list(self)
        min_len = min(len(params), len(floats))
        for param, value in zip(params[:min_len], floats[:min_len]):
            param.value = value
        for param in params[min_len:]:
            param.value = param.default_value

    @property
    def description(self):
        """Compact but complete string representation ('tostring')."""
        active = np.nonzero([bool(p) for p in self])[0]
        last_active = active[-1] if len(active) > 0 else -1
        return " ".join([p.value_str for p in self][: last_active + 1])

    def fromstring(self, description):
        """Load model from description string (parameters only)."""
        self.header = {}
        # Split string either on commas or whitespace, for good measure
        param_vals = (
            [p.strip() for p in description.split(",")]
            if "," in description
            else description.split()
        )
        params = list(self)
        min_len = min(len(params), len(param_vals))
        for param, param_val in zip(params[:min_len], param_vals[:min_len]):
            param.value_str = param_val
        for param in params[min_len:]:
            param.value = param.default_value

    def tofile(self, file_like):
        """Save model to config file (both header and parameters).

        Parameters
        ----------
        file_like : object
            File-like object with write() method representing config file
        """
        cfg = configparser.ConfigParser()
        cfg.add_section("header")
        for key, val in self.header.items():
            cfg.set("header", key, str(val))
        cfg.add_section("params")
        for param_str in self.param_strs():
            # pylint: disable=consider-using-f-string
            cfg.set("params", param_str[0], "{} ; {} ({})".format(*param_str[1:]))
        cfg.write(file_like)

    def fromfile(self, file_like):
        """Load model from config file (both header and parameters).

        Parameters
        ----------
        file_like : object
            File-like object with readline() method representing config file

        Raises
        ------
        BadModelFile
            If `file_like` could not be read or parsed
        """
        # pylint: disable=protected-access
        defaults = {p.name: p._to_str(p.default_value) for p in self}
        cfg = configparser.ConfigParser(defaults, inline_comment_prefixes=(";", "#"))
        try:
            cfg.read_file(file_like)
            if cfg.sections() != ["header", "params"]:
                raise configparser.Error("Expected sections not found in model file")
        except configparser.Error as exc:
            filename = getattr(file_like, "name", "")
            input_descr = f"file {filename!r}" if filename else "file-like object"
            msg = f"Could not construct {self.__class__.__name__} from {input_descr}"
            raise BadModelFile(msg) from exc
        self.header = dict(cfg.items("header"))
        for param in defaults:
            self.header.pop(param.lower())
        for param in self:
            param.value_str = cfg.get("params", param.name)

    def set(self, model=None):
        """Load parameter values from the appropriate source.

        Parameters
        ----------
        model : file-like or model object, sequence of floats, or string, optional
            Model specification. If this is a file-like or model object, load
            the model from it (including header). If this is a sequence of
            floats, accept it directly as the model parameters. If it is a
            string, interpret it as a comma-separated (or whitespace-
            separated) sequence of parameters in their string form (i.e. a
            description string). The default is an empty model.

        Raises
        ------
        BadModelFile
            If `model` is of incorrect type
        """
        if isinstance(model, Model):
            if not isinstance(model, type(self)):
                raise BadModelFile(
                    f"Cannot construct a '{self.__class__.__name__}' "
                    f"from a '{model.__class__.__name__}'"
                )
            self.fromlist(model.values())
            self.header = dict(model.header)
        elif isinstance(model, str):
            self.fromstring(model)
        else:
            array = np.atleast_1d(model)
            if array.dtype.kind in "iuf" and array.ndim == 1:
                self.fromlist(model)
            elif model is not None:
                self.fromfile(model)
            else:
                self.fromlist([])
