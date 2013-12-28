""" Engine classes for parallel support """

import abc
from pandas.core import config

try:
    import joblib
    _JOBLIB_INSTALLED = True
except:
    _JOBLIB_INSTALLED = False

class ParallelException(ValueError): pass

def create_parallel_engine(name=None, **kwargs):
    """
    Parameters
    ----------
    name : engine name, default for None is options.parallel.engine
    pass thru kwargs to engine creation

    Returns
    -------
    an engine instance or None
    """

    # passed in an engine
    if isinstance(name, ParallelEngine):
        return name

    # try the option
    if name is None:
        name = config.get_option('parallel.default_engine')

    if name is None:
        return None
    try:
        return _engines[name](**kwargs)
    except (KeyError):
        raise ParallelException("cannot find engine [{0}]".format(name))

class ParallelEngine(object):
    """Object serving as a base class for all engines."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, force=False, max_cpu=None, **kwargs):
        self.force = force
        self.max_cpu = max_cpu
        self.is_installed = self.install()

    def install(self):
        return False

    def can_evaluate(self, obj):
        """ return True if we can evaluate the object """
        return True

class JoblibEngine(ParallelEngine):
    """ joblib engine class """

    def install(self):
        return _JOBLIB_INSTALLED

#### engines & options ####

_engines = { 'joblib' : JoblibEngine }

default_engine_doc = """
: string/None
    default engine for parallel operations
"""

with config.config_prefix('parallel'):
    config.register_option(
        'default_engine', 'joblib', default_engine_doc,
        validator=config.is_one_of_factory(_engines.keys() + [ None])
    )

