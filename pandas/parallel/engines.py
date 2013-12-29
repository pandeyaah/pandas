""" Engine classes for parallel support """

import abc
import numpy as np
from pandas.core import config

try:
    import joblib
    from joblib import Parallel, delayed
    _JOBLIB_INSTALLED = True
except:
    _JOBLIB_INSTALLED = False

class ParallelException(ValueError): pass

# use module level variables to set the function object
# works even if its an anonymous function
_target_function_ref = None

def _target_function(x):
    global _target_function_ref
    return _target_function_ref(x)

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

    def __init__(self, force=False, max_cpu=None, verbose=0, **kwargs):
        self.force = force

        if max_cpu is None:
            max_cpu = -1
        self.max_cpu = max_cpu
        self.verbose = verbose
        self.is_installed = self.install()

    def install(self):
        return False

    def can_evaluate(self, obj):
        """ return True if we can evaluate the object """
        return False

class PythonEngine(ParallelEngine):
    """ the python engine - use a single proc """

    def install(self):
        return True

    def can_evaluate(self, obj):
        """ we never evaluate """
        return False

class JoblibEngine(ParallelEngine):
    """ joblib engine class """

    def install(self):
        return _JOBLIB_INSTALLED

    def can_evaluate(self, obj):
        """ we never evaluate """
        return self.force or np.prod(obj.shape) > 20000

    def apply_frame(self,obj,func,axis):
        """ parallel apply for data frame """

        # split according to each cpu
        n_jobs = self.max_cpu
        if n_jobs < 0:
            n_jobs += joblib.cpu_count() + 1

        if axis == 0:
            l = len(obj.columns)
            perc = (len(obj.columns) // n_jobs) + 1
            gen = (obj.iloc[:,(i*perc):min(((i+1)*perc),l)] for i in range(n_jobs) if i*perc < l)

            def f(x):
                return [ func(x.icol(i)) for i in range(len(x.columns)) ]

        elif axis == 1:
            values = obj.values
            gen = (Series.from_array(arr, index=obj.columns, name=name)
                   for i, (arr, name) in
                   enumerate(zip(values, obj.index)))

            f = func

        else:  # pragma : no cover
            raise AssertionError('Axis must be 0 or 1, got %s' % str(axis))

        # set the function to a module level reference in order to be useable by the targets
        global _target_function_ref
        _target_function_ref = f
        p = Parallel(n_jobs=self.max_cpu,verbose=self.verbose,max_nbytes=None)

        # this returns a list of a list of the results
        results = p(delayed(_target_function)(v) for i, v in enumerate(gen))

        count = 0
        d = dict()
        for r in results:
            for sr in r:
                d[count] = sr
                count += 1

        return d

#### engines & options ####

_engines = { 'joblib' : JoblibEngine, 'python' : PythonEngine }

default_engine_doc = """
: string/None
    default engine for parallel operations
"""

with config.config_prefix('parallel'):
    config.register_option(
        'default_engine', None, default_engine_doc,
        validator=config.is_one_of_factory(list(_engines.keys()) + [ None])
    )

