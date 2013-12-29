#!/usr/bin/env python

import nose
from nose.tools import assert_raises, assert_true, assert_false, assert_equal

from numpy.random import randn, rand, randint
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy.testing.decorators import slow

import pandas as pd
from pandas.core import common as com
from pandas.core import config
from pandas import DataFrame, Series, Panel, date_range
from pandas.util.testing import makeCustomDataframe as mkdf

import pandas.util.testing as tm
from pandas.util.testing import (assert_frame_equal, randbool,
                                 assertRaisesRegexp,
                                 assert_produces_warning, assert_series_equal)
from pandas.compat import PY3, u

from pandas.parallel import engines as pp

class Base(tm.TestCase):
    engine = None

    def setUp(self):
        self.frame = DataFrame(np.random.rand(100,10))

    def tearDown(self):
        pass

    def test_create_engine(self):

        # pass default or None
        name = self.engine
        engine = pd.create_parallel_engine(name=name)
        self.assert_(engine is not (name is not None))

        # no default and passed None
        with config.option_context('parallel.default_engine',None):
            engine = pd.create_parallel_engine(name=None)
            self.assert_(engine is None)

            # invalid
            self.assertRaises(pp.ParallelException, lambda : pd.create_parallel_engine(name='foo'))

    def test_apply(self):

        # base case
        expected = self.frame.apply(np.sqrt, engine=pd.create_parallel_engine(name='python'))

        # since these are ufuncs, no effect here
        for proc in [1,2,4,None]:
            result = self.frame.apply(np.sqrt, engine=pd.create_parallel_engine(name=self.engine,max_cpu=proc))
            assert_frame_equal(result,expected)

        # real func
        def f(x):
            return np.sqrt(x)

        for proc in [1,2,4,None]:

            # too small to execute
            result = self.frame.apply(f, engine=pd.create_parallel_engine(name=self.engine,force=False,max_cpu=proc))
            assert_frame_equal(result,expected)

            # parallel execution
            result = self.frame.apply(f, engine=pd.create_parallel_engine(name=self.engine,force=True,max_cpu=proc))
            assert_frame_equal(result,expected)

            # anonymous function
            result = self.frame.apply(lambda x: np.sqrt(x),
                                      engine=pd.create_parallel_engine(name=self.engine,force=True,max_cpu=proc))
            assert_frame_equal(result,expected)


class TestJoblib(Base):

    @classmethod
    def setUpClass(cls):
        super(TestJoblib, cls).setUpClass()
        if not pp._JOBLIB_INSTALLED:
            raise nose.SkipTest("no joblib installed")
        cls.engine = 'joblib'

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
