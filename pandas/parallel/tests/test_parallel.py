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

    @classmethod
    def tearDownClass(cls):
        super(Base, cls).tearDownClass()
        if getattr(cls,'engine',None):
            del cls.engine

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_create_engine(self):

        # pass default or None
        name = getattr(self,'engine',None)
        engine = pd.create_parallel_engine(name=name)
        self.assert_(engine is not (name is not None))

        # no default and passed None
        with config.option_context('parallel.default_engine',None):
            engine = pd.create_parallel_engine(name=None)
            self.assert_(engine is None)

            # invalid
            self.assertRaises(pp.ParallelException, lambda : pd.create_parallel_engine(name='foo'))

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
