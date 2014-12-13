# coding=utf-8
# pylint: disable-msg=E1101,W0612

import nose

import numpy as np
import pandas as pd

from pandas import (Index, Series, DataFrame, isnull, notnull, bdate_range,
                    date_range, period_range, timedelta_range, MultiIndex)

from pandas.compat import StringIO, lrange, range, zip, u, OrderedDict, long
from pandas.core.common import cast_to_na, as_nd
from pandas.core import common as com
from pandas import compat, set_option, lib
from pandas.util.testing import (assert_series_equal,
                                 assert_almost_equal,
                                 assert_frame_equal,
                                 assert_numpy_array_equal,
                                 array_equivalent,
                                 ensure_clean)
import pandas.util.testing as tm

set_option('support.dynd',True)
if not com._DYND:
    raise nose.SkipTest("dynd not installed")

import dynd
from dynd import ndt, nd
import datashape
from datashape import dshape

class TestDynd(tm.TestCase):
    """ test certain attributes/method on nd arrays directly """

    _multiprocess_can_split_ = False


    def test_indexing(self):

        t = nd.array([1,2,3])
        assert_numpy_array_equal(t, t)

        # take currently requires int64 for indexers
        indexer = nd.array([1,2]).ucast('int64').eval()
        result = nd.take(t, indexer)
        assert_numpy_array_equal(result, nd.array([2,3]))

        result = t[[1,2]]
        assert_numpy_array_equal(result, nd.array([2,3]))

        result = t[[False,True,True]]
        assert_numpy_array_equal(result, nd.array([2,3]))

    def test_types(self):

        t = ndt.type('?int32')

        ds = dshape(t.dshape)
        ty = ds.measure.ty

        # to numpy dtype
        self.assertTrue(com.to_numpy_dtype(t) == np.dtype('int32'))

        # type testing
        self.assertTrue(datashape.isnumeric(ty))

        # type testing
        from datashape import integral, floating
        self.assertTrue(ty in integral.types)
        self.assertFalse(ty in floating.types)

    def test_types_void(self):

        t = ndt.type('?void')

        # to numpy dtype
        self.assertTrue(com.to_numpy_dtype(t) == np.dtype('object'))

    def test_type_conversions(self):

        t = ndt.type('3 * int64')
        self.assertEqual(cast_to_na(t),ndt.type('3 * ?int64'))
        t = ndt.type('int64')
        self.assertEqual(cast_to_na(t),ndt.type('?int64'))
        t = ndt.type('?int64')
        self.assertEqual(cast_to_na(t),ndt.type('?int64'))
        t = ndt.type('3 * int32')
        self.assertEqual(cast_to_na(t),ndt.type('3 * ?int32'))
        t = ndt.type('3 * ?int32')
        self.assertEqual(cast_to_na(t),ndt.type('3 * ?int32'))

        # opportunistic
        t = ndt.type('3 * float32')
        self.assertEqual(cast_to_na(t),ndt.type('3 * ?int32'))

    def test_as_nd(self):

        self.assertEqual(as_nd(np.array([1,2,3],dtype='int64')).dtype,
                         nd.array([1,2,3],type='3 * ?int64').dtype)
        self.assertEqual(as_nd(np.array([1.,2.,3.],dtype='float64')).dtype,
                         nd.array([1,2,3],type='3 * ?int64').dtype)
        self.assertEqual(as_nd(np.array([1.,np.nan,3.],dtype='float64')).dtype,
                         nd.array([1,2,3],type='3 * ?int64').dtype)

        # no conversion
        self.assertEqual(as_nd(np.array([1.,np.nan,3.5],dtype='float64')).dtype,
                         np.array([1,2,3],dtype='float64').dtype)

    def test_as_numpy(self):

        expected = np.array([1,2,3],dtype='int32')
        arr = nd.array([1,2,3],dtype='int32')
        result = com.as_numpy(arr, errors='raise')
        assert_numpy_array_equal(result, expected)
        result = com.as_numpy(arr, errors='coerce')
        assert_numpy_array_equal(result, expected)

        expected = nd.array([1,np.iinfo('int32').min,3],dtype='int32')
        arr = nd.array([1,None,3],dtype='?int32')
        self.assertRaises(TypeError, lambda : com.as_numpy(arr, errors='raise'))
        result = com.as_numpy(arr, errors='coerce')
        assert_numpy_array_equal(result, expected)

    def test_isnull(self):

        expected = np.array([False, True, False])

        result = com.isnull_compat(np.array([1, np.nan, 1]))
        assert_numpy_array_equal(result, expected)

        result = com.isnull_compat(Index([1, np.nan, 1]))
        assert_numpy_array_equal(result, expected)

        result = com.isnull_compat(Series([1, np.nan, 1]))
        assert_numpy_array_equal(result, expected)

class TestBasic(tm.TestCase):
    _multiprocess_can_split_ = False

    def test_config_option(self):

        set_option('support.dynd',False)
        set_option('support.dynd',True)

    def test_api_compat(self):

        arr = nd.array([1,2,3],type='3 * ?int32')
        self.assertTrue(arr.dtype == ndt.type('?int32'))
        self.assertTrue(arr.ndim == 1)

        set_option('support.dynd',False)
        self.assertFalse(com.is_ndt_type(ndt.type('int32')))
        set_option('support.dynd',True)
        self.assertTrue(com.is_ndt_type(ndt.type('int32')))

    def test_formatting(self):

        s = Series([1,2,3])
        result = str(s)
        expected = "0   1\n1   2\n2   3\ndtype: int64"
        self.assertEqual(result, expected)

        s = Series([1,None,3])
        result = str(s)
        expected = "0     1\n1   NaN\n2     3\ndtype: int64"
        self.assertEqual(result, expected)

    def test_na(self):

        s = Series([1,2,3])
        result = s.isnull()
        tm.assert_series_equal(result, Series(False,index=s.index))

        s = Series([1,None,3])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        s = Series([1,None,3.])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        s = Series([1,None,3.5])
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        #### FIXME: this is technically a raise_cast_failure
        # but can handle now
        s = Series([1,np.nan,3],dtype='int64')
        result = s.isnull()
        tm.assert_series_equal(result, Series([False,True,False],index=s.index))

        s = nd.array([1,2,3])
        result = pd.isnull(s)
        assert_numpy_array_equal(result, nd.array([False, False, False]))

        s = nd.array([1,None,3])
        result = pd.isnull(s)
        assert_numpy_array_equal(result, nd.array([False, True, False]))


    def test_construction(self):

        s = Series([1,2,3])
        self.assertTrue(s.dtype == ndt.type('?int64'))

        s = Series([1,None,3])
        self.assertTrue(s.dtype == ndt.type('?int64'))

        s = Series([1,2,3],dtype='int32')
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([1,2,3],dtype='?int32')
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([1,2,3],dtype=np.dtype('int32'))
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([1,2,3],dtype=ndt.type('int32'))
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([1,2,3],dtype=ndt.type('?int32'))
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([],dtype='int32')
        self.assertTrue(s.dtype == ndt.type('?int32'))

        s = Series([1,None,3.])
        self.assertTrue(s.dtype == ndt.type('?int64'))

        s = Series([1,None,3.5])
        self.assertTrue(s.dtype == np.dtype('float64'))

        # we should not be converting bools
        s = Series([True,True,True])
        self.assertTrue(s.dtype == np.dtype('bool'))

        s = Series([np.nan, np.nan],dtype='int64',index=[1,2])
        self.assertTrue(s.dtype == ndt.type('?int64'))

    def test_construction_array(self):

        # construct from nd.array types
        # with various dtypes

        arr = nd.array([1,2,3])
        s = Series(arr)
        expected = Series([1,2,3],dtype='int32')
        assert_series_equal(s, expected)

        arr = nd.array([1,None,3])
        s = Series(arr)
        expected = Series([1,None,3],dtype='?int32')
        assert_series_equal(s, expected)

        arr = np.array([1,2,3],dtype='int64')
        s = Series(arr)
        expected = Series([1,2,3],dtype='int64')
        assert_series_equal(s, expected)

        # we need to cast these back to our np types
        arr = nd.array([1,2,3],dtype='?float64')
        s = Series(arr)
        expected = Series([1,2,3],dtype='int64')
        assert_series_equal(s, expected)

        s = Series(arr, dtype='float64')
        expected = Series([1,2,3],dtype='float64')
        assert_series_equal(s, expected)

    def test_construction_edge(self):

        # this would normally be ?void type, but we are explicity casting it
        s = Series([None, None],dtype='int64',index=[1,2])
        self.assertTrue(s.dtype == ndt.type('?int64'))

        s = Series([None, None],dtype='?int64',index=[1,2])
        self.assertTrue(s.dtype == ndt.type('?int64'))

    def test_internal_accessors(self):

        s = Series([1,2,3])
        result = s.get_values()
        expected = nd.array([1,2,3],dtype='?int64')
        assert_numpy_array_equal(result, expected)

    def test_indexing_get(self):

        # scalar
        s = Series([1,2,3])
        result = s.iloc[1]
        self.assertEqual(result,2)

        result = s[1]
        self.assertEqual(result,2)

        s = Series([1,np.nan,3])
        result = s.iloc[1]
        self.assertTrue(result is np.nan)

        result = s[1]
        self.assertTrue(result is np.nan)

        # slice
        s = Series([1,np.nan,3])
        expected = Series([np.nan,3],index=[1,2])
        result = s[1:3]
        assert_series_equal(result, expected)

        # boolean
        result = s[[False,True,True]]
        assert_series_equal(result, expected)

    def test_indexing_comparison_get(self):
        raise nose.SkipTest('comparison ops not working yet!')

        import pdb; pdb.set_trace()
        s = Series([1,2,3])
        result = s[s>1]

    def test_indexing_set(self):

        # scalar with coercion
        s = Series([1,2,3])
        s.iloc[1] = 4
        result = s.iloc[1]

        self.assertTrue(result is 4)
        s = Series([1,2,3])
        s.iloc[1] = np.nan
        result = s.iloc[1]
        self.assertTrue(result is np.nan)

        expected = Series([4,4],index=[1,2])
        s = Series([1,2,3])
        s.iloc[1:3] = 4
        result = s.iloc[1:3]
        assert_series_equal(result, expected)

        expected = Series([1, np.nan, np.nan])
        s = Series([1,2,3])
        s.iloc[1:3] = np.nan
        assert_series_equal(s, expected)

    def test_indexing_set_fancy(self):

        s = Series([1,2,3])
        s.iloc[[False,True,True]] = np.nan
        assert_series_equal(s, expected)

        # array with coercion
        expected = Series([1, np.nan, 1])
        s = Series([1,2,3])
        s.iloc[1:3] = [np.nan,1]
        assert_series_equal(s, expected)

        s = Series([1,2,3])
        s.iloc[[False,True,True]] = [np.nan, 1]
        assert_series_equal(s, expected)

    def test_ops_with_others(self):

        # arithmetic with scalars / dynd / numpy arrays
        s = Series([1,None,3])
        result = s + 1
        expected = Series([2,None,4])
        assert_series_equal(result, expected)

        s = Series([1,None,3])
        result = s + np.array([1,1,1])
        expected = Series([2,np.nan,4])
        assert_series_equal(result, expected)

        s = Series([1,None,3])
        result = s + nd.array([1,1,1])
        expected = Series([2,np.nan,4])
        assert_series_equal(result, expected)

        # type conversion
        s = Series([1,2,3])
        result = s + 1.0
        expected = Series([2,3,4],dtype='int64')
        assert_series_equal(result, expected)

        result = s + 1.5
        expected = Series([2.5,3.5,4.5])
        assert_series_equal(result, expected)

        s = Series([1,2,3])
        result = s + np.nan
        expected = Series([np.nan,np.nan,np.nan],dtype='int64')
        assert_series_equal(result, expected)

        # TODO: broken ATM in dynd
        # https://github.com/libdynd/dynd-python/issues/367
        #s = Series([1,2,3])
        #result = s + None
        #expected = Series([np.nan,np.nan,np.nan],dtype='int64')
        #assert_series_equal(result, expected)

        s = Series([1,2,3])
        result = s + Series([0,np.nan,0])
        expected = Series([1,np.nan,3],dtype='?int64')
        assert_series_equal(result, expected)

        s = Series([1,None,3])
        result = s + 1.0
        expected = Series([2,np.nan,4],dtype='int64')
        assert_series_equal(result, expected)


    def test_ops(self):

        s = Series([1,2,3])
        result = s.sum()
        self.assertEqual(result, 6)

        s = Series([1,None,3])
        result = s.sum()
        self.assertEqual(result, 4)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
