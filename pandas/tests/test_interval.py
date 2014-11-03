import numpy as np

from pandas.core.interval import Interval, IntervalIndex
from pandas.core.index import Index

import pandas.util.testing as tm
import pandas as pd


class TestInterval(tm.TestCase):
    def setUp(self):
        self.interval = Interval(0, 1)

    def test_properties(self):
        self.assertEqual(self.interval.closed, 'right')
        self.assertEqual(self.interval.left, 0)
        self.assertEqual(self.interval.right, 1)
        self.assertEqual(self.interval.mid, 0.5)

    def test_repr(self):
        self.assertEqual(repr(self.interval),
                         "Interval(0, 1, closed='right')")
        self.assertEqual(str(self.interval), "(0, 1]")

        interval_left = Interval(0, 1, closed='left')
        self.assertEqual(repr(interval_left),
                         "Interval(0, 1, closed='left')")
        self.assertEqual(str(interval_left), "[0, 1)")

    def test_equal(self):
        self.assertEqual(Interval(0, 1), Interval(0, 1, closed='right'))
        self.assertNotEqual(Interval(0, 1), Interval(0, 1, closed='left'))

    def test_comparison(self):
        self.assertLess(Interval(0, 1), 2)
        self.assertLess(Interval(0, 1, closed='left'), 1)
        self.assertLess(Interval(0, 1), Interval(2, 3))
        self.assertLess(Interval(0, 1), Interval(1, 2))

        self.assertFalse(Interval(0, 1) < 1)
        self.assertFalse(Interval(0, 1) < Interval(1, 2, closed='left'))

    def test_hash(self):
        # should not raise
        hash(self.interval)

    def test_math(self):
        expected = Interval(1, 2)
        actual = self.interval + 1
        self.assertEqual(expected, actual)


class TestIntervalIndex(tm.TestCase):
    def setUp(self):
        self.index = IntervalIndex([0, 1], [1, 2])

    def test_constructors(self):
        expected = self.index
        actual = IntervalIndex.from_breaks(np.arange(3), closed='right')
        self.assertTrue(expected.equals(actual))

        alternate = IntervalIndex.from_breaks(np.arange(3), closed='left')
        self.assertFalse(expected.equals(alternate))

        actual = IntervalIndex.from_intervals([Interval(0, 1), Interval(1, 2)])
        self.assertTrue(expected.equals(actual))

        self.assertRaises(ValueError, IntervalIndex, [0], [1], closed='invalid')

        # TODO: fix all these commented out tests (here and below)

        # intervals = [Interval(0, 1), Interval(1, 2, closed='left')]
        # with self.assertRaises(ValueError):
        #     IntervalIndex.from_intervals(intervals)

        # actual = Index([Interval(0, 1), Interval(1, 2)])
        # self.assertTrue(expected.equals(actual))

        # no point in nesting periods in an IntervalIndex
        # self.assertRaises(ValueError, IntervalIndex.from_breaks,
        #                   pd.period_range('2000-01-01', periods=3))

    def test_properties(self):
        self.assertEqual(len(self.index), 2)
        self.assertEqual(self.index.size, 2)

        self.assert_numpy_array_equal(self.index.mid, [0.5, 1.5])
        self.assert_numpy_array_equal(self.index.left, [0, 1])
        self.assert_numpy_array_equal(self.index.right, [1, 2])

        self.assertEqual(self.index.closed, 'right')

        expected = np.array([Interval(0, 1), Interval(1, 2)], dtype=object)
        self.assert_numpy_array_equal(np.asarray(self.index), expected)
        self.assert_numpy_array_equal(self.index.values, expected)

    def test_monotonic_and_unique(self):
        self.assertTrue(self.index.is_monotonic)
        self.assertTrue(self.index.is_unique)

        idx = IntervalIndex([0, 2], [1, 3])
        self.assertFalse(idx.is_monotonic)

        idx = IntervalIndex([0, 1], [1, 2], closed='both')
        self.assertFalse(idx.is_monotonic)

        idx = IntervalIndex([0, 2], [0, 2])
        self.assertFalse(self.index.is_unique)

    def test_repr(self):
        expected = ("<class 'pandas.core.interval.IntervalIndex'>\n"
                    "(0, 1]\n(1, 2]\nLength: 2, Closed: 'right'")
        IntervalIndex((0, 1), (1, 2), closed='right')
        self.assertEqual(repr(self.index), expected)

    def test_get_loc_value(self):
        self.assertRaises(KeyError, self.index.get_loc, 0)
        self.assertEqual(self.index.get_loc(0.5), 0)
        self.assertEqual(self.index.get_loc(1), 0)
        self.assertEqual(self.index.get_loc(1.5), 1)
        self.assertEqual(self.index.get_loc(2), 1)
        self.assertRaises(KeyError, self.index.get_loc, -1)
        self.assertRaises(KeyError, self.index.get_loc, 3)

        idx = IntervalIndex([0, 1], [2, 3])
        self.assertEqual(idx.get_loc(0.5), 0)
        self.assertEqual(idx.get_loc(1), 0)
        self.assertEqual(idx.get_loc(1.5), slice(0, 2))
        self.assertEqual(idx.get_loc(2), slice(0, 2))
        self.assertEqual(idx.get_loc(3), 1)
        self.assertRaises(KeyError, idx.get_loc, 3.5)

        idx = IntervalIndex([0, 2], [1, 3])
        self.assertRaises(KeyError, idx.get_loc, 1.5)

    def test_get_loc_interval(self):
        self.assertEqual(self.index.get_loc(Interval(0, 1)), 0)
        self.assertRaises(KeyError, self.index.get_loc,
                          Interval(0, 1, 'left'))
        # self.assertEqual(self.index.get_loc(Interval(0, 0.5)), 0)

    def test_get_indexer(self):
        actual = self.index.get_indexer([-1, 0, 0.5, 1, 1.5, 2, 3])
        expected = [-1, -1, 0, 0, 1, 1, -1]
        self.assert_numpy_array_equal(actual, expected)

        actual = self.index.get_indexer(self.index)
        expected = [0, 1]
        self.assert_numpy_array_equal(actual, expected)

        index = IntervalIndex.from_breaks([0, 1, 2], closed='left')
        actual = index.get_indexer([-1, 0, 0.5, 1, 1.5, 2, 3])
        expected = [-1, 0, 0, 1, 1, -1, -1]
        self.assert_numpy_array_equal(actual, expected)

        # verify that closed='left' and closed='right' cannot be interchanged
        actual = self.index.get_indexer(index)
        expected = [-1, -1]
        self.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_subintervals(self):
        # return indexers for wholly contained subintervals
        target = IntervalIndex.from_breaks(np.linspace(0, 2, 5))
        actual = self.index.get_indexer(target)
        expected = [0, 0, 1, 1]
        self.assert_numpy_array_equal(actual, expected)

        target = IntervalIndex.from_breaks([0, 0.67, 1.33, 2])
        actual = self.index.get_indexer(target)
        expected = [0, -1, 1]
        self.assert_numpy_array_equal(actual, expected)

        # optional, but would be nice to have
        target = IntervalIndex.from_breaks([0, 0.33, 0.67, 1], closed='left')
        actual = self.index.get_indexer(target)
        expected = [-1, 0, -1]
        self.assert_numpy_array_equal(actual, expected)

    def test_contains(self):
        self.assertNotIn(0, self.index)
        self.assertIn(0.5, self.index)
        self.assertIn(2, self.index)

        self.assertIn(Interval(0, 1), self.index)
        self.assertNotIn(Interval(0, 2), self.index)
        # self.assertIn(Interval(0, 0.5), self.index)

    def test_non_contiguous(self):
        index = IntervalIndex([0, 1], [2, 3])
        target = [0.5, 1.5, 2.5]
        actual = index.get_indexer(target)
        expected = [0, -1, 1]
        self.assert_numpy_array_equal(actual, expected)

        self.assertNotIn(1.5, index)

    def test_union(self):
        other = IntervalIndex([2], [3])
        expected = IntervalIndex(range(3), range(1, 4))
        actual = self.index.union(other)
        self.assertTrue(expected.equals(actual))

        actual = other.union(self.index)
        self.assertTrue(expected.equals(actual))

    def test_math(self):
        # add, subtract, multiply, divide with scalers should be OK
        actual = 2 * self.index + 1
        expected = IntervalIndex.from_breaks((2 * np.arange(3) + 1))
        self.assertTrue(expected.equals(actual))

        actual = self.index / 2.0 - 1
        expected = IntervalIndex.from_breaks((np.arange(3) / 2.0 - 1))
        self.assertTrue(expected.equals(actual))

        with self.assertRaises(TypeError):
            # doesn't make sense to add two IntervalIndex objects
            self.index + self.index

    def test_datetime(self):
        dates = pd.date_range('2000', periods=3)
        idx = IntervalIndex.from_breaks(dates)

        self.assert_numpy_array_equal(idx.left, dates[:2])
        self.assert_numpy_array_equal(idx.right, dates[-2:])

        expected = pd.date_range('2000-01-01T12:00', periods=2)
        self.assert_numpy_array_equal(idx.mid, expected)

        self.assertIn('2000-01-01T12', idx)

        target = pd.date_range('1999-12-31T12:00', periods=7, freq='12H')
        actual = idx.get_indexer(target)
        expected = [-1, -1, 0, 0, 1, 1, -1]
        self.assert_numpy_array_equal(actual, expected)

        expected = IntervalIndex(dates.shift(1))
        actual = idx.shift(1)
        self.assertTrue(expected.equals(actual))

        expected = IntervalIndex(pd.date_range('2000-01-02', periods=3))
        actual = idx + pd.to_timedelta(1, unit='D')
        self.assertTrue(expected.equals(actual))

    # TODO: other set operations (left join, right join, intersection),
    # set operations with conflicting IntervalIndex objects or other dtypes,
    # groupby, cut, reset_index...
