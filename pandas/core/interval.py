import numpy as np

from pandas.core.base import PandasObject, IndexOpsMixin
from pandas.core.common import _values_from_object, _ensure_platform_int
from pandas.core.index import Index, _ensure_index, InvalidIndexError
from pandas.util.decorators import cache_readonly
import pandas.core.common as com


_VALID_CLOSED = set(['left', 'right', 'both', 'neither'])


class IntervalMixin(object):
    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def closed(self):
        return self._closed

    @cache_readonly
    def closed_left(self):
        return self.closed == 'left' or self.closed == 'both'

    @cache_readonly
    def closed_right(self):
        return self.closed == 'right' or self.closed == 'both'

    @property
    def open_left(self):
        return not self.closed_left

    @property
    def open_right(self):
        return not self.closed_right

    @cache_readonly
    def mid(self):
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * (self.right - self.left)

    def _validate(self):
        # TODO: exclude periods?
        if self.closed not in _VALID_CLOSED:
            raise ValueError("invalid options for 'closed': %s" % self.closed)

    def __lt__(self, other):
        other_left = getattr(other, 'left', other)
        if self.open_right or getattr(other, 'open_left', False):
            return self.right <= other_left
        return self.right < other_left

    def __gt__(self, other):
        other_right = getattr(other, 'right', other)
        if self.open_left or getattr(other, 'open_right', False):
            return self.left >= other_right
        return self.left > other_right


# TODO: cythonize this whole class?
class Interval(PandasObject, IntervalMixin):
    _typ = 'interval'

    def __init__(self, left, right, closed='right'):
        """Object representing an interval
        """
        self._left = left
        self._right = right
        self._closed = closed
        self._validate()

    def __hash__(self):
        return hash((self.left, self.right, self.closed))

    def __contains__(self, key):
        if isinstance(key, Interval):
            raise TypeError('__contains__ not defined for two intervals')
        return ((self.left < key if self.open_left else self.left <= key) and
                (key < self.right if self.open_right else key <= self.right))

    def __eq__(self, other):
        try:
            return (self.left == other.left
                    and self.right == other.right
                    and self.closed == other.closed)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    # TODO: add arithmetic operations

    def __unicode__(self):
        start_symbol = u'[' if self.closed_left else u'('
        end_symbol = u']' if self.closed_right else u')'
        return u'%s%s, %s%s' % (start_symbol, self.left, self.right, end_symbol)

    def __repr__(self):
        return ('%s(%r, %r, closed=%r)' %
                (type(self).__name__, self.left, self.right, self.closed))


class IntervalIndex(IntervalMixin, Index):
    _typ = 'intervalindex'
    _comparables = ['name']
    _attributes = ['name', 'closed']
    _allow_index_ops = True
    _engine = None # disable it

    def __new__(cls, left, right, closed='right', freq=None, name=None,
                fastpath=False):
        # TODO: validation
        result = object.__new__(cls)
        result._left = _ensure_index(left)
        result._right = _ensure_index(right)
        result._closed = closed
        result._freq = freq
        result.name = name
        result._validate()
        result._reset_identity()
        return result

    def _validate(self):
        IntervalMixin._validate(self)
        if len(self.left) != len(self.right):
            raise ValueError('left and right must have the same length')
        if self.freq is not None:
            if self.closed in ['neither', 'both']:
                raise ValueError("closed must be 'left' or 'right' if freq is"
                                 "provided")
        else:
            # infer freq?
            pass

    def _simple_new(cls, values, name=None, **kwargs):
        # ensure we don't end up here (this is a superclass method)
        raise NotImplementedError

    @property
    def _constructor(self):
        return type(self).from_intervals

    @classmethod
    def from_breaks(cls, breaks, closed='right', freq=None, name=None):
        return cls(breaks[:-1], breaks[1:], closed, freq, name)

    @classmethod
    def from_intervals(cls, data, freq=None, name=None):
        # TODO: cythonize (including validation for closed)
        closed = data[0].closed
        left = []
        right = []
        for interval in data:
            if interval.closed != closed:
                raise ValueError("inconsistent value of 'closed'")
            left.append(interval.left)
            right.append(interval.right)
        return cls(left, right, closed, freq, name)

    @classmethod
    def from_tuples(cls, data, closed='right', freq=None, name=None):
        left = []
        right = []
        for l, r in data:
            left.append(l)
            right.append(r)
        return cls(np.array(left), np.array(right), closed, freq, name)

    def to_tuples(self):
        return Index(com._asarray_tuplesafe(zip(self.left, self.right)))

    @property
    def freq(self):
        return self._freq

    @property
    def freqstr(self):
        return str(self.freq)

    def __len__(self):
        return len(self.left)

    @cache_readonly
    def values(self):
        # TODO: cythonize
        zipped = zip(self.left, self.right)
        items = [Interval(l, r, self.closed) for l, r in zipped]
        return np.array(items, dtype=object)

    def __array__(self, result=None):
        """ the array interface, return my values """
        return self.values

    def _array_values(self):
        return self.values

    def copy(self, deep=False):
        left = self.left.copy(deep=True) if deep else self.left
        right = self.right.copy(deep=True) if deep else self.right
        return type(self)(left, right, closed=self.closed, freq=self.freq,
                          name=self.name)

    @cache_readonly
    def dtype(self):
        return np.dtype('O')

    @cache_readonly
    def mid(self):
        try:
            return Index(0.5 * (self.left.values + self.right.values))
        except TypeError:
            # datetime safe version
            delta = self.right.values - self.left.values
            return Index(self.left.values + 0.5 * delta)

    @cache_readonly
    def is_monotonic(self):
        if not self.left.is_monotonic:
            return False
        if not self.right.is_monotonic:
            return False
        assert self.is_unique
        # needs cython to handle non-unique but still monotonic?
        if self.closed == 'both':
            return (self.left[1:] > self.right[:-1]).all()
        return (self.left[1:] >= self.right[:-1]).all()

    @cache_readonly
    def is_unique(self):
        return self.to_tuples().is_unique

    def _round_key(self, key, side='left'):
        if self.freq is None:
            raise KeyError('cannot round key if freq is unknown')
        # TODO: handle key as a string (if the left index can handle it)
        op = np.floor if side == 'left' else np.ceil
        int_key = _ensure_platform_int(op((key - self.left[0]) / self.freq))
        new_key = self.left[0] + int_key * self.freq
        return new_key

    def _assert_bounds_monotonic(self):
        if not self.left.is_monotonic and self.right.is_monotonic:
            raise KeyError("cannot lookup values on an IntervalIndex with "
                           "non-monotonic bounds")

    def _convert_scalar_indexer(self, key, typ=None):
        return key

    def _convert_list_indexer_for_mixed(self, keyarr, typ=None):
        """
        passed a key that is tuplesafe that is integer based
        and we have a mixed index (e.g. number/labels). figure out
        the indexer. return None if we can't help
        """
        # this doesn't take into account closed ATM
        indexer = (self._left<keyarr) & (self._right>keyarr)

        # we want integer indices for taking
        return np.arange(len(self))[indexer]

    def _get_regular(self, key, method='get_loc'):
        try:
            sub_index = getattr(self, self.closed)
        except AttributeError:
            raise KeyError
        try:
            key = self._round_key(key, self.closed)
        except TypeError:
            raise KeyError
        return getattr(sub_index, method)(key)

    def get_loc(self, key):
        if isinstance(key, Interval):
            start, end = self._searchsorted_bounds(key)
        else:
            try:
                return self._get_regular(key, 'get_loc')
            except KeyError:
                # TODO: handle decreasing monotonic intervals
                start, end = self._searchsorted_bounds(key)

        if start == end:
            raise KeyError(key)

        if start + 1 == end:
            return start
        else:
            return slice(start, end)

    def get_value(self, series, key):
        loc = self.get_loc(key)
        return series.iloc[loc]

    def get_indexer(self, target, **kwargs):
        target = _ensure_index(target)
        try:
            if isinstance(target, IntervalIndex):
                if (self.freq is not None and target.freq == self.freq
                        and target.closed == self.closed):
                    return self.left.get_indexer(target.left)
                else:
                    raise KeyError
            # TODO: try looking up key directly before rounding?
            return self._get_regular(target, 'get_indexer')
        except KeyError:
            # fall back on binary search
            start, end = self._searchsorted_bounds(target)
            if np.any(end - start > 1):
                raise KeyError('cannot uniquely map target to an indexer')
            start[start == end] = -1
            return start

    def get_indexer_non_unique(target, **kwargs):
        raise KeyError('cannot index an non-unique IntervalIndex')

    def delete(self, loc):
        new_left = self.left.delete(loc)
        new_right = self.right.delete(loc)
        return type(self)(new_left, new_right, self.closed, self.freq,
                          self.name, fastpath=True)

    def insert(self, loc, item):
        if not isinstance(item, Interval):
            raise ValueError('can only insert Interval objects into an '
                             'IntervalIndex')
        if not item.closed == self.closed:
            raise ValueError('inserted item must be closed on the same side '
                             'as the index')
        new_left = self.left.insert(loc, item.left)
        new_right = self.right.insert(loc, item.right)
        return type(self)(new_left, new_right, self.closed, self.freq,
                          self.name, fastpath=True)

    def take(self, indexer, axis=0):
        indexer = com._ensure_platform_int(indexer)
        new_left = self.left.take(indexer)
        new_right = self.right.take(indexer)
        return type(self)(new_left, new_right, self.closed, self.freq,
                          self.name, fastpath=True)

    def _searchsorted_bounds(self, key):
        """
        Parameters
        ----------
        key : label, interval, array of labels or IntervalIndex

        Returns
        -------
        start_slice, end_slice : int or array of ints
        """
        self._assert_bounds_monotonic()

        key_closed_left = getattr(key, 'closed_left', True)
        key_left = getattr(key, 'left', key)

        key_closed_right = getattr(key, 'closed_right', True)
        key_right = getattr(key, 'right', key)

        overlapping_start = self.closed_right and key_closed_left
        side_start = 'left' if overlapping_start else 'right'
        start_slice = self.right.searchsorted(key_left, side=side_start)

        overlapping_end = self.closed_left and key_closed_right
        side_end = 'right' if overlapping_end else 'left'
        end_slice = self.left.searchsorted(key_right, side=side_end)

        return start_slice, end_slice

    def slice_locs(self, start=None, end=None):
        if start is None:
            start = self.left[0]
        if end is None:
            end = self.right[-1]
        interval = Interval(start, end, closed='both')
        return self._searchsorted_bounds(interval)

    def __contains__(self, key):
        try:
            self.get_loc(key)
            return True
        except KeyError:
            return False

    def __getitem__(self, value):
        left = self.left[value]
        right = self.right[value]
        if not isinstance(left, Index):
            return Interval(left, right, self.closed)
        else:
            return type(self)(left, right, self.closed, self.freq, self.name)

    def __repr__(self):
        lines = [repr(type(self))]
        if len(self) > 10:
            lines.extend(str(interval) for interval in self[:5])
            lines.append('...')
            lines.extend(str(interval) for interval in self[-5:])
        else:
            lines.extend(str(interval) for interval in self)
        lines.append('Length: %s, Closed: %r, Freq: %r' %
                     (len(self), self.closed, self.freq))
        return '\n'.join(lines)

    def equals(self, other):
        if self.is_(other):
            return True
        try:
            return (self.left.equals(other.left)
                    and self.right.equals(other.right)
                    and self.closed == other.closed)
        except AttributeError:
            return False

    def __eq__(self, other):
        result = ((self.left == getattr(other, 'left', other))
                  & (self.right == getattr(other, 'right', other)))
        if self.closed != getattr(other, 'closed', 'both'):
            # do the actual comparisons first anyways to ensure broadcast
            # compatibility
            result[:] = False
        return result

    def __ne__(self, other):
        return np.logical_not(self == other)

    def __le__(self, other):
        return (self < other) | (self == other)

    def __ge__(self, other):
        return (self > other) | (self == other)

    # TODO: arithmetic operations
