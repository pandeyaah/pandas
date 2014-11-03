import numpy as np

from pandas.core.base import PandasObject, IndexOpsMixin
from pandas.core.common import _values_from_object, _ensure_platform_int
from pandas.core.index import Index, _ensure_index
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
        # TODO: figure out how to do add/sub as arithemtic even on Index
        # objects. Is there a work around while we have deprecated +/- as
        # union/difference? Possibly need to add `add` and `sub` methods.
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * (self.right - self.left)

    def _validate(self):
        # TODO: exclude periods?
        if self.closed not in _VALID_CLOSED:
            raise ValueError("invalid options for 'closed': %s" % self.closed)


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

    def __eq__(self, other):
        try:
            return (self.left == other.left
                    and self.right == other.right
                    and self.closed == other.closed)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        other_left = getattr(other, 'left', other)
        if self.open_right or getattr(other, 'open_left', False):
            return self.right <= other_left
        return self.right < other_left

    def __le__(self, other):
        return NotImplementedError

    def __gt__(self, other):
        return NotImplementedError

    def __ge__(self, other):
        return NotImplementedError

    # TODO: finish comparisons
    # TODO: add arithmetic operations

    def __str__(self):
        start_symbol = '[' if self.closed_left else '('
        end_symbol = ']' if self.closed_right else ')'
        return '%s%s, %s%s' % (start_symbol, self.left, self.right, end_symbol)

    def __repr__(self):
        return ('%s(%r, %r, closed=%r)' %
                (type(self).__name__, self.left,
                 self.right, self.closed))


class IntervalIndex(Index, IntervalMixin):
    _typ = 'intervalindex'
    _comparables = ['name']
    _attributes = ['name', 'closed']
    _allow_index_ops = True
    _engine = None # disable it

    def __new__(cls, left, right, closed='right', freq=None, name=None):
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
        return cls(left, right, closed, freq, name)

    def to_tuples(self):
        return Index(com._asarray_tuplesafe(zip(self.left, self.right)))

    @property
    def freq(self):
        return self._freq

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

    @cache_readonly
    def dtype(self):
        return np.dtype('O')

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

    def get_loc(self, key):
        if isinstance(key, Interval):
            # TODO: handle key closed/open
            start, end = self.slice_locs(key.left, key.right)
        else:
            try:
                sub_index = getattr(self, self.closed)
                return sub_index.get_loc(self._round_key(key, self.closed))
            except KeyError:
                # TODO: handle decreasing monotonic intervals
                self._assert_bounds_monotonic()

                side_start = 'left' if self.closed_right else 'right'
                start = self.right.searchsorted(key, side=side_start)

                side_end = 'right' if self.closed_left else 'left'
                end = self.left.searchsorted(key, side=side_end)

        if start == end:
            raise KeyError(key)

        if start + 1 == end:
            return start
        else:
            return slice(start, end)

    def get_indexer(self, target):
        # should reuse the core of get_loc
        # if the key consists of intervals, needs unique values to give
        # sensible results (like DatetimeIndex)
        # if the key consists of scalars, the index's intervals must also be
        # non-overlapping
        target = _ensure_index(target)
        if isinstance(target, IntervalIndex):
            left_indexer = self.get_indexer(target.left)
            right_indexer = self.get_indexer(target.right)
            different = left_indexer != right_indexer
            indexer = left_indexer.copy()
            indexer[different] = -1
            return indexer
        try:
            # TODO: try looking up key directly before rounding?
            sub_index = getattr(self, self.closed)
            return sub_index.get_indexer(self._round_key(key, self.closed))
        except KeyError:
            raise NotImplementedError

    def slice_locs(self, start=None, end=None):
        # should be more efficient than directly calling the superclass method,
        # which calls get_loc (we don't need to do binary search twice for each
        # key)
        self._assert_bounds_monotonic()

        side_start = 'left' if self.closed_right else 'right'
        start_slice = self.right.searchsorted(start, side=side_start)

        side_end = 'right' if self.closed_left else 'left'
        end_slice = self.left.searchsorted(end, side=side_end)

        return start_slice, end_slice

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
            return type(self)(left, right, self.closed)

    def __repr__(self):
        lines = [repr(type(self))]
        if len(self) > 10:
            lines.extend(str(interval) for interval in self[:5])
            lines.append('...')
            lines.extend(str(interval) for interval in self[-5:])
        else:
            lines.extend(str(interval) for interval in self)
        lines.append('Length: %s, Closed: %r' %
                     (len(self), self.closed))
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

    # TODO: add comparisons and arithmetic operations
