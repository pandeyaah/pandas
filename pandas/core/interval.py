import numpy as np

from pandas.core.base import PandasObject, IndexOpsMixin
from pandas.core.common import _values_from_object
from pandas.core.index import Index
from pandas.util.decorators import cache_readonly


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
    def __new__(cls, left, right, closed='right', name=None):
        # TODO: validation
        result = object.__new__(cls)
        result._left = Index(left)
        result._right = Index(right)
        result._closed = closed
        result.name = name
        result._validate()
        result._reset_identity()
        return result

    def _simple_new(cls, values, name=None, **kwargs):
        # ensure we don't end up here (this is a superclass method)
        raise NotImplementedError

    @property
    def _constructor(self):
        return type(self).from_intervals

    @classmethod
    def from_breaks(cls, breaks, closed='right', name=None):
        return cls(breaks[:-1], breaks[1:], closed, name)

    @classmethod
    def from_intervals(cls, data, name=None):
        # TODO: cythonize (including validation for closed)
        left = [i.left for i in data]
        right = [i.right for i in data]
        closed = data[0].closed
        return cls(left, right, closed, name)

    @cache_readonly
    def _data(self):
        # TODO: cythonize
        zipped = zip(self.left, self.right)
        items = [Interval(l, r, self.closed) for l, r in zipped]
        return np.array(items, dtype=object)

    @cache_readonly
    def dtype(self):
        return np.dtype('O')

    def get_loc(self, key):
        if isinstance(key, Interval):
            # TODO: fall back to something like slice_locs if key not found
            return self._engine.get_loc(_values_from_object(key))
        else:
            # TODO: handle decreasing monotonic intervals
            if not self.left.is_monotonic and self.right.is_monotonic:
                raise KeyError("cannot lookup values on a non-monotonic "
                               "IntervalIndex")

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

    def get_indexer(self, key):
        # should reuse the core of get_loc
        # if the key consists of intervals, needs unique values to give
        # sensible results (like DatetimeIndex)
        # if the key consists of scalars, the index's intervals must also be
        # non-overlapping
        raise NotImplementedError

    def slice_locs(self, start, end):
        # should be more efficient than directly calling the superclass method,
        # which calls get_loc (we don't need to do binary search twice for each
        # key)
        raise NotImplementedError

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
