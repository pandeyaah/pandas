# min heap cython code originally from https://github.com/ncloudioj/cyheap.git

# implement a min priority heap, with special _remove operation to
# that pops based on the death value
# push, remove are O(log n), where n is window size
# support min & max representations

from numpy cimport (int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float16_t, float32_t, float64_t)

cdef extern from "minheap.h":
    ctypedef struct minheap_t:
        pass

    ctypedef struct item_t:
        int32_t death
        float64_t value

    minheap_t *minheap_create(uint32_t, size_t, bint)
    void    minheap_free(minheap_t*)
    void    minheap_dump(minheap_t*) nogil
    int     minheap_push(minheap_t*, item_t *) nogil
    item_t  *minheap_pop(minheap_t*) nogil
    item_t  *minheap_remove(minheap_t*, int *, item_t *) nogil
    item_t  *minheap_min(minheap_t*) nogil

cdef class MinHeap:
    """ Minimum heap container, a wrapper based on an implementation in C."""

    cdef:
        minheap_t *_c_minheap

    def __cinit__(self, uint32_t number, bint is_max):
        self._c_minheap = minheap_create(
            <uint32_t> number,
            sizeof(item_t),
            is_max)
        if self._c_minheap is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_minheap is not NULL:
            minheap_free(self._c_minheap)

    cdef void push(self, float64_t value, uint32_t death) nogil:
        """ push our data on the heap """

        cdef:
            item_t item
        item.death = death
        item.value = value
        minheap_push(self._c_minheap, &item)

    cdef void dump(self) nogil:
        """ print the heap """
        minheap_dump(self._c_minheap)

    cdef int remove(self, int *endi, float64_t value, uint32_t curval) nogil:
        """ remove the element only if death < curval """
        cdef:
            item_t *data
            item_t item
        item.death = curval
        item.value = value

        data = minheap_remove(self._c_minheap, endi, &item)

        if data is NULL:
            return 0
        else:
            return 1

    cdef int pop(self) nogil:
        """ pop the biggest element """
        cdef:
            item_t *data

        data = minheap_pop(self._c_minheap)

        if data is NULL:
            return 0
        else:
            return 1

    cdef item_t peek(self) nogil:
        """ return the min element """
        cdef item_t *data

        data = <item_t*>minheap_min(self._c_minheap)
        return data[0]
