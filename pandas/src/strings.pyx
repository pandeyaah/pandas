cimport util

def string_na_map_bool(ndarray arr, object fn, list args, ndarray[uint8_t] mask,
                   bint convert=1):
    '''
    Substitute for np.vectorize with pandas-friendly dtype inference

    Parameters
    ----------
    arr : ndarray
    fm : function name
    args : a list of the args to pass function

    Returns
    -------
    mapped : ndarray
    '''
    cdef:
        Py_ssize_t i, n
        ndarray[uint8_t] result

    n = len(arr)
    result = np.empty(n, dtype=np.uint8)
    for i in range(n):
        if mask[i]:
            result[i] = 0
        else:
            result[i] = string_endswith(arr[i], args[0])

    return result

cdef inline bint string_endswith(char *x, object pat):
     return x.endswith(pat)

