/*

This implements an ordered doubly-linked list, ordered from smallest to largest
so peeking at the head is min and tail is max in O(1) after insertion.

We add a remove operation that will traverses in order and will remove
based on the death value (matches the death value of an index
entry)

 */



#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#ifndef PANDAS_INLINE
  #if defined(__GNUC__)
    #define PANDAS_INLINE static __inline__
  #elif defined(_MSC_VER)
    #define PANDAS_INLINE static __inline
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define PANDAS_INLINE static inline
  #else
    #define PANDAS_INLINE
  #endif
#endif

/* array impl */

typedef struct array_t {
    uint32_t    nelm;        /* current # of elements in array   */
    uint32_t    nrest;       /* rest # of elements in array      */
    size_t      size;        /* size of each item in array       */
    void        *elm;        /* point to the first item in array */
} array_t;

#define ARRAY_AT(a, i) ((void *)(((char *)(a)->elm + (a)->size*(i))))
#define ARRAY_LEN(a) ((a)->nelm)

array_t *
array_create(uint32_t n, size_t s) {
    array_t *a;
    uint32_t initial_number; /* initial number of elements in array, 1 by default */

    a = malloc(sizeof(array_t));
    if (a == NULL) {
        return NULL;
    }

    initial_number = n==0 ? 1 : n;
    a->elm = malloc(initial_number * s);
    if (a->elm == NULL) {
    	free(a);
    	return NULL;
    }
    a->nelm = 0;
    a->nrest = initial_number;
    a->size = s;

    return a;
}

void
array_free(array_t *array) {
	free(array->elm);
    free(array);
}

void *
array_push(array_t * array) {
    void *ret, *elm;

    if (array->nrest > 0) {
        ret = ARRAY_AT(array, array->nelm);
        array->nrest--;
        array->nelm++;
        return ret;
    }

    /* double array's size, if failed, return NULL */
    if ((elm=realloc(array->elm, 2*array->size*array->nelm)) == NULL) {
    	return NULL;
    } else {
        array->elm = elm;
        ret = ARRAY_AT(array, array->nelm);
        array->nrest = array->nelm - 1;
        array->nelm++;
    }

    return ret;
}

void *
array_push_n(array_t * array, uint32_t n) {
    void *ret, *elm;

    if (n == 0) return NULL;

    if (array->nrest >= n) {
        ret = ARRAY_AT(array, array->nelm);
        array->nrest -= n;
        array->nelm += n;
        return ret;
    }

    /* expand array's size, if failed, return NULL */
    size_t nslots;
    nslots = (2*array->nelm > n) ? 2*array->nelm: array->nelm+array->nrest+n;
    if ((elm=realloc(array->elm, nslots*array->size)) == NULL) {
    	return NULL;
    } else {
        array->elm = elm;
        ret = ARRAY_AT(array, array->nelm);
        array->nelm += n;
        array->nrest = nslots - array->nelm;
    }

    return ret;
}

uint32_t
array_len(array_t * array) {
    return array->nelm;
}

void *
array_at(array_t * array, int i) {
	if (i >= ARRAY_LEN(array) || (i < 0 && -i > ARRAY_LEN(array))) return NULL;
	return i >= 0 ? ARRAY_AT(array, i): ARRAY_AT(array, ARRAY_LEN(array)+i);
}

/* heap impl */

typedef struct item_t {
  int death;     /* counter for when to remove this item */
  double value;  /* value of the item */
} item_t;

typedef int (*compare)(const item_t *, const item_t *);

typedef struct minheap_t {
    uint32_t    len;        /* length of minheap, might be less than the actual size of array. */
    struct array_t     *array;     /* dynamic array for storing the elmentes of minheap */
    compare     cmp;       /* customized compare operator */
} minheap_t;

int item_cmp_min(const item_t *self, const item_t *other) {
  if (self->value > other->value) {
    return 1;
  }
  else if (self->value < other->value) {
    return -1;
  }
  return 0;
}

int item_cmp_max(const item_t *self, const item_t *other) {
  if (self->value > other->value) {
    return -1;
  }
  else if (self->value < other->value) {
    return 1;
  }
  return 0;
}

item_t *
item_copy(item_t *self, const item_t *other) {
  self->death = other->death;
  self->value = other->value;
  return self;
}

void item_swap(item_t *self, item_t *other) {
  if (self != other) {
    int death = self->death;
    double value = self->value;
    self->death = other->death;
    self->value = other->value;
    other->death = death;
    other->value = value;
  }
  return;
}

void item_dump(const item_t *self, const char *msg) {
  // print in item to stdout
  printf("%s %.2f [%d]\n", msg, self->value, self->death);
}

static void
shiftdown(minheap_t *heap, int start, int end) {
    item_t *child, *parent;
    int i;  // index for the parent

    i = end;
    while (end > start) {
        child = array_at(heap->array, i);
        i = (end - 1) >> 1;
        parent = array_at(heap->array, i);
        if (heap->cmp(child, parent) < 0) {
            item_swap(child, parent);
            end = i;
        } else
            break;
    }

    return;
}

static void
shiftup(minheap_t *heap, int start) {
    int iend, istart, ichild, iright;
    item_t *child, *parent;

    iend = (int)heap->len;
    istart = start;
    ichild = 2 * istart + 1;
    while (ichild < iend) {
        iright = ichild + 1;
        if (iright < iend && heap->cmp(array_at(heap->array, ichild),
                    array_at(heap->array, iright)) > 0) {
            ichild = iright;
        }
        parent = array_at(heap->array, istart);
        child = array_at(heap->array, ichild);
        item_swap(parent, child);
        istart = ichild;
        ichild = 2 * istart + 1;
    }
    shiftdown(heap, start, istart);
    return;
}

static void
heapify(minheap_t *heap, int start) {
    int i;

    i = (int)(heap->len >> 2);
    for (; i >=0; i--) {
        shiftup(heap, i);
    }
    return;
}

struct minheap_t *
minheap_create(uint32_t n, size_t size, int is_max) {
    minheap_t *heap;

    heap = malloc(sizeof(minheap_t));
    if (heap == NULL) return NULL;
    heap->cmp = is_max ? item_cmp_max : item_cmp_min;
    heap->len = 0;
    heap->array = array_create(n, size);
    if (heap->array == NULL) {
		free(heap);
		return NULL;
    }
    return heap;
}

void
minheap_free(minheap_t *heap) {
    array_free(heap->array);
    free(heap);
}

int
minheap_push(minheap_t *heap, const item_t *new) {
    item_t *item;

    // item_dump(new, "push");
    if (heap->len == array_len(heap->array)) {
        if ((item=array_push(heap->array)) == NULL) {
            return -1;
        }
    } else {
        item = array_at(heap->array, heap->len);
    }

    item_copy(item, new);
    heap->len++;
    shiftdown(heap, 0, heap->len - 1);
    return 0;
}

void
minheap_dump(minheap_t *heap) {
  int i;
  printf("\ndumpstart:\n");
  printf("----------\n");
  for (i=0; i <(int)(heap->len); i++) {
    printf("[%d] ", i);
    item_dump(array_at(heap->array, i), "dump");
  }
  printf("\n");
}

item_t *
minheap_pop(minheap_t *heap) {
    item_t * root;
    item_t *child, *parent;

    if (heap->len == 0) return NULL;

    parent = array_at(heap->array, 0);
    child = array_at(heap->array, heap->len-1);
    item_swap(parent, child);
    heap->len--;
    shiftup(heap, 0);
    root = array_at(heap->array, heap->len); /* note the previous first elm has been swapped to here. */
    return root;
}

item_t *
minheap_remove(minheap_t *heap, int *endi, const item_t *find) {
    item_t *item, *swp;
    int iend, istart, ichild, iright, cmp, deathval, death;

    if (heap->len == 0) return NULL;

    // item_dump(find, "remove");

    /* find the item */
    iend = (int)heap->len;
    istart = 0;
    ichild = 0;
    while (ichild < iend) {

      item = array_at(heap->array, ichild);
      cmp = heap->cmp(item, find);

      if (!cmp) {
        /* found the node */
        // item_dump(item, "found!!!");

        /* remove if the node is dead */
        death = item->death;
        deathval = endi[death];
        if (deathval <= find->death) {
          // item_dump(find, "dead!!!");

          swp = array_at(heap->array, heap->len-1);
          item_swap(item, swp);
          heap->len--;
          shiftup(heap, ichild);
          return item;
        }

        /* found but still alive, we are done */
        break;
      }

      iright = ichild + 1;
      istart = ichild;
      if (cmp > 0) {
        ichild = iright;
      }
      else {
        ichild = 2 * istart + 1;
      }
    }
    return NULL;
}

void *
minheap_min(minheap_t *heap) {
    if (heap->len == 0) return NULL;
    return array_at(heap->array, 0);
}
