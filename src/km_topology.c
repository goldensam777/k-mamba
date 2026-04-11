#include "km_topology.h"

#include <limits.h>
#include <string.h>

#include "wavefront_nd.h"

int km_spatial_dims_product(const long *dims, long ndims, size_t *product_out) {
    size_t product = 1;

    if (!dims || !product_out || ndims <= 0 || ndims > KMAMBA_MAX_NDIMS) return 0;
    if (!wavefront_nd_validate_dims(dims, ndims)) return 0;

    for (long axis = 0; axis < ndims; axis++) {
        size_t axis_extent = (size_t)dims[axis];
        if (product > ((size_t)-1) / axis_extent) return 0;
        product *= axis_extent;
    }

    *product_out = product;
    return 1;
}

int km_make_row_major_strides(const long *dims, long ndims, long *strides_out) {
    long stride;

    if (!dims || !strides_out || ndims <= 0 || ndims > KMAMBA_MAX_NDIMS) return 0;
    if (!wavefront_nd_validate_dims(dims, ndims)) return 0;

    stride = 1;
    for (long axis = ndims - 1; axis >= 0; axis--) {
        strides_out[axis] = stride;
        stride *= dims[axis];
    }

    return 1;
}

void km_unravel_index(long linear,
                      const long *dims,
                      const long *strides,
                      long ndims,
                      long *coords_out) {
    if (!dims || !strides || !coords_out || ndims <= 0 || ndims > KMAMBA_MAX_NDIMS) return;
    if (linear < 0) return;

    for (long axis = 0; axis < ndims; axis++) {
        coords_out[axis] = (linear / strides[axis]) % dims[axis];
    }
}

long km_ravel_index(const long *coords,
                    const long *dims,
                    const long *strides,
                    long ndims) {
    long offset;

    if (!coords || !dims || !strides || ndims <= 0 || ndims > KMAMBA_MAX_NDIMS) return -1;

    offset = 0;
    for (long axis = 0; axis < ndims; axis++) {
        if (coords[axis] < 0 || coords[axis] >= dims[axis]) return -1;
        offset += coords[axis] * strides[axis];
    }

    return offset;
}

long km_power_long(long base, long exp) {
    long result;

    if (exp < 0) return 0;
    if (base == 0 && exp == 0) return 0;

    result = 1;
    for (long i = 0; i < exp; i++) {
        result *= base;
    }

    return result;
}

int km_normalize_spatial_topology(long *spatial_ndims,
                                  long *spatial_dims,
                                  size_t seq_len,
                                  int use_convnd,
                                  long *convnd_ndims,
                                  long convnd_K) {
    size_t total_points;

    fprintf(stderr, "[DEBUG] km_normalize_spatial_topology: spatial_ndims=%ld, seq_len=%zu\n", 
            *spatial_ndims, seq_len);
    fprintf(stderr, "[DEBUG] spatial_dims[0]=%ld, spatial_dims[1]=%ld\n",
            spatial_dims[0], spatial_dims[1]);

    if (!spatial_ndims || !spatial_dims) {
        fprintf(stderr, "[DEBUG] FAIL: NULL pointers\n");
        return -1;
    }
    if (seq_len > (size_t)LONG_MAX) {
        fprintf(stderr, "[DEBUG] FAIL: seq_len too large\n");
        return -1;
    }

    if (*spatial_ndims <= 0) {
        fprintf(stderr, "[DEBUG] Auto-setting spatial_ndims to 1\n");
        memset(spatial_dims, 0, (size_t)KMAMBA_MAX_NDIMS * sizeof(long));
        *spatial_ndims = 1;
        spatial_dims[0] = (long)seq_len;
    }

    if (!km_spatial_dims_product(spatial_dims, *spatial_ndims, &total_points)) {
        fprintf(stderr, "[DEBUG] FAIL: km_spatial_dims_product failed\n");
        return -1;
    }
    fprintf(stderr, "[DEBUG] total_points=%zu, seq_len=%zu\n", total_points, seq_len);
    if (total_points != seq_len) {
        fprintf(stderr, "[DEBUG] FAIL: total_points != seq_len\n");
        return -1;
    }

    if (use_convnd) {
        if (!convnd_ndims || convnd_K <= 0) return -1;
        if (*convnd_ndims <= 0) *convnd_ndims = *spatial_ndims;
        if (*convnd_ndims != *spatial_ndims) return -1;
    }

    return 0;
}
