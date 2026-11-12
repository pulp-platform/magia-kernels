// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "kernels_profiling_utils.h"

#include "slice_fp16_spatz.h"
#include "slice_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "slice_fp16_spatz"

static int alloc_l1(void **params,
                    uint32_t outer_dim,
                    uint32_t slice_dim,
                    uint32_t inner_dim,
                    uint32_t start_idx,
                    uint32_t out_slice_dim)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    uint32_t start_outer;
    uint32_t len_outer;

    size_t x_size;
    size_t y_size;

    mg_split(outer_dim, NUM_HARTS, HID, &start_outer, &len_outer);

    x_size = len_outer * slice_dim * inner_dim;
    y_size = len_outer * out_slice_dim * inner_dim;

    l1_alloc_init();

    slice_params = l1_alloc(sizeof(slice_fp16_spatz_params_t));
    if (!slice_params)
        return ENOMEM;

    shard_X = l1_alloc(x_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(y_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    slice_params->shard_X       = shard_X;
    slice_params->shard_Y       = shard_Y;
    slice_params->slice_dim     = slice_dim;
    slice_params->out_slice_dim = out_slice_dim;
    slice_params->inner_dim     = inner_dim;
    slice_params->start_outer   = start_outer;
    slice_params->len_outer     = len_outer;
    slice_params->start_idx     = start_idx;

    *params = (void *)slice_params;

    return 0;
}

static int init_input_params(void *params, const float16 *data)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    uint32_t start_outer;
    uint32_t len_outer;
    uint32_t slice_dim;
    uint32_t inner_dim;
    uint32_t global_offset;
    uint32_t x_bytes;

    slice_params = (volatile slice_fp16_spatz_params_t *)params;
    start_outer  = slice_params->start_outer;
    len_outer    = slice_params->len_outer;
    slice_dim    = slice_params->slice_dim;
    inner_dim    = slice_params->inner_dim;

    if (len_outer == 0)
        return 0;

    global_offset = start_outer * slice_dim * inner_dim;
    x_bytes       = len_outer * slice_dim * inner_dim * sizeof(float16);

    /* Load this tile's contiguous [len_outer, slice_dim, inner] block; the Spatz
       task extracts the slice and fully writes shard_Y (so no zeroing needed). */
    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(data + global_offset), slice_params->shard_X, x_bytes);

    return 0;
}

static int store_result(void *params, float16 *sliced)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    uint32_t start_outer;
    uint32_t len_outer;
    uint32_t out_stride;
    uint32_t global_offset;
    uint32_t y_bytes;

    slice_params = (volatile slice_fp16_spatz_params_t *)params;
    start_outer  = slice_params->start_outer;
    len_outer    = slice_params->len_outer;
    out_stride   = slice_params->out_slice_dim * slice_params->inner_dim;

    if (len_outer == 0)
        return 0;

    global_offset = start_outer * out_stride;
    y_bytes       = len_outer * out_stride * sizeof(float16);

    /* This tile's output rows [start_outer, start_outer+len_outer) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(
        WFE, 1, (uint32_t)(sliced + global_offset), slice_params->shard_Y, y_bytes);

    return 0;
}

void MAGIA_slice_fp16_spatz(const float16 *data,
                            float16 *sliced,
                            uint32_t outer_dim,
                            uint32_t slice_dim,
                            uint32_t inner_dim,
                            uint32_t start_idx,
                            uint32_t out_slice_dim)
{
    int ret;
    volatile slice_fp16_spatz_params_t *params;

    PROF_PHASE(
        prof_cyc_alloc,
        ret,
        alloc_l1((void **)&params, outer_dim, slice_dim, inner_dim, start_idx, out_slice_dim));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, data));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(SLICE_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, sliced));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_slice_fp16_spatz_l1_start(slice_fp16_spatz_params_t *p,
                                     const float16 *data,
                                     float16 *sliced,
                                     uint32_t outer_dim,
                                     uint32_t slice_dim,
                                     uint32_t inner_dim,
                                     uint32_t start_idx,
                                     uint32_t out_slice_dim,
                                     mg_event_t *event)
{
    slice_fp16_spatz_params_set(
        p, data, sliced, outer_dim, slice_dim, inner_dim, start_idx, out_slice_dim);
    mg_spatz_run(WFE, SLICE_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_slice_fp16_spatz_l1(const float16 *data,
                               float16 *sliced,
                               uint32_t outer_dim,
                               uint32_t slice_dim,
                               uint32_t inner_dim,
                               uint32_t start_idx,
                               uint32_t out_slice_dim)
{
    slice_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_slice_fp16_spatz_l1_start(
        &params, data, sliced, outer_dim, slice_dim, inner_dim, start_idx, out_slice_dim, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
