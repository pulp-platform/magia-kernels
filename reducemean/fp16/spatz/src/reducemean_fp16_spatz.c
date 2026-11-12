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

#include "reducemean_fp16_spatz.h"
#include "reducemean_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "reducemean_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile reducemean_fp16_spatz_params_t *rm_params;
    uint32_t shard_X;
    uint32_t shard_Y;
    uint32_t out_start, out_len;

    mg_split(outer_dim, NUM_HARTS, HID, &out_start, &out_len);

    l1_alloc_init();

    rm_params = l1_alloc(sizeof(reducemean_fp16_spatz_params_t));
    if (!rm_params)
        return ENOMEM;

    shard_X = l1_alloc(out_len * reduce_dim * inner_dim * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(out_len * inner_dim * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    rm_params->shard_X     = shard_X;
    rm_params->shard_Y     = shard_Y;
    rm_params->reduce_dim  = reduce_dim;
    rm_params->inner_dim   = inner_dim;
    rm_params->outer_start = out_start;
    rm_params->outer_len   = out_len;

    *params = (void *)rm_params;
    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile reducemean_fp16_spatz_params_t *rm_params = params;
    uint32_t src_base;
    uint32_t in_elems;

    if (rm_params->outer_len == 0)
        return 0;

    src_base = rm_params->outer_start * rm_params->reduce_dim * rm_params->inner_dim;
    in_elems = rm_params->outer_len * rm_params->reduce_dim * rm_params->inner_dim;

    /*
     * This tile's outer slice [outer_start, outer_start+outer_len) is contiguous in L2. The
     * Spatz task fully writes shard_Y (one mean per output), so it is not zeroed here.
     */
    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(X + src_base), rm_params->shard_X, in_elems * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile reducemean_fp16_spatz_params_t *rm_params = params;
    uint32_t dst_base;
    uint32_t out_elems;

    if (rm_params->outer_len == 0)
        return 0;

    dst_base  = rm_params->outer_start * rm_params->inner_dim;
    out_elems = rm_params->outer_len * rm_params->inner_dim;

    /* This tile's output outer slice is contiguous in L2. */
    mg_idma_memcpy_1d_sync(
        WFE, 1, (uint32_t)(Y + dst_base), rm_params->shard_Y, out_elems * sizeof(float16));

    return 0;
}

void MAGIA_reducemean_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    int ret;
    volatile reducemean_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, outer_dim, reduce_dim, inner_dim));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(REDUCEMEAN_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_reducemean_fp16_spatz_l1_start(reducemean_fp16_spatz_params_t *p,
                                          const float16 *X,
                                          float16 *Y,
                                          uint32_t outer_len,
                                          uint32_t reduce_dim,
                                          uint32_t inner_dim,
                                          mg_event_t *event)
{
    reducemean_fp16_spatz_params_set(p, X, Y, outer_len, reduce_dim, inner_dim);
    mg_spatz_run(WFE, REDUCEMEAN_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_reducemean_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t outer_len, uint32_t reduce_dim, uint32_t inner_dim)
{
    reducemean_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_reducemean_fp16_spatz_l1_start(&params, X, Y, outer_len, reduce_dim, inner_dim, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
