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

#include "globalmaxpool_fp16_spatz.h"
#include "globalmaxpool_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "globalmaxpool_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4])
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params;
    uint32_t channel_len;
    uint32_t start;
    uint32_t len;

    channel_len = input_shape[2] * input_shape[3];

    mg_split(input_shape[0] * input_shape[1], NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    gap_params = l1_alloc(sizeof(globalmaxpool_fp16_spatz_params_t));
    if (!gap_params)
        return ENOMEM;

    gap_params->shard_X = (uintptr_t)l1_alloc(len * channel_len * sizeof(float16));
    if (!gap_params->shard_X)
        return ENOMEM;

    gap_params->shard_Y = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!gap_params->shard_Y)
        return ENOMEM;

    gap_params->hw_len = channel_len;
    gap_params->start  = start;
    gap_params->len    = len;

    *params = (void *)gap_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params = params;
    uint32_t start;
    uint32_t len;
    uint32_t hw_len;

    start  = gap_params->start;
    len    = gap_params->len;
    hw_len = gap_params->hw_len;

    if (len == 0)
        return 0;

    /* This tile's channels [start, start+len) are contiguous in L2. The Spatz task writes
       every output (one max per channel), so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + start * hw_len),
                           gap_params->shard_X,
                           len * hw_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params = params;

    if (gap_params->len == 0)
        return 0;

    /* This tile's outputs [start, start+len) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + gap_params->start),
                           gap_params->shard_Y,
                           gap_params->len * sizeof(float16));

    return 0;
}

void MAGIA_globalmaxpool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4])
{
    int ret;
    volatile globalmaxpool_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, input_shape));
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
        prof_cyc_compute, ret, mg_spatz_call(GLOBALMAXPOOL_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_globalmaxpool_fp16_spatz_l1_start(globalmaxpool_fp16_spatz_params_t *p,
                                             const float16 *X,
                                             float16 *Y,
                                             uint32_t input_shape[4],
                                             mg_event_t *event)
{
    globalmaxpool_fp16_spatz_params_set(p, X, Y, input_shape);
    mg_spatz_run(WFE, GLOBALMAXPOOL_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_globalmaxpool_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t input_shape[4])
{
    globalmaxpool_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_globalmaxpool_fp16_spatz_l1_start(&params, X, Y, input_shape, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
