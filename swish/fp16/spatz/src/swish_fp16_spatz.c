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

#include "swish_fp16_spatz.h"
#include "swish_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swish_fp16_spatz"

static int alloc_l1(void **params, uint32_t size, const float16 alpha)
{
    volatile swish_fp16_spatz_params_t *swish_params;
    uint32_t start;
    uint32_t len;

    mg_split(size, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    swish_params = l1_alloc(sizeof(swish_fp16_spatz_params_t));
    if (!swish_params)
        return ENOMEM;

    swish_params->shard_X = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!swish_params->shard_X)
        return ENOMEM;

    swish_params->shard_Y = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!swish_params->shard_Y)
        return ENOMEM;

    swish_params->alpha = (uintptr_t)l1_alloc(sizeof(float16));
    if (!swish_params->alpha)
        return ENOMEM;

    mmio_fp16(swish_params->alpha) = alpha;

    swish_params->start = start;
    swish_params->len   = len;
    swish_params->end   = start + len;

    *params = (void *)swish_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile swish_fp16_spatz_params_t *swish_params = params;

    if (swish_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = swish(X)), so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + swish_params->start),
                           swish_params->shard_X,
                           swish_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile swish_fp16_spatz_params_t *swish_params = params;

    if (swish_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + swish_params->start),
                           swish_params->shard_Y,
                           swish_params->len * sizeof(float16));

    return 0;
}

void MAGIA_swish_fp16_spatz(const float16 *X, float16 *Y, const float16 alpha, uint32_t size)
{
    int ret;
    volatile swish_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, size, alpha));
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

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(SWISH_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_swish_fp16_spatz_l1_start(swish_fp16_spatz_params_t *p,
                                     const float16 *X,
                                     float16 *Y,
                                     const float16 alpha,
                                     uint32_t size,
                                     mg_event_t *event)
{
    swish_fp16_spatz_params_set(p, X, Y, alpha, size);
    mg_spatz_run(WFE, SWISH_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_swish_fp16_spatz_l1(const float16 *X, float16 *Y, const float16 alpha, uint32_t size)
{
    swish_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_swish_fp16_spatz_l1_start(&params, X, Y, alpha, size, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
