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

#include "hardsigmoid_fp16_spatz.h"
#include "hardsigmoid_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "hardsigmoid_fp16_spatz"

static int alloc_l1(void **params, uint32_t size, const float16 alpha, const float16 beta)
{
    volatile hardsigmoid_fp16_spatz_params_t *hardsigmoid_params;
    uint32_t start;
    uint32_t len;

    mg_split(size, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    hardsigmoid_params = l1_alloc(sizeof(hardsigmoid_fp16_spatz_params_t));
    if (!hardsigmoid_params)
        return ENOMEM;

    hardsigmoid_params->shard_X = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!hardsigmoid_params->shard_X)
        return ENOMEM;

    hardsigmoid_params->shard_Y = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!hardsigmoid_params->shard_Y)
        return ENOMEM;

    hardsigmoid_params->alpha = (uintptr_t)l1_alloc(sizeof(float16));
    if (!hardsigmoid_params->alpha)
        return ENOMEM;

    hardsigmoid_params->beta = (uintptr_t)l1_alloc(sizeof(float16));
    if (!hardsigmoid_params->beta)
        return ENOMEM;

    mmio_fp16(hardsigmoid_params->alpha) = alpha;
    mmio_fp16(hardsigmoid_params->beta)  = beta;

    hardsigmoid_params->start = start;
    hardsigmoid_params->len   = len;
    hardsigmoid_params->end   = start + len;

    *params = (void *)hardsigmoid_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile hardsigmoid_fp16_spatz_params_t *hardsigmoid_params = params;

    if (hardsigmoid_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = hardsigmoid(X)), so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + hardsigmoid_params->start),
                           hardsigmoid_params->shard_X,
                           hardsigmoid_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile hardsigmoid_fp16_spatz_params_t *hardsigmoid_params = params;

    if (hardsigmoid_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + hardsigmoid_params->start),
                           hardsigmoid_params->shard_Y,
                           hardsigmoid_params->len * sizeof(float16));

    return 0;
}

void MAGIA_hardsigmoid_fp16_spatz(
    const float16 *X, float16 *Y, const float16 alpha, const float16 beta, uint32_t size)
{
    int ret;
    volatile hardsigmoid_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, size, alpha, beta));
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
        prof_cyc_compute, ret, mg_spatz_call(HARDSIGMOID_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_hardsigmoid_fp16_spatz_l1_start(hardsigmoid_fp16_spatz_params_t *p,
                                           const float16 *X,
                                           float16 *Y,
                                           const float16 alpha,
                                           const float16 beta,
                                           uint32_t size,
                                           mg_event_t *event)
{
    hardsigmoid_fp16_spatz_params_set(p, X, Y, alpha, beta, size);
    mg_spatz_run(WFE, HARDSIGMOID_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_hardsigmoid_fp16_spatz_l1(
    const float16 *X, float16 *Y, const float16 alpha, const float16 beta, uint32_t size)
{
    hardsigmoid_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_hardsigmoid_fp16_spatz_l1_start(&params, X, Y, alpha, beta, size, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
