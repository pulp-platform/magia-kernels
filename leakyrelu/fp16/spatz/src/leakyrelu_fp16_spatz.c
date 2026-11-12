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

#include "leakyrelu_fp16_spatz.h"
#include "leakyrelu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "leakyrelu_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params;
    uint32_t shard_start;
    uint32_t shard_len;

    mg_split(size, NUM_HARTS, HID, &shard_start, &shard_len);

    l1_alloc_init();

    leakyrelu_params = l1_alloc(sizeof(leakyrelu_fp16_spatz_params_t));
    if (!leakyrelu_params)
        return ENOMEM;

    leakyrelu_params->shard_X = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!leakyrelu_params->shard_X)
        return ENOMEM;

    leakyrelu_params->shard_Y = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!leakyrelu_params->shard_Y)
        return ENOMEM;

    leakyrelu_params->alpha = (uintptr_t)l1_alloc(sizeof(float16));
    if (!leakyrelu_params->alpha)
        return ENOMEM;

    leakyrelu_params->start = shard_start;
    leakyrelu_params->end   = shard_start + shard_len;
    leakyrelu_params->len   = shard_len;

    *params = (void *)leakyrelu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, float16 a)
{
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params = params;

    mmio_fp16(leakyrelu_params->alpha) = a;

    if (leakyrelu_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = leakyrelu(X, alpha)), so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + leakyrelu_params->start),
                           leakyrelu_params->shard_X,
                           leakyrelu_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params = params;

    if (leakyrelu_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + leakyrelu_params->start),
                           leakyrelu_params->shard_Y,
                           leakyrelu_params->len * sizeof(float16));

    return 0;
}

void MAGIA_leakyrelu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha)
{
    int ret;
    volatile leakyrelu_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, allocate_l1((void **)&params, size));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X, alpha));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(LEAKYRELU_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_leakyrelu_fp16_spatz_l1_start(leakyrelu_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         uint32_t size,
                                         float16 alpha,
                                         mg_event_t *event)
{
    leakyrelu_fp16_spatz_params_set(p, X, Y, size, alpha);
    mg_spatz_run(WFE, LEAKYRELU_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_leakyrelu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t size, float16 alpha)
{
    leakyrelu_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_leakyrelu_fp16_spatz_l1_start(&params, X, Y, size, alpha, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
