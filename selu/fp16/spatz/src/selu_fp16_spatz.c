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

#include "selu_fp16_spatz.h"
#include "selu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "selu_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile selu_fp16_spatz_params_t *selu_params;
    uint32_t shard_start;
    uint32_t shard_len;

    mg_split(size, NUM_HARTS, HID, &shard_start, &shard_len);

    l1_alloc_init();

    selu_params = l1_alloc(sizeof(selu_fp16_spatz_params_t));
    if (!selu_params)
        return ENOMEM;

    selu_params->shard_X = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!selu_params->shard_X)
        return ENOMEM;

    selu_params->shard_Y = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!selu_params->shard_Y)
        return ENOMEM;

    selu_params->alpha = (uintptr_t)l1_alloc(sizeof(float16));
    if (!selu_params->alpha)
        return ENOMEM;

    selu_params->gamma = (uintptr_t)l1_alloc(sizeof(float16));
    if (!selu_params->gamma)
        return ENOMEM;

    selu_params->start = shard_start;
    selu_params->end   = shard_start + shard_len;
    selu_params->len   = shard_len;

    *params = (void *)selu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, float16 a, float16 g)
{
    volatile selu_fp16_spatz_params_t *selu_params = params;

    mmio_fp16(selu_params->alpha) = a;
    mmio_fp16(selu_params->gamma) = g;

    if (selu_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = selu(X, alpha, gamma)), so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + selu_params->start),
                           selu_params->shard_X,
                           selu_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile selu_fp16_spatz_params_t *selu_params = params;

    if (selu_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + selu_params->start),
                           selu_params->shard_Y,
                           selu_params->len * sizeof(float16));

    return 0;
}

void MAGIA_selu_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma)
{
    int ret;
    volatile selu_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, allocate_l1((void **)&params, size));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X, alpha, gamma));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(SELU_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_selu_fp16_spatz_l1_start(selu_fp16_spatz_params_t *p,
                                    const float16 *X,
                                    float16 *Y,
                                    uint32_t size,
                                    float16 alpha,
                                    float16 gamma,
                                    mg_event_t *event)
{
    selu_fp16_spatz_params_set(p, X, Y, size, alpha, gamma);
    mg_spatz_run(WFE, SELU_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_selu_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma)
{
    selu_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_selu_fp16_spatz_l1_start(&params, X, Y, size, alpha, gamma, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
