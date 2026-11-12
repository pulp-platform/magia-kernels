// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "swiglu_fast_fp16_spatz.h"
#include "swiglu_fast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swiglu_fast_fp16_spatz"

static int alloc_l1(void **params, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uintptr_t shard_G;
    uintptr_t shard_U;
    uint32_t elem_start;
    uint32_t elem_len;

    mg_split(len, NUM_HARTS, HID, &elem_start, &elem_len);

    l1_alloc_init();

    sf_params = l1_alloc(sizeof(swiglu_fast_fp16_spatz_params_t));
    if (!sf_params)
        return ENOMEM;

    /* The task writes Y over G: one shard for G and Y. */
    shard_G = (uintptr_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_U = (uintptr_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_U)
        return ENOMEM;

    sf_params->shard_G = shard_G;
    sf_params->shard_U = shard_U;
    sf_params->shard_Y = shard_G;
    sf_params->len     = elem_len;

    *params = (void *)sf_params;

    return 0;
}

static int init_input_params(void *params, const float16 *G, const float16 *U, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sf_params = (volatile swiglu_fast_fp16_spatz_params_t *)params;
    mg_split(len, NUM_HARTS, HID, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(G + elem_start),
                           (uint32_t)sf_params->shard_G,
                           elem_len * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(U + elem_start),
                           (uint32_t)sf_params->shard_U,
                           elem_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y, uint32_t len)
{
    volatile swiglu_fast_fp16_spatz_params_t *sf_params;
    uint32_t elem_start;
    uint32_t elem_len;

    sf_params = (volatile swiglu_fast_fp16_spatz_params_t *)params;
    mg_split(len, NUM_HARTS, HID, &elem_start, &elem_len);

    if (elem_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + elem_start),
                           (uint32_t)sf_params->shard_Y,
                           elem_len * sizeof(float16));

    return 0;
}

void MAGIA_swiglu_fast_fp16_spatz(float16 *Y, const float16 *G, const float16 *U, uint32_t len)
{
    int ret;
    volatile swiglu_fast_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, G, U, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(SWIGLU_FAST_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y, len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_swiglu_fast_fp16_spatz_l1_start(swiglu_fast_fp16_spatz_params_t *p,
                                           float16 *Y,
                                           const float16 *G,
                                           const float16 *U,
                                           uint32_t len,
                                           mg_event_t *event)
{
    swiglu_fast_fp16_spatz_params_set(p, Y, G, U, len);
    mg_spatz_run(WFE, SWIGLU_FAST_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_swiglu_fast_fp16_spatz_l1(float16 *Y, const float16 *G, const float16 *U, uint32_t len)
{
    swiglu_fast_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_swiglu_fast_fp16_spatz_l1_start(&params, Y, G, U, len, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
