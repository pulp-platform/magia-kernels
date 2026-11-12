// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "silu_fp16_spatz.h"
#include "silu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "silu_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile silu_fp16_spatz_params_t *silu_params;
    uint32_t shard_X;
    uint32_t elem_start;
    uint32_t elem_len;

    mg_split(total_elems, NUM_HARTS, HID, &elem_start, &elem_len);

    l1_alloc_init();

    silu_params = l1_alloc(sizeof(silu_fp16_spatz_params_t));
    if (!silu_params)
        return ENOMEM;

    /* The task works in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    silu_params->shard_X     = shard_X;
    silu_params->shard_Y     = shard_X;
    silu_params->elem_start  = elem_start;
    silu_params->elem_len    = elem_len;
    silu_params->total_elems = total_elems;

    *params = (void *)silu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile silu_fp16_spatz_params_t *silu_params = params;

    if (silu_params->elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in X. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + silu_params->elem_start),
                           silu_params->shard_X,
                           silu_params->elem_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile silu_fp16_spatz_params_t *silu_params = params;

    if (silu_params->elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + silu_params->elem_start),
                           silu_params->shard_Y,
                           silu_params->elem_len * sizeof(float16));

    return 0;
}

void MAGIA_silu_fp16_spatz(const float16 *X, float16 *Y, uint32_t total_elems)
{
    int ret;
    volatile silu_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(SILU_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_silu_fp16_spatz_l1_start(
    silu_fp16_spatz_params_t *p, const float16 *X, float16 *Y, uint32_t n, mg_event_t *event)
{
    silu_fp16_spatz_params_set(p, X, Y, n);
    mg_spatz_run(WFE, SILU_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_silu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t n)
{
    silu_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_silu_fp16_spatz_l1_start(&params, X, Y, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
