// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "max_fp16_spatz.h"
#include "max_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "max_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile max_fp16_spatz_params_t *max_params;
    uint32_t start;
    uint32_t len;

    mg_split(size, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    max_params = l1_alloc(sizeof(max_fp16_spatz_params_t));
    if (!max_params)
        return ENOMEM;

    max_params->shard_A = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!max_params->shard_A)
        return ENOMEM;

    max_params->shard_B = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!max_params->shard_B)
        return ENOMEM;

    max_params->shard_C = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!max_params->shard_C)
        return ENOMEM;

    max_params->start = start;
    max_params->len = len;
    max_params->end = start + len;

    *params = (void *)max_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile max_fp16_spatz_params_t *max_params = params;

    if (max_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in both operands. The Spatz task
       writes every output (C = max(A, B)), so shard_C is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(A + max_params->start),
                           max_params->shard_A,
                           max_params->len * sizeof(float16));
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(B + max_params->start),
                           max_params->shard_B,
                           max_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile max_fp16_spatz_params_t *max_params = params;

    if (max_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + max_params->start),
                           max_params->shard_C,
                           max_params->len * sizeof(float16));

    return 0;
}

void MAGIA_max_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile max_fp16_spatz_params_t *params;

    ret = allocate_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(MAX_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_max_fp16_spatz_l1_start(max_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t size,
                                   mg_event_t *event)
{
    max_fp16_spatz_params_set(p, A, B, C, size);
    mg_spatz_run(WFE, MAX_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_max_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    max_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_max_fp16_spatz_l1_start(&params, A, B, C, size, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
