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

#include "add_fp16_spatz.h"
#include "add_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "add_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile add_fp16_spatz_params_t *add_params;
    uint32_t start;
    uint32_t len;

    mg_split(size, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    add_params = l1_alloc(sizeof(add_fp16_spatz_params_t));
    if (!add_params)
        return ENOMEM;

    add_params->shard_A = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!add_params->shard_A)
        return ENOMEM;

    add_params->shard_B = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!add_params->shard_B)
        return ENOMEM;

    add_params->shard_C = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!add_params->shard_C)
        return ENOMEM;

    add_params->start = start;
    add_params->len = len;
    add_params->end = start + len;

    *params = (void *)add_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile add_fp16_spatz_params_t *add_params = params;

    if (add_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in both operands. The Spatz task
       writes every output (C = A + B), so shard_C is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(A + add_params->start),
                           add_params->shard_A,
                           add_params->len * sizeof(float16));
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(B + add_params->start),
                           add_params->shard_B,
                           add_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile add_fp16_spatz_params_t *add_params = params;

    if (add_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + add_params->start),
                           add_params->shard_C,
                           add_params->len * sizeof(float16));

    return 0;
}

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile add_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, allocate_l1((void **)&params, size));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, A, B));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(ADD_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, C));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_add_fp16_spatz_l1_start(add_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t n,
                                   mg_event_t *event)
{
    add_fp16_spatz_params_set(p, A, B, C, n);
    mg_spatz_run(WFE, ADD_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_add_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n)
{
    add_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_add_fp16_spatz_l1_start(&params, A, B, C, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
