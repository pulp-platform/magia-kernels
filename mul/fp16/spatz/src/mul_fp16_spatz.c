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

#include "mul_fp16_spatz.h"
#include "mul_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "mul_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile mul_fp16_spatz_params_t *mul_params;
    uint32_t start;
    uint32_t len;

    mg_split(total_elems, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    mul_params = l1_alloc(sizeof(mul_fp16_spatz_params_t));
    if (!mul_params)
        return ENOMEM;

    mul_params->shard_A = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!mul_params->shard_A)
        return ENOMEM;

    mul_params->shard_B = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!mul_params->shard_B)
        return ENOMEM;

    mul_params->shard_C = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!mul_params->shard_C)
        return ENOMEM;

    mul_params->elem_start = start;
    mul_params->elem_len = len;
    mul_params->total_elems = total_elems;

    *params = (void *)mul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile mul_fp16_spatz_params_t *mul_params = params;

    if (mul_params->elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in both operands.
       The Spatz task writes every output (C = A * B), so shard_C is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(A + mul_params->elem_start),
                           mul_params->shard_A,
                           mul_params->elem_len * sizeof(float16));
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(B + mul_params->elem_start),
                           mul_params->shard_B,
                           mul_params->elem_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile mul_fp16_spatz_params_t *mul_params = params;

    if (mul_params->elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + mul_params->elem_start),
                           mul_params->shard_C,
                           mul_params->elem_len * sizeof(float16));

    return 0;
}

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems)
{
    int ret;
    volatile mul_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, total_elems));
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

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(MUL_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, C));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_mul_fp16_spatz_l1_start(mul_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t n,
                                   mg_event_t *event)
{
    mul_fp16_spatz_params_set(p, A, B, C, n);
    mg_spatz_run(WFE, MUL_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_mul_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n)
{
    mul_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_mul_fp16_spatz_l1_start(&params, A, B, C, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
