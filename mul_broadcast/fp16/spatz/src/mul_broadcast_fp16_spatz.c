// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "mul_broadcast_fp16_spatz.h"
#include "mul_broadcast_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "mul_broadcast_fp16_spatz"

static int alloc_l1(void **params, const float16 *B, uint32_t total_elems)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params;
    uint32_t shard_A;
    uint32_t shard_C;
    uint32_t scalar;

    uint32_t elem_start;
    uint32_t elem_len;

    mg_split(total_elems, NUM_HARTS, HID, &elem_start, &elem_len);

    l1_alloc_init();

    mul_params = l1_alloc(sizeof(mul_broadcast_fp16_spatz_params_t));
    if (!mul_params)
        return ENOMEM;

    shard_A = l1_alloc(elem_len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_C = l1_alloc(elem_len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    scalar = l1_alloc(sizeof(float16));
    if (!scalar)
        return ENOMEM;

    /* The scalar is a single element: the CV32 copies it to L1 itself, no iDMA needed. */
    mmio_fp16(scalar) = B[0];

    mul_params->shard_A     = shard_A;
    mul_params->shard_C     = shard_C;
    mul_params->scalar      = scalar;
    mul_params->elem_start  = elem_start;
    mul_params->elem_len    = elem_len;
    mul_params->total_elems = total_elems;

    *params = (void *)mul_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params = params;

    if (mul_params->elem_len == 0)
        return 0;

    /*
     * This tile's slice [elem_start, elem_start+elem_len) is contiguous in A.
     * The Spatz task writes every output (C = A * s), so shard_C is not zeroed here.
     */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(A + mul_params->elem_start),
                           mul_params->shard_A,
                           mul_params->elem_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *C)
{
    volatile mul_broadcast_fp16_spatz_params_t *mul_params = params;

    if (mul_params->elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(C + mul_params->elem_start),
                           mul_params->shard_C,
                           mul_params->elem_len * sizeof(float16));

    return 0;
}

void MAGIA_mul_broadcast_fp16_spatz(const float16 *A,
                                    const float16 *B,
                                    float16 *C,
                                    uint32_t total_elems)
{
    int ret;
    volatile mul_broadcast_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, B, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, A);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(MUL_BROADCAST_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_mul_broadcast_fp16_spatz_l1_start(mul_broadcast_fp16_spatz_params_t *p,
                                             const float16 *A,
                                             const float16 *B,
                                             float16 *C,
                                             uint32_t n,
                                             mg_event_t *event)
{
    mul_broadcast_fp16_spatz_params_set(p, A, B, C, n);
    mg_spatz_run(WFE, MUL_BROADCAST_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_mul_broadcast_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n)
{
    mul_broadcast_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_mul_broadcast_fp16_spatz_l1_start(&params, A, B, C, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
