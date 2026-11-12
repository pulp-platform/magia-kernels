// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "bias_rope_fp16_spatz.h"
#include "bias_rope_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "bias_rope_fp16_spatz"

static int alloc_l1(void **params,
                    const float16 *C,
                    uint32_t rows,
                    uint32_t cols,
                    uint32_t col0,
                    uint32_t head_dim,
                    uint32_t group)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uintptr_t shard_X;
    uintptr_t bias;
    uintptr_t cos = 0;
    uintptr_t sin = 0;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    br_params = l1_alloc(sizeof(bias_rope_fp16_spatz_params_t));
    if (!br_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    bias = (uintptr_t)l1_alloc(cols * sizeof(float16));
    if (!bias)
        return ENOMEM;

    if (C) {
        cos = (uintptr_t)l1_alloc(row_len * (head_dim / 2) * sizeof(float16));
        if (!cos)
            return ENOMEM;

        sin = (uintptr_t)l1_alloc(row_len * (head_dim / 2) * sizeof(float16));
        if (!sin)
            return ENOMEM;
    }

    br_params->shard_X  = shard_X;
    br_params->bias     = bias;
    br_params->cos      = cos;
    br_params->sin      = sin;
    br_params->dims     = row_len | (cols << 16);
    br_params->col0     = col0;
    br_params->head_dim = head_dim;
    br_params->group    = group;

    *params = (void *)br_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *X,
                             const float16 *B,
                             const float16 *C,
                             const float16 *S,
                             uint32_t rows)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;
    uint32_t half;

    br_params = (volatile bias_rope_fp16_spatz_params_t *)params;
    cols      = br_params->dims >> 16;
    half      = br_params->head_dim / 2;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /*
     * This tile's rows (tokens) are contiguous in X, and so are their angles in C and S. Every
     * tile needs the whole bias.
     */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)br_params->shard_X,
                           row_len * cols * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE, 0, (uint32_t)B, (uint32_t)br_params->bias, cols * sizeof(float16));

    if (C) {
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(C + row_start * half),
                               (uint32_t)br_params->cos,
                               row_len * half * sizeof(float16));

        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(S + row_start * half),
                               (uint32_t)br_params->sin,
                               row_len * half * sizeof(float16));
    }

    return 0;
}

static int store_result(void *params, float16 *X, uint32_t rows)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    br_params = (volatile bias_rope_fp16_spatz_params_t *)params;
    cols      = br_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)br_params->shard_X,
                           row_len * cols * sizeof(float16));

    return 0;
}

void MAGIA_bias_rope_fp16_spatz(float16 *X,
                                const float16 *B,
                                const float16 *C,
                                const float16 *S,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t col0,
                                uint32_t head_dim,
                                uint32_t group)
{
    int ret;
    volatile bias_rope_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, C, rows, cols, col0, head_dim, group);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, B, C, S, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(BIAS_ROPE_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, X, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_bias_rope_fp16_spatz_l1_start(bias_rope_fp16_spatz_params_t *p,
                                         float16 *X,
                                         const float16 *B,
                                         const float16 *C,
                                         const float16 *S,
                                         uint32_t rows,
                                         uint32_t cols,
                                         uint32_t col0,
                                         uint32_t head_dim,
                                         uint32_t group,
                                         mg_event_t *event)
{
    bias_rope_fp16_spatz_params_set(p, X, B, C, S, rows, cols, col0, head_dim, group);
    mg_spatz_run(WFE, BIAS_ROPE_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_bias_rope_fp16_spatz_l1(float16 *X,
                                   const float16 *B,
                                   const float16 *C,
                                   const float16 *S,
                                   uint32_t rows,
                                   uint32_t cols,
                                   uint32_t col0,
                                   uint32_t head_dim,
                                   uint32_t group)
{
    bias_rope_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_bias_rope_fp16_spatz_l1_start(
        &params, X, B, C, S, rows, cols, col0, head_dim, group, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
