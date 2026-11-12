// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "rope_fp16_spatz.h"
#include "rope_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rope_fp16_spatz"

static int alloc_l1(void **params, uint32_t rows, uint32_t n_heads, uint32_t head_dim)
{
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t shard_X;
    uint32_t cos;
    uint32_t sin;

    uint32_t r_start;
    uint32_t r_len;

    mg_split(rows, NUM_HARTS, HID, &r_start, &r_len);

    l1_alloc_init();

    rope_params = l1_alloc(sizeof(rope_fp16_spatz_params_t));
    if (!rope_params)
        return ENOMEM;

    /* The task rotates the rows in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(r_len * n_heads * head_dim * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    cos = (uint32_t)l1_alloc(r_len * (head_dim / 2) * sizeof(float16));
    if (!cos)
        return ENOMEM;

    sin = (uint32_t)l1_alloc(r_len * (head_dim / 2) * sizeof(float16));
    if (!sin)
        return ENOMEM;

    rope_params->shard_X  = shard_X;
    rope_params->shard_Y  = shard_X;
    rope_params->cos      = cos;
    rope_params->sin      = sin;
    rope_params->r_start  = r_start;
    rope_params->r_len    = r_len;
    rope_params->n_heads  = n_heads;
    rope_params->head_dim = head_dim;

    *params = (void *)rope_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *cos, const float16 *sin)
{
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t width;
    uint32_t half;

    rope_params = (volatile rope_fp16_spatz_params_t *)params;
    r_start     = rope_params->r_start;
    r_len       = rope_params->r_len;
    width       = rope_params->n_heads * rope_params->head_dim;
    half        = rope_params->head_dim / 2;

    if (r_len == 0)
        return 0;

    /* This tile's rows [r_start, r_start+r_len) are contiguous in X, cos and sin. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + r_start * width),
                           rope_params->shard_X,
                           r_len * width * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(cos + r_start * half), rope_params->cos, r_len * half * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(sin + r_start * half), rope_params->sin, r_len * half * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t width;

    rope_params = (volatile rope_fp16_spatz_params_t *)params;
    r_start     = rope_params->r_start;
    r_len       = rope_params->r_len;
    width       = rope_params->n_heads * rope_params->head_dim;

    if (r_len == 0)
        return 0;

    /* This tile's output rows are contiguous in Y. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + r_start * width),
                           rope_params->shard_Y,
                           r_len * width * sizeof(float16));

    return 0;
}

void MAGIA_rope_fp16_spatz(const float16 *X,
                           const float16 *cos,
                           const float16 *sin,
                           uint32_t rows,
                           uint32_t n_heads,
                           uint32_t head_dim,
                           float16 *Y)
{
    int ret;
    volatile rope_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, n_heads, head_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, cos, sin);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(ROPE_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_rope_fp16_spatz_l1_start(rope_fp16_spatz_params_t *p,
                                    const float16 *X,
                                    const float16 *cos,
                                    const float16 *sin,
                                    uint32_t rows,
                                    uint32_t n_heads,
                                    uint32_t head_dim,
                                    float16 *Y,
                                    mg_event_t *event)
{
    rope_fp16_spatz_params_set(p, X, cos, sin, rows, n_heads, head_dim, Y);
    mg_spatz_run(WFE, ROPE_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_rope_fp16_spatz_l1(const float16 *X,
                              const float16 *cos,
                              const float16 *sin,
                              uint32_t rows,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              float16 *Y)
{
    rope_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_rope_fp16_spatz_l1_start(&params, X, cos, sin, rows, n_heads, head_dim, Y, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
