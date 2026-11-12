// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "rowslice_sum_div_fp16_spatz.h"
#include "rowslice_sum_div_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowslice_sum_div_fp16_spatz"

static int alloc_l1(void **params, uint32_t n, uint32_t own, uint32_t rows, uint32_t cols)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uintptr_t shard_Y;
    uintptr_t shard_O;
    uintptr_t shard_G;
    uintptr_t shard_S;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    rd_params = l1_alloc(sizeof(rowslice_sum_div_fp16_spatz_params_t));
    if (!rd_params)
        return ENOMEM;

    shard_Y = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    shard_O = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_O)
        return ENOMEM;

    shard_G = (uintptr_t)l1_alloc(n * row_len * cols * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    rd_params->shard_Y = shard_Y;
    rd_params->shard_O = shard_O;
    rd_params->shard_G = shard_G;
    rd_params->shard_S = shard_S;
    rd_params->dims    = row_len | (cols << 16);
    rd_params->slices  = n | (own << 16);

    *params = (void *)rd_params;

    return 0;
}

static int
init_input_params(void *params, const float16 *O, const float16 *G, const float16 *S, uint32_t rows)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;
    uint32_t n;

    rd_params = (volatile rowslice_sum_div_fp16_spatz_params_t *)params;
    cols      = rd_params->dims >> 16;
    n         = rd_params->slices & 0xffff;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(O + row_start * cols),
                           (uint32_t)rd_params->shard_O,
                           row_len * cols * sizeof(float16));

    /* This tile's rows of each of the n slices, packed into an n x row_len x cols block. */
    if (n > 0) {
        mg_idma_memcpy_2d_sync(WFE,
                               0,
                               (uint32_t)(G + row_start * cols),
                               (uint32_t)rd_params->shard_G,
                               row_len * cols * sizeof(float16),
                               rows * cols * sizeof(float16),
                               n);
    }

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(S + row_start),
                           (uint32_t)rd_params->shard_S,
                           row_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y, uint32_t rows)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rd_params = (volatile rowslice_sum_div_fp16_spatz_params_t *)params;
    cols      = rd_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + row_start * cols),
                           (uint32_t)rd_params->shard_Y,
                           row_len * cols * sizeof(float16));

    return 0;
}

void MAGIA_rowslice_sum_div_fp16_spatz(float16 *Y,
                                       const float16 *O,
                                       const float16 *G,
                                       const float16 *S,
                                       uint32_t n,
                                       uint32_t own,
                                       uint32_t rows,
                                       uint32_t cols)
{
    int ret;
    volatile rowslice_sum_div_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, n, own, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, O, G, S, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(ROWSLICE_SUM_DIV_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_rowslice_sum_div_fp16_spatz_l1_start(rowslice_sum_div_fp16_spatz_params_t *p,
                                                float16 *Y,
                                                const float16 *O,
                                                const float16 *G,
                                                const float16 *S,
                                                uint32_t n,
                                                uint32_t own,
                                                uint32_t rows,
                                                uint32_t cols,
                                                mg_event_t *event)
{
    rowslice_sum_div_fp16_spatz_params_set(p, Y, O, G, S, n, own, rows, cols);
    mg_spatz_run(WFE, ROWSLICE_SUM_DIV_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols)
{
    rowslice_sum_div_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_rowslice_sum_div_fp16_spatz_l1_start(&params, Y, O, G, S, n, own, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
