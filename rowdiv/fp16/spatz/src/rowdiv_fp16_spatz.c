// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "rowdiv_fp16_spatz.h"
#include "rowdiv_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowdiv_fp16_spatz"

static int alloc_l1(void **params, uint32_t rows, uint32_t cols)
{
    volatile rowdiv_fp16_spatz_params_t *rd_params;
    uintptr_t shard_X;
    uintptr_t shard_S;
    uintptr_t shard_Y;

    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    rd_params = l1_alloc(sizeof(rowdiv_fp16_spatz_params_t));
    if (!rd_params)
        return ENOMEM;

    shard_X = l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_S = l1_alloc(row_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    shard_Y = l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    rd_params->shard_X   = shard_X;
    rd_params->shard_S   = shard_S;
    rd_params->shard_Y   = shard_Y;
    rd_params->cols      = cols;
    rd_params->row_start = row_start;
    rd_params->row_len   = row_len;

    *params = (void *)rd_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *S)
{
    volatile rowdiv_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rd_params = (volatile rowdiv_fp16_spatz_params_t *)params;
    row_start = rd_params->row_start;
    row_len   = rd_params->row_len;
    cols      = rd_params->cols;

    if (row_len == 0)
        return 0;

    /*
     * This tile's rows [row_start, row_start+row_len) are contiguous in X, and so are their
     * divisors in S. The Spatz task writes every output, so shard_Y is not zeroed here.
     */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)rd_params->shard_X,
                           row_len * cols * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(S + row_start), (uint32_t)rd_params->shard_S, row_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile rowdiv_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rd_params = (volatile rowdiv_fp16_spatz_params_t *)params;
    row_start = rd_params->row_start;
    row_len   = rd_params->row_len;
    cols      = rd_params->cols;

    if (row_len == 0)
        return 0;

    /* This tile's output rows are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + row_start * cols),
                           (uint32_t)rd_params->shard_Y,
                           row_len * cols * sizeof(float16));

    return 0;
}

void MAGIA_rowdiv_fp16_spatz(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile rowdiv_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, S);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(ROWDIV_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_rowdiv_fp16_spatz_l1_start(rowdiv_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      const float16 *S,
                                      float16 *Y,
                                      uint32_t rows,
                                      uint32_t cols,
                                      mg_event_t *event)
{
    rowdiv_fp16_spatz_params_set(p, X, S, Y, rows, cols);
    mg_spatz_run(WFE, ROWDIV_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_rowdiv_fp16_spatz_l1(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols)
{
    rowdiv_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_rowdiv_fp16_spatz_l1_start(&params, X, S, Y, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
