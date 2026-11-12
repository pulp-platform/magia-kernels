// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "softmax_rows_fp16_spatz.h"
#include "softmax_rows_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "softmax_rows_fp16_spatz"

static int alloc_l1(void **params, float16 s, uint32_t rows, uint32_t cols)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uintptr_t shard_X;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    sm_params = l1_alloc(sizeof(softmax_rows_fp16_spatz_params_t));
    if (!sm_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    sm_params->shard_X  = shard_X;
    sm_params->dims     = row_len | (cols << 16);
    sm_params->min_coef = softmax_rows_fp16_spatz_min_coef(s);

    *params = (void *)sm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, uint32_t rows)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    sm_params = (volatile softmax_rows_fp16_spatz_params_t *)params;
    cols      = sm_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)sm_params->shard_X,
                           row_len * cols * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *X, uint32_t rows)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    sm_params = (volatile softmax_rows_fp16_spatz_params_t *)params;
    cols      = sm_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)sm_params->shard_X,
                           row_len * cols * sizeof(float16));

    return 0;
}

void MAGIA_softmax_rows_fp16_spatz(float16 *X, float16 s, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile softmax_rows_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, s, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(SOFTMAX_ROWS_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, X, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_softmax_rows_fp16_spatz_l1_start(softmax_rows_fp16_spatz_params_t *p,
                                            float16 *X,
                                            float16 s,
                                            uint32_t rows,
                                            uint32_t cols,
                                            mg_event_t *event)
{
    softmax_rows_fp16_spatz_params_set(p, X, s, rows, cols);
    mg_spatz_run(WFE, SOFTMAX_ROWS_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_softmax_rows_fp16_spatz_l1(float16 *X, float16 s, uint32_t rows, uint32_t cols)
{
    softmax_rows_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_softmax_rows_fp16_spatz_l1_start(&params, X, s, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
