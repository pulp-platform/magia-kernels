// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "rowmax_fp16_spatz.h"
#include "rowmax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowmax_fp16_spatz"

static int alloc_l1(void **params, const float16 *P, uint32_t rows, uint32_t cols)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uintptr_t shard_X;
    uintptr_t shard_P = 0;
    uintptr_t shard_M;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    rm_params = l1_alloc(sizeof(rowmax_fp16_spatz_params_t));
    if (!rm_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    if (P) {
        shard_P = (uintptr_t)l1_alloc(row_len * sizeof(float16));
        if (!shard_P)
            return ENOMEM;
    }

    shard_M = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    rm_params->shard_X = shard_X;
    rm_params->shard_P = shard_P;
    rm_params->shard_M = shard_M;
    rm_params->dims    = row_len | (cols << 16);

    *params = (void *)rm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *P, uint32_t rows)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rm_params = (volatile rowmax_fp16_spatz_params_t *)params;
    cols      = rm_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X, and so are their previous maxima in P. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)rm_params->shard_X,
                           row_len * cols * sizeof(float16));

    if (P) {
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(P + row_start),
                               (uint32_t)rm_params->shard_P,
                               row_len * sizeof(float16));
    }

    return 0;
}

static int store_result(void *params, float16 *M, uint32_t rows)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uint32_t row_start;
    uint32_t row_len;

    rm_params = (volatile rowmax_fp16_spatz_params_t *)params;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(M + row_start),
                           (uint32_t)rm_params->shard_M,
                           row_len * sizeof(float16));

    return 0;
}

void MAGIA_rowmax_fp16_spatz(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile rowmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, P, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, P, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(ROWMAX_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, M, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_rowmax_fp16_spatz_l1_start(rowmax_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      const float16 *P,
                                      float16 *M,
                                      uint32_t rows,
                                      uint32_t cols,
                                      mg_event_t *event)
{
    rowmax_fp16_spatz_params_set(p, X, P, M, rows, cols);
    mg_spatz_run(WFE, ROWMAX_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_rowmax_fp16_spatz_l1(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols)
{
    rowmax_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_rowmax_fp16_spatz_l1_start(&params, X, P, M, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
