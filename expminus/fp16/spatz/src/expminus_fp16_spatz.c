// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "expminus_fp16_spatz.h"
#include "expminus_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "expminus_fp16_spatz"

static int alloc_l1(void **params, uint32_t rows, uint32_t cols)
{
    volatile expminus_fp16_spatz_params_t *em_params;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    em_params = l1_alloc(sizeof(expminus_fp16_spatz_params_t));
    if (!em_params)
        return ENOMEM;

    em_params->shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!em_params->shard_X)
        return ENOMEM;

    em_params->shard_M = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!em_params->shard_M)
        return ENOMEM;

    em_params->shard_Y = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!em_params->shard_Y)
        return ENOMEM;

    em_params->cols      = cols;
    em_params->row_start = row_start;
    em_params->row_len   = row_len;

    *params = (void *)em_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *M)
{
    volatile expminus_fp16_spatz_params_t *em_params = params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    row_start = em_params->row_start;
    row_len   = em_params->row_len;
    cols      = em_params->cols;

    if (row_len == 0)
        return 0;

    /*
     * This tile's rows [row_start, row_start + row_len) are contiguous in X, and so are their
     * subtrahends in M. The Spatz task writes every output, so shard_Y is not zeroed here.
     */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           em_params->shard_X,
                           row_len * cols * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)(M + row_start), em_params->shard_M, row_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile expminus_fp16_spatz_params_t *em_params = params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    row_start = em_params->row_start;
    row_len   = em_params->row_len;
    cols      = em_params->cols;

    if (row_len == 0)
        return 0;

    /* This tile's output rows are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + row_start * cols),
                           em_params->shard_Y,
                           row_len * cols * sizeof(float16));

    return 0;
}

void MAGIA_expminus_fp16_spatz(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile expminus_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, M);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(EXPMINUS_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_expminus_fp16_spatz_l1_start(expminus_fp16_spatz_params_t *p,
                                        const float16 *X,
                                        const float16 *M,
                                        float16 *Y,
                                        uint32_t rows,
                                        uint32_t cols,
                                        mg_event_t *event)
{
    expminus_fp16_spatz_params_set(p, X, M, Y, rows, cols);
    mg_spatz_run(WFE, EXPMINUS_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_expminus_fp16_spatz_l1(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols)
{
    expminus_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_expminus_fp16_spatz_l1_start(&params, X, M, Y, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
