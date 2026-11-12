// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "expminus_rowsum_fp16_spatz.h"
#include "expminus_rowsum_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "expminus_rowsum_fp16_spatz"

static int alloc_l1(void **params, const float16 *A, float16 s, uint32_t rows, uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uintptr_t shard_X;
    uintptr_t shard_M;
    uintptr_t shard_S;
    uintptr_t shard_A = 0;
    uint32_t row_start;
    uint32_t row_len;

    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    l1_alloc_init();

    er_params = l1_alloc(sizeof(expminus_rowsum_fp16_spatz_params_t));
    if (!er_params)
        return ENOMEM;

    /* The task may write Y over X: one shard for X and Y. */
    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_M = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    if (A) {
        shard_A = (uintptr_t)l1_alloc(row_len * sizeof(float16));
        if (!shard_A)
            return ENOMEM;
    }

    er_params->shard_X = shard_X;
    er_params->shard_M = shard_M;
    er_params->shard_Y = shard_X;
    er_params->shard_S = shard_S;
    er_params->shard_A = shard_A;
    expminus_rowsum_fp16_spatz_set_scale(er_params, s, row_len, cols);

    *params = (void *)er_params;

    return 0;
}

static int
init_input_params(void *params, const float16 *X, const float16 *M, const float16 *A, uint32_t rows)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    er_params = (expminus_rowsum_fp16_spatz_params_t *)params;
    cols      = er_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X, and so are their entries in M and A. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + row_start * cols),
                           (uint32_t)er_params->shard_X,
                           row_len * cols * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(M + row_start),
                           (uint32_t)er_params->shard_M,
                           row_len * sizeof(float16));

    if (A) {
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(A + row_start),
                               (uint32_t)er_params->shard_A,
                               row_len * sizeof(float16));
    }

    return 0;
}

void MAGIA_expminus_rowsum_fp16_spatz_l1(const float16 *X,
                                         const float16 *M,
                                         float16 *Y,
                                         float16 *S,
                                         float16 *A,
                                         float16 s,
                                         uint32_t rows,
                                         uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_M = (uintptr_t)M,
        .shard_Y = (uintptr_t)Y,
        .shard_S = (uintptr_t)S,
        .shard_A = (uintptr_t)A,
    };
    int ret;

    expminus_rowsum_fp16_spatz_set_scale(&params, s, rows, cols);

    ret = mg_spatz_call(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, &params);
    if (ret != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
}

static int store_result(void *params, float16 *Y, float16 *S, float16 *A, uint32_t rows)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    er_params = (expminus_rowsum_fp16_spatz_params_t *)params;
    cols      = er_params->dims >> 16;
    mg_split(rows, NUM_HARTS, HID, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + row_start * cols),
                           (uint32_t)er_params->shard_Y,
                           row_len * cols * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(S + row_start),
                           (uint32_t)er_params->shard_S,
                           row_len * sizeof(float16));

    /* A is updated in place. */
    if (A) {
        mg_idma_memcpy_1d_sync(WFE,
                               1,
                               (uint32_t)(A + row_start),
                               (uint32_t)er_params->shard_A,
                               row_len * sizeof(float16));
    }

    return 0;
}

void MAGIA_expminus_rowsum_fp16_spatz(const float16 *X,
                                      const float16 *M,
                                      float16 *Y,
                                      float16 *S,
                                      float16 *A,
                                      float16 s,
                                      uint32_t rows,
                                      uint32_t cols)
{
    int ret;
    expminus_rowsum_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, A, s, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, M, A, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y, S, A, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
