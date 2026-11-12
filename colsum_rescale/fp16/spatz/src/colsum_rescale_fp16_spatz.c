// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "colsum_rescale_fp16_spatz.h"
#include "colsum_rescale_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "colsum_rescale_fp16_spatz"

/*
 * This tile's columns [*col_start, *col_start + *col_len) out of cols, split in pairs so that
 * each tile gets an even number of columns; an odd last column goes to the last tile.
 */
static void tile_cols(uint32_t cols, uint32_t *col_start, uint32_t *col_len)
{
    mg_split(cols / 2, NUM_HARTS, HID, col_start, col_len);
    *col_start *= 2;
    *col_len *= 2;
    if (HID == NUM_HARTS - 1)
        *col_len += cols & 1;
}

static int alloc_l1(void **params, const float16 *P, uint32_t rows, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uintptr_t shard_S;
    uintptr_t shard_G;
    uintptr_t shard_P = 0;
    uintptr_t shard_A = 0;
    uint32_t col_start;
    uint32_t col_len;

    tile_cols(cols, &col_start, &col_len);

    l1_alloc_init();

    cs_params = l1_alloc(sizeof(colsum_rescale_fp16_spatz_params_t));
    if (!cs_params)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(col_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    shard_G = (uintptr_t)l1_alloc(rows * col_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    if (P) {
        shard_P = (uintptr_t)l1_alloc(col_len * sizeof(float16));
        if (!shard_P)
            return ENOMEM;

        shard_A = (uintptr_t)l1_alloc(col_len * sizeof(float16));
        if (!shard_A)
            return ENOMEM;
    }

    cs_params->shard_S = shard_S;
    cs_params->shard_G = shard_G;
    cs_params->shard_P = shard_P;
    cs_params->shard_A = shard_A;
    cs_params->dims    = rows | (col_len << 16);

    *params = (void *)cs_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *S,
                             const float16 *G,
                             const float16 *P,
                             const float16 *A,
                             uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uint32_t col_start;
    uint32_t col_len;
    uint32_t rows;

    cs_params = (volatile colsum_rescale_fp16_spatz_params_t *)params;
    rows      = cs_params->dims & 0xffff;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(S + col_start),
                           (uint32_t)cs_params->shard_S,
                           col_len * sizeof(float16));

    /*
     * This tile's columns of every row of G, packed into a rows x col_len matrix (row 0 is
     * unused by the task, but keeps the row indexing).
     */
    if (rows > 0) {
        mg_idma_memcpy_2d_sync(WFE,
                               0,
                               (uint32_t)(G + col_start),
                               (uint32_t)cs_params->shard_G,
                               col_len * sizeof(float16),
                               cols * sizeof(float16),
                               rows);
    }

    if (P) {
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(P + col_start),
                               (uint32_t)cs_params->shard_P,
                               col_len * sizeof(float16));

        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(A + col_start),
                               (uint32_t)cs_params->shard_A,
                               col_len * sizeof(float16));
    }

    return 0;
}

static int store_result(void *params, float16 *S, float16 *P, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uint32_t col_start;
    uint32_t col_len;

    cs_params = (volatile colsum_rescale_fp16_spatz_params_t *)params;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(S + col_start),
                           (uint32_t)cs_params->shard_S,
                           col_len * sizeof(float16));

    /* P is rescaled in place. */
    if (P) {
        mg_idma_memcpy_1d_sync(WFE,
                               1,
                               (uint32_t)(P + col_start),
                               (uint32_t)cs_params->shard_P,
                               col_len * sizeof(float16));
    }

    return 0;
}

void MAGIA_colsum_rescale_fp16_spatz(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile colsum_rescale_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, P, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, S, G, P, A, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(COLSUM_RESCALE_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, S, P, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_colsum_rescale_fp16_spatz_l1_start(colsum_rescale_fp16_spatz_params_t *p,
                                              float16 *S,
                                              const float16 *G,
                                              float16 *P,
                                              const float16 *A,
                                              uint32_t rows,
                                              uint32_t cols,
                                              mg_event_t *event)
{
    colsum_rescale_fp16_spatz_params_set(p, S, G, P, A, rows, cols);
    mg_spatz_run(WFE, COLSUM_RESCALE_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols)
{
    colsum_rescale_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_colsum_rescale_fp16_spatz_l1_start(&params, S, G, P, A, rows, cols, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
