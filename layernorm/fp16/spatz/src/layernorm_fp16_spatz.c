// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "kernels_profiling_utils.h"

#include "layernorm_fp16_spatz.h"
#include "layernorm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "layernorm_fp16_spatz"

static int alloc_l1(void **params, uint32_t *input_shape, uint32_t rank, float16 epsilon)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t eps;

    uint32_t total_rows;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;
    uint32_t local_shard_size;

    /* LayerNorm normalizes over the last axis (axis == -1): the last dimension
       is the row length, the product of the leading dimensions is the number of
       rows. This is rank-agnostic (any tensor rank). */
    w_len      = input_shape[rank - 1];
    total_rows = 1;
    for (uint32_t d = 0; d < rank - 1; d++)
        total_rows *= input_shape[d];

    mg_split(total_rows, NUM_HARTS, HID, &r_start, &r_len);

    local_shard_size = r_len * w_len;

    l1_alloc_init();

    layernorm_params = l1_alloc(sizeof(layernorm_fp16_spatz_params_t));
    if (!layernorm_params)
        return ENOMEM;

    shard_X = l1_alloc(local_shard_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(local_shard_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gamma = l1_alloc(w_len * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(w_len * sizeof(float16));
    if (!beta)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    layernorm_params->shard_X = shard_X;
    layernorm_params->shard_Y = shard_Y;
    layernorm_params->gamma   = gamma;
    layernorm_params->beta    = beta;
    layernorm_params->eps     = eps;
    layernorm_params->r_start = r_start;
    layernorm_params->r_len   = r_len;
    layernorm_params->w_len   = w_len;

    *params = (void *)layernorm_params;

    return 0;
}

static int init_input_params(
    void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uint32_t r_start;
    uint32_t w_len;
    uint32_t rows_bytes;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *)params;
    r_start          = layernorm_params->r_start;
    w_len            = layernorm_params->w_len;
    rows_bytes       = layernorm_params->r_len * w_len * sizeof(float16);

    /* This tile's rows [r_start, r_start+r_len) are contiguous in L2, so X, gamma,
       beta and the output shard are plain 1D transfers. The Spatz task overwrites
       the whole output shard, so it does not need zeroing here. */
    if (rows_bytes > 0) {
        mg_idma_memcpy_1d_sync(
            WFE, 0, (uint32_t)(X + r_start * w_len), layernorm_params->shard_X, rows_bytes);
    }

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)scale, layernorm_params->gamma, w_len * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE, 0, (uint32_t)B, layernorm_params->beta, w_len * sizeof(float16));

    mmio_fp16(layernorm_params->eps) = epsilon;

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uint32_t r_start;
    uint32_t w_len;
    uint32_t rows_bytes;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *)params;
    r_start          = layernorm_params->r_start;
    w_len            = layernorm_params->w_len;
    rows_bytes       = layernorm_params->r_len * w_len * sizeof(float16);

    if (rows_bytes == 0)
        return 0;

    /* This tile's output rows are contiguous in L2 -> single 1D transfer (L1->L2). */
    mg_idma_memcpy_1d_sync(
        WFE, 1, (uint32_t)(Y + r_start * w_len), layernorm_params->shard_Y, rows_bytes);

    return 0;
}

void MAGIA_layernorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 epsilon,
                                uint32_t *input_shape,
                                uint32_t rank,
                                float16 *Y)
{
    int ret;
    volatile layernorm_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, input_shape, rank, epsilon));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X, scale, B, epsilon));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(LAYERNORM_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_layernorm_fp16_spatz_l1_start(layernorm_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         const float16 *scale,
                                         const float16 *B,
                                         const float16 epsilon,
                                         uint32_t *input_shape,
                                         uint32_t rank,
                                         float16 *Y,
                                         mg_event_t *event)
{
    layernorm_fp16_spatz_params_set(p, X, scale, B, epsilon, input_shape, rank, Y);
    mg_spatz_run(WFE, LAYERNORM_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_layernorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 epsilon,
                                   uint32_t *input_shape,
                                   uint32_t rank,
                                   float16 *Y)
{
    layernorm_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_layernorm_fp16_spatz_l1_start(
        &params, X, scale, B, epsilon, input_shape, rank, Y, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
