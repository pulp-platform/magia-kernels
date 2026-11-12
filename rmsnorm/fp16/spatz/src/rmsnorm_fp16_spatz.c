// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "rmsnorm_fp16_spatz.h"
#include "rmsnorm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rmsnorm_fp16_spatz"

static int alloc_l1(void **params, const float16 epsilon, uint32_t rows, uint32_t w_len)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t shard_X;
    uint32_t gamma;
    uint32_t eps;

    uint32_t r_start;
    uint32_t r_len;

    mg_split(rows, NUM_HARTS, HID, &r_start, &r_len);

    l1_alloc_init();

    rms_params = l1_alloc(sizeof(rmsnorm_fp16_spatz_params_t));
    if (!rms_params)
        return ENOMEM;

    /* The task normalizes the rows in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(r_len * w_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    gamma = (uint32_t)l1_alloc(w_len * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    eps = (uint32_t)l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    /* epsilon is a single element: the CV32 copies it to L1 itself, no iDMA needed. */
    mmio_fp16(eps) = epsilon;

    rms_params->shard_X = shard_X;
    rms_params->shard_Y = shard_X;
    rms_params->gamma   = gamma;
    rms_params->eps     = eps;
    rms_params->r_start = r_start;
    rms_params->r_len   = r_len;
    rms_params->w_len   = w_len;

    *params = (void *)rms_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *scale)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    rms_params = (volatile rmsnorm_fp16_spatz_params_t *)params;
    r_start    = rms_params->r_start;
    r_len      = rms_params->r_len;
    w_len      = rms_params->w_len;

    if (r_len == 0)
        return 0;

    /* This tile's rows [r_start, r_start+r_len) are contiguous in X. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + r_start * w_len),
                           rms_params->shard_X,
                           r_len * w_len * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE, 0, (uint32_t)scale, rms_params->gamma, w_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile rmsnorm_fp16_spatz_params_t *rms_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    rms_params = (volatile rmsnorm_fp16_spatz_params_t *)params;
    r_start    = rms_params->r_start;
    r_len      = rms_params->r_len;
    w_len      = rms_params->w_len;

    if (r_len == 0)
        return 0;

    /* This tile's output rows are contiguous in Y. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + r_start * w_len),
                           rms_params->shard_Y,
                           r_len * w_len * sizeof(float16));

    return 0;
}

void MAGIA_rmsnorm_fp16_spatz(const float16 *X,
                              const float16 *scale,
                              const float16 epsilon,
                              uint32_t rows,
                              uint32_t w_len,
                              float16 *Y)
{
    int ret;
    volatile rmsnorm_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, epsilon, rows, w_len);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, scale);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(RMSNORM_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_rmsnorm_fp16_spatz_l1_start(rmsnorm_fp16_spatz_params_t *p,
                                       const float16 *X,
                                       const float16 *scale,
                                       const float16 *epsilon,
                                       uint32_t rows,
                                       uint32_t w_len,
                                       float16 *Y,
                                       mg_event_t *event)
{
    rmsnorm_fp16_spatz_params_set(p, X, scale, epsilon, rows, w_len, Y);
    mg_spatz_run(WFE, RMSNORM_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_rmsnorm_fp16_spatz_l1(const float16 *X,
                                 const float16 *scale,
                                 const float16 *epsilon,
                                 uint32_t rows,
                                 uint32_t w_len,
                                 float16 *Y)
{
    rmsnorm_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_rmsnorm_fp16_spatz_l1_start(&params, X, scale, epsilon, rows, w_len, Y, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
