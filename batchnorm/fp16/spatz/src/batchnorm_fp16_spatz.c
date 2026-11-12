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

#include "batchnorm_fp16_spatz.h"
#include "batchnorm_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "batchnorm_fp16_spatz"

static int alloc_l1(void **params, uint32_t shape[4])
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard_X;
    uint32_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t mean;
    uintptr_t var;
    uintptr_t eps;

    uint32_t c_start;
    uint32_t c_len;

    size_t B;
    size_t C;
    size_t HW;
    size_t iterations;

    B          = shape[0];
    C          = shape[1];
    iterations = B * C;
    HW         = shape[2] * shape[3];

    mg_split(iterations, NUM_HARTS, HID, &c_start, &c_len);

    l1_alloc_init();

    batchnorm_params = l1_alloc(sizeof(batchnorm_fp16_spatz_params_t));
    if (!batchnorm_params)
        return ENOMEM;

    shard_X = l1_alloc(c_len * HW * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(c_len * HW * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gamma = l1_alloc(C * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(C * sizeof(float16));
    if (!beta)
        return ENOMEM;

    mean = l1_alloc(C * sizeof(float16));
    if (!mean)
        return ENOMEM;

    var = l1_alloc(C * sizeof(float16));
    if (!var)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    batchnorm_params->shard_X  = shard_X;
    batchnorm_params->shard_Y  = shard_Y;
    batchnorm_params->gamma    = gamma;
    batchnorm_params->beta     = beta;
    batchnorm_params->mean     = mean;
    batchnorm_params->var      = var;
    batchnorm_params->eps      = eps;
    batchnorm_params->c_start  = c_start;
    batchnorm_params->c_len    = c_len;
    batchnorm_params->hw_len   = HW;
    batchnorm_params->channels = C;

    *params = (void *)batchnorm_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *X,
                             const float16 *scale,
                             const float16 *B,
                             const float16 *input_mean,
                             const float16 *input_var,
                             const float16 epsilon)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t channels;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t hw;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *)params;
    channels         = batchnorm_params->channels;
    c_start          = batchnorm_params->c_start;
    c_len            = batchnorm_params->c_len;
    hw               = batchnorm_params->hw_len;

    mmio_fp16(batchnorm_params->eps) = epsilon;

    if (c_len == 0)
        return 0;

    /* X: this tile's [c_start, c_end) planes are contiguous in L2 (NCHW). The Spatz task
       writes every output element, so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + c_start * hw),
                           (uint32_t)batchnorm_params->shard_X,
                           c_len * hw * sizeof(float16));

    /* Per-channel params (full C, contiguous). Every tile needs them all (channel index
       wraps mod C). */
    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)input_mean, (uint32_t)batchnorm_params->mean, channels * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)input_var, (uint32_t)batchnorm_params->var, channels * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)scale, (uint32_t)batchnorm_params->gamma, channels * sizeof(float16));

    mg_idma_memcpy_1d_sync(
        WFE, 0, (uint32_t)B, (uint32_t)batchnorm_params->beta, channels * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t hw;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *)params;
    c_start          = batchnorm_params->c_start;
    c_len            = batchnorm_params->c_len;
    hw               = batchnorm_params->hw_len;

    if (c_len == 0)
        return 0;

    /* This tile's output planes [c_start, c_end) are contiguous in L2 (NCHW). */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + c_start * hw),
                           (uint32_t)batchnorm_params->shard_Y,
                           c_len * hw * sizeof(float16));

    return 0;
}

void MAGIA_batchnorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 *input_mean,
                                const float16 *input_var,
                                const float16 epsilon,
                                float16 *Y,
                                uint32_t shape[4])
{
    int ret;
    volatile batchnorm_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, shape));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in,
               ret,
               init_input_params((void *)params, X, scale, B, input_mean, input_var, epsilon));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(BATCHNORM_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_batchnorm_fp16_spatz_l1_start(batchnorm_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         const float16 *scale,
                                         const float16 *B,
                                         const float16 *input_mean,
                                         const float16 *input_var,
                                         const float16 epsilon,
                                         float16 *Y,
                                         uint32_t shape[4],
                                         mg_event_t *event)
{
    batchnorm_fp16_spatz_params_set(p, X, scale, B, input_mean, input_var, epsilon, Y, shape);
    mg_spatz_run(WFE, BATCHNORM_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_batchnorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 *input_mean,
                                   const float16 *input_var,
                                   const float16 epsilon,
                                   float16 *Y,
                                   uint32_t shape[4])
{
    /* In one 64-byte line: on lvsoc the task, which reads the block at every channel, runs about
       6% slower when the block straddles two. */
    batchnorm_fp16_spatz_params_t params __attribute__((aligned(64)));
    mg_event_t event;

    MAGIA_batchnorm_fp16_spatz_l1_start(
        &params, X, scale, B, input_mean, input_var, epsilon, Y, shape, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
