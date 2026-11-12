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

#include "maxpool2d_fp16_spatz.h"
#include "maxpool2d_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "maxpool2d_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4])
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t total_channels;
    uint32_t c_start;
    uint32_t c_len;

    size_t in_hw_len;
    size_t out_hw_len;
    size_t local_in_size;
    size_t local_out_size;

    total_channels = input_shape[0] * input_shape[1];
    mg_split(total_channels, NUM_HARTS, HID, &c_start, &c_len);

    in_hw_len  = input_shape[2] * input_shape[3];
    out_hw_len = output_shape[2] * output_shape[3];

    local_in_size  = c_len * in_hw_len;
    local_out_size = c_len * out_hw_len;

    l1_alloc_init();

    maxpool_params = l1_alloc(sizeof(maxpool2d_fp16_spatz_params_t));
    if (!maxpool_params)
        return ENOMEM;

    shard_X = l1_alloc(local_in_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(local_out_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    maxpool_params->shard_X = shard_X;
    maxpool_params->shard_Y = shard_Y;
    maxpool_params->c_start = c_start;
    maxpool_params->c_len   = c_len;
    maxpool_params->h_in    = input_shape[2];
    maxpool_params->w_in    = input_shape[3];
    maxpool_params->h_out   = output_shape[2];
    maxpool_params->w_out   = output_shape[3];

    *params = (void *)maxpool_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *X,
                             uint32_t kernel_h,
                             uint32_t kernel_w,
                             uint32_t stride_h,
                             uint32_t stride_w,
                             uint32_t pad_h,
                             uint32_t pad_w)
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uint32_t in_hw_len;
    uint32_t c_start;
    uint32_t c_len;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *)params;
    in_hw_len      = maxpool_params->h_in * maxpool_params->w_in;
    c_start        = maxpool_params->c_start;
    c_len          = maxpool_params->c_len;

    if (c_len > 0) {
        /* This tile's channels [c_start, c_start+c_len) are contiguous in L2. The Spatz task
           writes every output (one max per window), so shard_Y is not zeroed here. */
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(X + c_start * in_hw_len),
                               maxpool_params->shard_X,
                               c_len * in_hw_len * sizeof(float16));
    }

    maxpool_params->kernel_h = kernel_h;
    maxpool_params->kernel_w = kernel_w;
    maxpool_params->stride_h = stride_h;
    maxpool_params->stride_w = stride_w;
    maxpool_params->pad_h    = pad_h;
    maxpool_params->pad_w    = pad_w;

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uint32_t out_hw_len;
    uint32_t c_start;
    uint32_t c_len;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *)params;
    out_hw_len     = maxpool_params->h_out * maxpool_params->w_out;
    c_start        = maxpool_params->c_start;
    c_len          = maxpool_params->c_len;

    if (c_len == 0)
        return 0;

    /* This tile's output channels [c_start, c_start+c_len) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + c_start * out_hw_len),
                           maxpool_params->shard_Y,
                           c_len * out_hw_len * sizeof(float16));

    return 0;
}

void MAGIA_maxpool2d_fp16_spatz(const float16 *X,
                                float16 *Y,
                                uint32_t kernel_h,
                                uint32_t kernel_w,
                                uint32_t stride_h,
                                uint32_t stride_w,
                                uint32_t pad_h,
                                uint32_t pad_w,
                                uint32_t input_shape[4],
                                uint32_t output_shape[4])
{
    int ret;
    volatile maxpool2d_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, input_shape, output_shape));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_data_in,
        ret,
        init_input_params((void *)params, X, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(MAXPOOL2D_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_maxpool2d_fp16_spatz_l1_start(maxpool2d_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         uint32_t kernel_h,
                                         uint32_t kernel_w,
                                         uint32_t stride_h,
                                         uint32_t stride_w,
                                         uint32_t pad_h,
                                         uint32_t pad_w,
                                         uint32_t input_shape[4],
                                         uint32_t output_shape[4],
                                         mg_event_t *event)
{
    maxpool2d_fp16_spatz_params_set(
        p, X, Y, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, input_shape, output_shape);
    mg_spatz_run(WFE, MAXPOOL2D_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_maxpool2d_fp16_spatz_l1(const float16 *X,
                                   float16 *Y,
                                   uint32_t kernel_h,
                                   uint32_t kernel_w,
                                   uint32_t stride_h,
                                   uint32_t stride_w,
                                   uint32_t pad_h,
                                   uint32_t pad_w,
                                   uint32_t input_shape[4],
                                   uint32_t output_shape[4])
{
    maxpool2d_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_maxpool2d_fp16_spatz_l1_start(&params,
                                        X,
                                        Y,
                                        kernel_h,
                                        kernel_w,
                                        stride_h,
                                        stride_w,
                                        pad_h,
                                        pad_w,
                                        input_shape,
                                        output_shape,
                                        &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
