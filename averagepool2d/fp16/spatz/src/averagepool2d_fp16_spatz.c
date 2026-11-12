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

#include "averagepool2d_fp16_spatz.h"
#include "averagepool2d_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "averagepool2d_fp16_spatz"

static int alloc_l1(void **params,
                    uint32_t input_shape[4],
                    uint32_t output_shape[4],
                    uint32_t kernel_h,
                    uint32_t kernel_w,
                    uint32_t stride_h,
                    uint32_t stride_w,
                    uint32_t pad_h,
                    uint32_t pad_w)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t input_B;
    size_t input_C;
    size_t input_H;
    size_t input_W;
    size_t output_H;
    size_t output_W;
    size_t iterations;
    size_t input_HW_len;
    size_t output_HW_len;

    uint32_t c_start;
    uint32_t c_len;

    input_B    = input_shape[0];
    input_C    = input_shape[1];
    iterations = input_B * input_C;

    input_H  = input_shape[2];
    input_W  = input_shape[3];
    output_H = output_shape[2];
    output_W = output_shape[3];

    mg_split(iterations, NUM_HARTS, HID, &c_start, &c_len);

    input_HW_len  = input_H * input_W;
    output_HW_len = output_H * output_W;

    l1_alloc_init();

    averagepool_params = l1_alloc(sizeof(averagepool2d_fp16_spatz_params_t));
    if (!averagepool_params)
        return ENOMEM;

    shard_X = l1_alloc(c_len * input_HW_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(c_len * output_HW_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    averagepool_params->shard_X  = shard_X;
    averagepool_params->shard_Y  = shard_Y;
    averagepool_params->h_out    = output_H;
    averagepool_params->w_out    = output_W;
    averagepool_params->h_in     = input_H;
    averagepool_params->w_in     = input_W;
    averagepool_params->c_start  = c_start;
    averagepool_params->c_len    = c_len;
    averagepool_params->kernel_h = kernel_h;
    averagepool_params->kernel_w = kernel_w;
    averagepool_params->stride_h = stride_h;
    averagepool_params->stride_w = stride_w;
    averagepool_params->pad_h    = pad_h;
    averagepool_params->pad_w    = pad_w;

    *params = (void *)averagepool_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t in_hw_len;

    averagepool_params = (volatile averagepool2d_fp16_spatz_params_t *)params;
    c_start            = averagepool_params->c_start;
    c_len              = averagepool_params->c_len;
    in_hw_len          = averagepool_params->h_in * averagepool_params->w_in;

    if (c_len == 0)
        return 0;

    /* This tile's channels [c_start, c_end) are contiguous in L2 (NCHW). The Spatz
       task writes every output element, so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + c_start * in_hw_len),
                           (uint32_t)averagepool_params->shard_X,
                           c_len * in_hw_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t out_hw_len;

    averagepool_params = (volatile averagepool2d_fp16_spatz_params_t *)params;
    c_start            = averagepool_params->c_start;
    c_len              = averagepool_params->c_len;
    out_hw_len         = averagepool_params->h_out * averagepool_params->w_out;

    if (c_len == 0)
        return 0;

    /* This tile's output channels [c_start, c_end) are contiguous in L2 (NCHW). */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + c_start * out_hw_len),
                           (uint32_t)averagepool_params->shard_Y,
                           c_len * out_hw_len * sizeof(float16));

    return 0;
}

void MAGIA_averagepool2d_fp16_spatz(const float16 *X,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w)
{
    int ret;
    volatile averagepool2d_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc,
               ret,
               alloc_l1((void **)&params,
                        input_shape,
                        output_shape,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, X));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(
        prof_cyc_compute, ret, mg_spatz_call(AVERAGEPOOL2D_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_averagepool2d_fp16_spatz_l1_start(averagepool2d_fp16_spatz_params_t *p,
                                             const float16 *X,
                                             float16 *Y,
                                             uint32_t input_shape[4],
                                             uint32_t output_shape[4],
                                             uint32_t kernel_h,
                                             uint32_t kernel_w,
                                             uint32_t stride_h,
                                             uint32_t stride_w,
                                             uint32_t pad_h,
                                             uint32_t pad_w,
                                             mg_event_t *event)
{
    averagepool2d_fp16_spatz_params_set(
        p, X, Y, input_shape, output_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    mg_spatz_run(WFE, AVERAGEPOOL2D_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_averagepool2d_fp16_spatz_l1(const float16 *X,
                                       float16 *Y,
                                       uint32_t input_shape[4],
                                       uint32_t output_shape[4],
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t pad_h,
                                       uint32_t pad_w)
{
    averagepool2d_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_averagepool2d_fp16_spatz_l1_start(&params,
                                            X,
                                            Y,
                                            input_shape,
                                            output_shape,
                                            kernel_h,
                                            kernel_w,
                                            stride_h,
                                            stride_w,
                                            pad_h,
                                            pad_w,
                                            &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
