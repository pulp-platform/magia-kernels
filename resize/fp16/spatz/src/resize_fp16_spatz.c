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

#include "resize_fp16_spatz.h"
#include "resize_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "resize_fp16_spatz"

static int alloc_l1(void **params,
                    uint32_t batch_size,
                    uint32_t channels,
                    uint32_t in_h,
                    uint32_t in_w,
                    uint32_t out_h,
                    uint32_t out_w)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t it_start;
    uint32_t it_len;

    mg_split(batch_size * channels, NUM_HARTS, HID, &it_start, &it_len);

    l1_alloc_init();

    resize_params = l1_alloc(sizeof(resize_fp16_spatz_params_t));
    if (!resize_params)
        return ENOMEM;

    shard_X = l1_alloc(it_len * in_h * in_w * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(it_len * out_h * out_w * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    resize_params->shard_X         = shard_X;
    resize_params->shard_Y         = shard_Y;
    resize_params->in_h            = in_h;
    resize_params->in_w            = in_w;
    resize_params->out_h           = out_h;
    resize_params->out_w           = out_w;
    resize_params->iteration_start = it_start;
    resize_params->iteration_len   = it_len;

    *params = (void *)resize_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uint32_t it_start;
    uint32_t it_len;
    uint32_t in_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *)params;
    it_start      = resize_params->iteration_start;
    it_len        = resize_params->iteration_len;
    in_hw         = resize_params->in_h * resize_params->in_w;

    if (it_len == 0)
        return 0;

    /* This tile's (batch,channel) planes [it_start, it_start+it_len) are contiguous in L2.
       The Spatz task writes every output pixel, so shard_Y is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(X + it_start * in_hw),
                           resize_params->shard_X,
                           it_len * in_hw * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uint32_t it_start;
    uint32_t it_len;
    uint32_t out_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *)params;
    it_start      = resize_params->iteration_start;
    it_len        = resize_params->iteration_len;
    out_hw        = resize_params->out_h * resize_params->out_w;

    if (it_len == 0)
        return 0;

    /* This tile's output (batch,channel) planes [it_start, it_start+it_len) are contiguous. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + it_start * out_hw),
                           resize_params->shard_Y,
                           it_len * out_hw * sizeof(float16));

    return 0;
}

void MAGIA_resize_fp16_spatz(const float16 *X,
                             float16 *Y,
                             uint32_t batch_size,
                             uint32_t channels,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t out_h,
                             uint32_t out_w)
{
    int ret;
    volatile resize_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc,
               ret,
               alloc_l1((void **)&params, batch_size, channels, in_h, in_w, out_h, out_w));
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

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(RESIZE_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, Y));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_resize_fp16_spatz_l1_start(resize_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      float16 *Y,
                                      uint32_t batch_size,
                                      uint32_t channels,
                                      uint32_t in_h,
                                      uint32_t in_w,
                                      uint32_t out_h,
                                      uint32_t out_w,
                                      mg_event_t *event)
{
    resize_fp16_spatz_params_set(p, X, Y, batch_size, channels, in_h, in_w, out_h, out_w);
    mg_spatz_run(WFE, RESIZE_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_resize_fp16_spatz_l1(const float16 *X,
                                float16 *Y,
                                uint32_t batch_size,
                                uint32_t channels,
                                uint32_t in_h,
                                uint32_t in_w,
                                uint32_t out_h,
                                uint32_t out_w)
{
    resize_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_resize_fp16_spatz_l1_start(
        &params, X, Y, batch_size, channels, in_h, in_w, out_h, out_w, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
