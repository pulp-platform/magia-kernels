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

#include "col2im_fp16_spatz.h"
#include "col2im_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "col2im_fp16_spatz"

static int allocate_l1(void **params,
                       uint32_t batch,
                       uint32_t channels,
                       uint32_t image_h,
                       uint32_t image_w,
                       uint32_t block_h,
                       uint32_t block_w,
                       uint32_t pad_h,
                       uint32_t pad_w,
                       uint32_t stride_h,
                       uint32_t stride_w,
                       uint32_t dilation_h,
                       uint32_t dilation_w,
                       uint32_t l_len)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t c_start;
    uint32_t c_len;
    uint32_t input_elems;
    uint32_t output_elems;

    mg_split(channels, NUM_HARTS, HID, &c_start, &c_len);

    l1_alloc_init();

    col2im_params = l1_alloc(sizeof(col2im_fp16_spatz_params_t));
    if (!col2im_params)
        return ENOMEM;

    /* shard_X_size: B * shard_C_len * block_h * block_w * l_len  */
    input_elems = batch * c_len * block_h * block_w * l_len;
    shard_X     = l1_alloc(input_elems * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    /* shard_Y_size: B * C_len * image_h * image_w */
    output_elems = batch * c_len * image_h * image_w;
    shard_Y      = l1_alloc(output_elems * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    col2im_params->shard_X        = shard_X;
    col2im_params->shard_Y        = shard_Y;
    col2im_params->total_channels = channels;
    col2im_params->batch          = batch;
    col2im_params->dilation_h     = dilation_h;
    col2im_params->dilation_w     = dilation_w;
    col2im_params->stride_h       = stride_h;
    col2im_params->stride_w       = stride_w;
    col2im_params->image_h        = image_h;
    col2im_params->image_w        = image_w;
    col2im_params->block_h        = block_h;
    col2im_params->block_w        = block_w;
    col2im_params->pad_h          = pad_h;
    col2im_params->pad_w          = pad_w;
    col2im_params->c_start        = c_start;
    col2im_params->c_len          = c_len;
    col2im_params->l_len          = l_len;

    *params = (void *)col2im_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uint32_t image_h;
    uint32_t image_w;
    uint32_t block_h;
    uint32_t block_w;
    uint32_t c_start;
    uint32_t batch;
    uint32_t c_len;
    uint32_t l_len;
    uint32_t tot_c;
    uint32_t bbl;

    col2im_params = (volatile col2im_fp16_spatz_params_t *)params;

    shard_X = col2im_params->shard_X;
    shard_Y = col2im_params->shard_Y;
    image_h = col2im_params->image_h;
    image_w = col2im_params->image_w;
    block_h = col2im_params->block_h;
    block_w = col2im_params->block_w;
    c_start = col2im_params->c_start;
    batch   = col2im_params->batch;
    c_len   = col2im_params->c_len;
    l_len   = col2im_params->l_len;
    tot_c   = col2im_params->total_channels;

    if (c_len == 0)
        return 0;

    bbl = block_h * block_w * l_len;

    /* Column buffer: for each batch this tile's channels [c_start, c_start+c_len) are a
       contiguous run of c_len*bbl elements. Gather them into the packed L1 shard with a 2D
       transfer -- one run per batch, source stride spanning the full channel count. */
    mg_idma_memcpy_2d_sync(WFE,
                           0,
                           (uint32_t)(input + c_start * bbl),
                           (uint32_t)shard_X,
                           c_len * bbl * sizeof(float16),
                           tot_c * bbl * sizeof(float16),
                           batch);

    /* col2im scatters with accumulation, so the output shard must start zeroed. */
    for (uint32_t i = 0; i < (batch * c_len * image_h * image_w); i++) {
        mmio_fp16(shard_Y + (i * sizeof(float16))) = 0;
    }

    return 0;
}

static int store_result(void *params, float16 *output)
{
    volatile col2im_fp16_spatz_params_t *col2im_params;
    uintptr_t shard_Y;
    uint32_t image_h;
    uint32_t image_w;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t batch;
    uint32_t tot_c;
    uint32_t ihw;

    col2im_params = (volatile col2im_fp16_spatz_params_t *)params;

    shard_Y = col2im_params->shard_Y;
    image_h = col2im_params->image_h;
    image_w = col2im_params->image_w;
    c_start = col2im_params->c_start;
    c_len   = col2im_params->c_len;
    batch   = col2im_params->batch;
    tot_c   = col2im_params->total_channels;

    if (c_len == 0)
        return 0;

    ihw = image_h * image_w;

    /* For each batch this tile's output channels [c_start, c_start+c_len) are a contiguous
       run of c_len*ihw elements in L2. Scatter the packed L1 shard back with a 2D transfer
       -- one run per batch, dest stride spanning the full channel count. */
    mg_idma_memcpy_2d_sync(WFE,
                           1,
                           (uint32_t)(output + c_start * ihw),
                           (uint32_t)shard_Y,
                           c_len * ihw * sizeof(float16),
                           tot_c * ihw * sizeof(float16),
                           batch);

    return 0;
}

void MAGIA_col2im_fp16_spatz(const float16 *input,
                             float16 *output,
                             uint32_t batch,
                             uint32_t channels,
                             uint32_t image_h,
                             uint32_t image_w,
                             uint32_t block_h,
                             uint32_t block_w,
                             uint32_t pad_h,
                             uint32_t pad_w,
                             uint32_t stride_h,
                             uint32_t stride_w,
                             uint32_t dilation_h,
                             uint32_t dilation_w,
                             uint32_t l_len)
{
    int ret;
    volatile col2im_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc,
               ret,
               allocate_l1((void **)&params,
                           batch,
                           channels,
                           image_h,
                           image_w,
                           block_h,
                           block_w,
                           pad_h,
                           pad_w,
                           stride_h,
                           stride_w,
                           dilation_h,
                           dilation_w,
                           l_len));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, input));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(COL2IM_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_col2im_fp16_spatz_l1_start(col2im_fp16_spatz_params_t *p,
                                      const float16 *input,
                                      float16 *output,
                                      uint32_t batch,
                                      uint32_t channels,
                                      uint32_t image_h,
                                      uint32_t image_w,
                                      uint32_t block_h,
                                      uint32_t block_w,
                                      uint32_t pad_h,
                                      uint32_t pad_w,
                                      uint32_t stride_h,
                                      uint32_t stride_w,
                                      uint32_t dilation_h,
                                      uint32_t dilation_w,
                                      uint32_t l_len,
                                      mg_event_t *event)
{
    col2im_fp16_spatz_params_set(p,
                                 input,
                                 output,
                                 batch,
                                 channels,
                                 image_h,
                                 image_w,
                                 block_h,
                                 block_w,
                                 pad_h,
                                 pad_w,
                                 stride_h,
                                 stride_w,
                                 dilation_h,
                                 dilation_w,
                                 l_len);

    /* col2im scatters with accumulation, so the output must start zeroed. */
    for (uint32_t i = 0; i < batch * channels * image_h * image_w; i++)
        mmio_fp16((uintptr_t)(output + i)) = 0;

    mg_spatz_run(WFE, COL2IM_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_col2im_fp16_spatz_l1(const float16 *input,
                                float16 *output,
                                uint32_t batch,
                                uint32_t channels,
                                uint32_t image_h,
                                uint32_t image_w,
                                uint32_t block_h,
                                uint32_t block_w,
                                uint32_t pad_h,
                                uint32_t pad_w,
                                uint32_t stride_h,
                                uint32_t stride_w,
                                uint32_t dilation_h,
                                uint32_t dilation_w,
                                uint32_t l_len)
{
    col2im_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_col2im_fp16_spatz_l1_start(&params,
                                     input,
                                     output,
                                     batch,
                                     channels,
                                     image_h,
                                     image_w,
                                     block_h,
                                     block_w,
                                     pad_h,
                                     pad_w,
                                     stride_h,
                                     stride_w,
                                     dilation_h,
                                     dilation_w,
                                     l_len,
                                     &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
