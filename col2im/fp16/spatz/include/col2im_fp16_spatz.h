// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef COL2IM_FP16_SPATZ_H_
#define COL2IM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "col2im_fp16_spatz_params.h"

/*
 * Col2Im of the columns `input` into the image `output`, in L2, with the channels split over the
 * tiles.
 */
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
                             uint32_t l_len);

/* Same operation on this tile only, with input, output already in this tile's L1. */
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
                                uint32_t l_len);

/*
 * As MAGIA_col2im_fp16_spatz_l1 (it zeroes `output` first), but only starts the Spatz task, with
 * the parameter block `p` (which must stay valid until it completes): mg_spatz_wait(mode, event)
 * waits for it.
 */
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
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_col2im_fp16_spatz_l1_start. */
static inline void col2im_fp16_spatz_params_set(col2im_fp16_spatz_params_t *p,
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
                                                uint32_t l_len)
{
    p->shard_X        = (uintptr_t)input;
    p->shard_Y        = (uintptr_t)output;
    p->batch          = batch;
    p->total_channels = channels;
    p->c_start        = 0;
    p->c_len          = channels;
    p->image_h        = image_h;
    p->image_w        = image_w;
    p->block_h        = block_h;
    p->block_w        = block_w;
    p->pad_h          = pad_h;
    p->pad_w          = pad_w;
    p->stride_h       = stride_h;
    p->stride_w       = stride_w;
    p->dilation_h     = dilation_h;
    p->dilation_w     = dilation_w;
    p->l_len          = l_len;
}

#endif /* COL2IM_FP16_SPATZ_H_ */
