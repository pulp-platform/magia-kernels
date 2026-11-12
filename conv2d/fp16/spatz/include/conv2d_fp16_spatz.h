// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONV2D_FP16_SPATZ_H_
#define CONV2D_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "conv2d_fp16_spatz_params.h"

/* Conv2D of the NCHW tensor X into Y, in L2, with the output planes split over the tiles. */
void MAGIA_conv2d_fp16_spatz(const float16 *X,
                             const float16 *W,
                             const float16 *B,
                             float16 *Y,
                             uint32_t input_shape[4],
                             uint32_t output_shape[4],
                             uint32_t kernel_h,
                             uint32_t kernel_w,
                             uint32_t stride_h,
                             uint32_t stride_w,
                             uint32_t pad_h,
                             uint32_t pad_w,
                             uint32_t group,
                             int has_bias);

/* Same operation on this tile only, with X, W, B and Y already in this tile's L1. */
void MAGIA_conv2d_fp16_spatz_l1(const float16 *X,
                                const float16 *W,
                                const float16 *B,
                                float16 *Y,
                                uint32_t input_shape[4],
                                uint32_t output_shape[4],
                                uint32_t kernel_h,
                                uint32_t kernel_w,
                                uint32_t stride_h,
                                uint32_t stride_w,
                                uint32_t pad_h,
                                uint32_t pad_w,
                                uint32_t group,
                                int has_bias);

/*
 * As MAGIA_conv2d_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_conv2d_fp16_spatz_l1_start(conv2d_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      const float16 *W,
                                      const float16 *B,
                                      float16 *Y,
                                      uint32_t input_shape[4],
                                      uint32_t output_shape[4],
                                      uint32_t kernel_h,
                                      uint32_t kernel_w,
                                      uint32_t stride_h,
                                      uint32_t stride_w,
                                      uint32_t pad_h,
                                      uint32_t pad_w,
                                      uint32_t group,
                                      int has_bias,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_conv2d_fp16_spatz_l1_start. */
static inline void conv2d_fp16_spatz_params_set(conv2d_fp16_spatz_params_t *p,
                                                const float16 *X,
                                                const float16 *W,
                                                const float16 *B,
                                                float16 *Y,
                                                uint32_t input_shape[4],
                                                uint32_t output_shape[4],
                                                uint32_t kernel_h,
                                                uint32_t kernel_w,
                                                uint32_t stride_h,
                                                uint32_t stride_w,
                                                uint32_t pad_h,
                                                uint32_t pad_w,
                                                uint32_t group,
                                                int has_bias)
{
    p->shard_X    = (uintptr_t)X;
    p->shard_W    = (uintptr_t)W;
    p->shard_B    = (uintptr_t)B;
    p->shard_Y    = (uintptr_t)Y;
    p->has_bias   = (uint32_t)has_bias;
    p->n_batches  = input_shape[0];
    p->c_out      = output_shape[1];
    p->iter_start = 0;
    p->iter_len   = input_shape[0] * output_shape[1];
    p->c_in_g     = input_shape[1] / group;
    p->c_out_g    = output_shape[1] / group;
    p->h_in       = input_shape[2];
    p->w_in       = input_shape[3];
    p->h_out      = output_shape[2];
    p->w_out      = output_shape[3];
    p->kernel_h   = kernel_h;
    p->kernel_w   = kernel_w;
    p->stride_h   = stride_h;
    p->stride_w   = stride_w;
    p->pad_h      = pad_h;
    p->pad_w      = pad_w;
}

#endif /* CONV2D_FP16_SPATZ_H_ */
