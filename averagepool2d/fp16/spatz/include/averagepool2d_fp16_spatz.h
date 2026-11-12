// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AVERAGEPOOL2D_FP16_SPATZ_H_
#define AVERAGEPOOL2D_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "averagepool2d_fp16_spatz_params.h"

/* AveragePool of the NCHW tensor X into Y, in L2, with the channels split over the tiles. */
void MAGIA_averagepool2d_fp16_spatz(const float16 *X,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w);

/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_averagepool2d_fp16_spatz_l1(const float16 *X,
                                       float16 *Y,
                                       uint32_t input_shape[4],
                                       uint32_t output_shape[4],
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t pad_h,
                                       uint32_t pad_w);

/*
 * As MAGIA_averagepool2d_fp16_spatz_l1, but only starts the Spatz task, with the parameter block
 * `p` (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
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
                                             mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_averagepool2d_fp16_spatz_l1_start. */
static inline void averagepool2d_fp16_spatz_params_set(averagepool2d_fp16_spatz_params_t *p,
                                                       const float16 *X,
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
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->c_start  = 0;
    p->c_len    = input_shape[0] * input_shape[1];
    p->h_in     = input_shape[2];
    p->w_in     = input_shape[3];
    p->h_out    = output_shape[2];
    p->w_out    = output_shape[3];
    p->kernel_h = kernel_h;
    p->kernel_w = kernel_w;
    p->stride_h = stride_h;
    p->stride_w = stride_w;
    p->pad_h    = pad_h;
    p->pad_w    = pad_w;
}

#endif /* AVERAGEPOOL2D_FP16_SPATZ_H_ */
