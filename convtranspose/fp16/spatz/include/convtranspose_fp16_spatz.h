// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONVTRANSPOSE_FP16_SPATZ_H_
#define CONVTRANSPOSE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "convtranspose_fp16_spatz_params.h"

/*
 * ConvTranspose of the NCHW tensor X into Y, in L2, with the output planes split over the tiles.
 */
void MAGIA_convtranspose_fp16_spatz(const float16 *X,
                                    const float16 *W,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w,
                                    uint32_t num_groups);

/* Same operation on this tile only, with X, W and Y already in this tile's L1. */
void MAGIA_convtranspose_fp16_spatz_l1(const float16 *X,
                                       const float16 *W,
                                       float16 *Y,
                                       uint32_t input_shape[4],
                                       uint32_t output_shape[4],
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t pad_h,
                                       uint32_t pad_w,
                                       uint32_t num_groups);

/*
 * As MAGIA_convtranspose_fp16_spatz_l1 (it zeroes Y first), but only starts the Spatz task, with
 * the parameter block `p` (which must stay valid until it completes): mg_spatz_wait(mode, event)
 * waits for it.
 */
void MAGIA_convtranspose_fp16_spatz_l1_start(convtranspose_fp16_spatz_params_t *p,
                                             const float16 *X,
                                             const float16 *W,
                                             float16 *Y,
                                             uint32_t input_shape[4],
                                             uint32_t output_shape[4],
                                             uint32_t kernel_h,
                                             uint32_t kernel_w,
                                             uint32_t stride_h,
                                             uint32_t stride_w,
                                             uint32_t pad_h,
                                             uint32_t pad_w,
                                             uint32_t num_groups,
                                             mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_convtranspose_fp16_spatz_l1_start. */
static inline void convtranspose_fp16_spatz_params_set(convtranspose_fp16_spatz_params_t *p,
                                                       const float16 *X,
                                                       const float16 *W,
                                                       float16 *Y,
                                                       uint32_t input_shape[4],
                                                       uint32_t output_shape[4],
                                                       uint32_t kernel_h,
                                                       uint32_t kernel_w,
                                                       uint32_t stride_h,
                                                       uint32_t stride_w,
                                                       uint32_t pad_h,
                                                       uint32_t pad_w,
                                                       uint32_t num_groups)
{
    p->shard_X    = (uintptr_t)X;
    p->shard_W    = (uintptr_t)W;
    p->shard_Y    = (uintptr_t)Y;
    p->n_batches  = input_shape[0];
    p->c_in       = input_shape[1];
    p->c_out      = output_shape[1];
    p->iter_start = 0;
    p->iter_len   = input_shape[0] * output_shape[1];
    p->c_in_g     = input_shape[1] / num_groups;
    p->c_out_g    = output_shape[1] / num_groups;
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

#endif /* CONVTRANSPOSE_FP16_SPATZ_H_ */
