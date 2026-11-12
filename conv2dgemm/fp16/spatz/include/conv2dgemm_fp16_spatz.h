// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONV2DGEMM_FP16_SPATZ_H_
#define CONV2DGEMM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "conv2dgemm_fp16_spatz_params.h"

/*
 * Conv2D as im2col + GEMM of the NCHW tensor X into Y, in L2, with the output rows or the output
 * channels split over the tiles.
 */
void MAGIA_conv2dgemm_fp16_spatz(const float16 *X,
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
 * Same operation on this tile only, with X, W, B and Y already in this tile's L1. scratch is an L1
 * buffer of N*C_in*K_h*K_w*H_out*W_out elements for im2col (none for a 1x1, unit-stride, unpadded
 * conv) plus C_out*H_out*W_out for the bias (if has_bias).
 */
void MAGIA_conv2dgemm_fp16_spatz_l1(const float16 *X,
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
                                    float16 *scratch);

/*
 * As MAGIA_conv2dgemm_fp16_spatz_l1 (it writes the bias broadcast into scratch first), but only
 * starts the Spatz task, with the parameter block `p` (which must stay valid until it completes):
 * mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_conv2dgemm_fp16_spatz_l1_start(conv2dgemm_fp16_spatz_params_t *p,
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
                                          float16 *scratch,
                                          mg_event_t *event);

/*
 * Fill the parameter block `p` for MAGIA_conv2dgemm_fp16_spatz_l1_start; alpha and beta are kept
 * in `p`.
 */
static inline void conv2dgemm_fp16_spatz_params_set(conv2dgemm_fp16_spatz_params_t *p,
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
                                                    float16 *scratch)
{
    uint32_t K      = input_shape[1] * kernel_h * kernel_w;
    uint32_t hw_out = output_shape[2] * output_shape[3];
    int is_1x1 = (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 && pad_h == 0 &&
                  pad_w == 0);

    (void)B;
    p->alpha_l1  = (float16)1.0f;
    p->beta_l1   = has_bias ? (float16)1.0f : (float16)0.0f;
    p->shard_X   = (uintptr_t)X;
    p->shard_A   = (uintptr_t)W;
    p->shard_B   = (uintptr_t)(is_1x1 ? (float16 *)X : scratch);
    p->shard_C   = (uintptr_t)(is_1x1 ? scratch : scratch + input_shape[0] * K * hw_out);
    p->shard_Y   = (uintptr_t)Y;
    p->alpha     = (uintptr_t)&p->alpha_l1;
    p->beta      = (uintptr_t)&p->beta_l1;
    p->M         = output_shape[1];
    p->N         = hw_out;
    p->K         = K;
    p->has_bias  = (uint32_t)has_bias;
    p->n_batches = input_shape[0];
    p->c_out     = output_shape[1];
    p->group     = group;
    p->oc_start  = 0;
    p->oh_start  = 0;
    p->oh_len    = output_shape[2];
    p->c_in      = input_shape[1];
    p->h_in      = input_shape[2];
    p->w_in      = input_shape[3];
    p->h_out     = output_shape[2];
    p->w_out     = output_shape[3];
    p->kernel_h  = kernel_h;
    p->kernel_w  = kernel_w;
    p->stride_h  = stride_h;
    p->stride_w  = stride_w;
    p->pad_h     = pad_h;
    p->pad_w     = pad_w;
}

#endif /* CONV2DGEMM_FP16_SPATZ_H_ */
