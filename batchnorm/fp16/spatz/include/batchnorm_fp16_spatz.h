// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef BATCHNORM_FP16_SPATZ_H_
#define BATCHNORM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "batchnorm_fp16_spatz_params.h"

/* BatchNorm of the NCHW tensor X into Y, in L2, with the channels split over the tiles. */
void MAGIA_batchnorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 *input_mean,
                                const float16 *input_var,
                                const float16 epsilon,
                                float16 *Y,
                                uint32_t shape[4]);

/*
 * Same operation on this tile only, with X, scale, B, input_mean, input_var and Y already in this
 * tile's L1.
 */
void MAGIA_batchnorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 *input_mean,
                                   const float16 *input_var,
                                   const float16 epsilon,
                                   float16 *Y,
                                   uint32_t shape[4]);

/*
 * As MAGIA_batchnorm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_batchnorm_fp16_spatz_l1_start(batchnorm_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         const float16 *scale,
                                         const float16 *B,
                                         const float16 *input_mean,
                                         const float16 *input_var,
                                         const float16 epsilon,
                                         float16 *Y,
                                         uint32_t shape[4],
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_batchnorm_fp16_spatz_l1_start; epsilon is kept in `p`. */
static inline void batchnorm_fp16_spatz_params_set(batchnorm_fp16_spatz_params_t *p,
                                                   const float16 *X,
                                                   const float16 *scale,
                                                   const float16 *B,
                                                   const float16 *input_mean,
                                                   const float16 *input_var,
                                                   const float16 epsilon,
                                                   float16 *Y,
                                                   uint32_t shape[4])
{
    p->eps_l1   = epsilon;
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->gamma    = (uintptr_t)scale;
    p->beta     = (uintptr_t)B;
    p->mean     = (uintptr_t)input_mean;
    p->var      = (uintptr_t)input_var;
    p->eps      = (uintptr_t)&p->eps_l1;
    p->channels = shape[1];
    p->c_start  = 0;
    p->c_len    = shape[0] * shape[1];
    p->hw_len   = shape[2] * shape[3];
}

#endif /* BATCHNORM_FP16_SPATZ_H_ */
