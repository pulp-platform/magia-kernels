// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef LAYERNORM_FP16_SPATZ_H_
#define LAYERNORM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "layernorm_fp16_spatz_params.h"

void MAGIA_layernorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 epsilon,
                                uint32_t *input_shape,
                                uint32_t rank,
                                float16 *Y);
/* Same operation on this tile only, with X, scale, B and Y already in this tile's L1. */
void MAGIA_layernorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 epsilon,
                                   uint32_t *input_shape,
                                   uint32_t rank,
                                   float16 *Y);

/*
 * As MAGIA_layernorm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_layernorm_fp16_spatz_l1_start(layernorm_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         const float16 *scale,
                                         const float16 *B,
                                         const float16 epsilon,
                                         uint32_t *input_shape,
                                         uint32_t rank,
                                         float16 *Y,
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_layernorm_fp16_spatz_l1_start; epsilon is kept in `p`. */
static inline void layernorm_fp16_spatz_params_set(layernorm_fp16_spatz_params_t *p,
                                                   const float16 *X,
                                                   const float16 *scale,
                                                   const float16 *B,
                                                   const float16 epsilon,
                                                   uint32_t *input_shape,
                                                   uint32_t rank,
                                                   float16 *Y)
{
    uint32_t w_len = input_shape[rank - 1];
    uint32_t rows  = 1;

    for (uint32_t d = 0; d < rank - 1; d++)
        rows *= input_shape[d];

    p->eps_l1     = epsilon;
    p->shard_X    = (uintptr_t)X;
    p->shard_Y    = (uintptr_t)Y;
    p->gamma      = (uintptr_t)scale;
    p->beta       = (uintptr_t)B;
    p->eps        = (uintptr_t)&p->eps_l1;
    p->r_start    = 0;
    p->r_len      = rows;
    p->w_len      = w_len;
    p->input_len  = rows * w_len;
    p->output_len = rows * w_len;
}

#endif /* LAYERNORM_FP16_SPATZ_H_ */
