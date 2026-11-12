// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GROUPNORM_FP16_SPATZ_H_
#define GROUPNORM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "groupnorm_fp16_spatz_params.h"

/* GroupNorm of the NCHW tensor X into Y, in L2, with the groups split over the tiles. */
void MAGIA_groupnorm_fp16_spatz(const float16 *X,
                                const float16 *Y,
                                const float16 *scale,
                                const float16 *B,
                                uint32_t input_shape[4],
                                uint32_t num_groups,
                                float16 epsilon);

/* Same operation on this tile only, with X, Y, scale and B already in this tile's L1. */
void MAGIA_groupnorm_fp16_spatz_l1(const float16 *X,
                                   float16 *Y,
                                   const float16 *scale,
                                   const float16 *B,
                                   uint32_t input_shape[4],
                                   uint32_t num_groups,
                                   float16 epsilon);

/*
 * As MAGIA_groupnorm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_groupnorm_fp16_spatz_l1_start(groupnorm_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         const float16 *scale,
                                         const float16 *B,
                                         uint32_t input_shape[4],
                                         uint32_t num_groups,
                                         float16 epsilon,
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_groupnorm_fp16_spatz_l1_start; epsilon is kept in `p`. */
static inline void groupnorm_fp16_spatz_params_set(groupnorm_fp16_spatz_params_t *p,
                                                   const float16 *X,
                                                   float16 *Y,
                                                   const float16 *scale,
                                                   const float16 *B,
                                                   uint32_t input_shape[4],
                                                   uint32_t num_groups,
                                                   float16 epsilon)
{
    uint32_t len = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];

    p->eps_l1     = epsilon;
    p->shard_X    = (uintptr_t)X;
    p->shard_Y    = (uintptr_t)Y;
    p->gamma      = (uintptr_t)scale;
    p->beta       = (uintptr_t)B;
    p->eps        = (uintptr_t)&p->eps_l1;
    p->g_start    = 0;
    p->g_len      = input_shape[0] * num_groups;
    p->c_per_g    = input_shape[1] / num_groups;
    p->hw_len     = input_shape[2] * input_shape[3];
    p->num_groups = num_groups;
    p->c_out      = input_shape[1];
    p->input_len  = len;
    p->output_len = len;
}

#endif /* GROUPNORM_FP16_SPATZ_H_ */
