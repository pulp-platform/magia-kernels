// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef BIAS_ROPE_T_FP16_SPATZ_H_
#define BIAS_ROPE_T_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "bias_rope_t_fp16_spatz_params.h"

/*
 * X[r, t] += B[r], then the RoPE of each column (token) with the channels of each head in groups
 * of group: in group g of a head, rows (g * group + j, g * group + group / 2 + j) are the pair of
 * angle g * group / 2 + j of cos/sin [head_dim / 2 x cols]. The block starts at row row0 (a
 * multiple of group) of its head. X is updated in place; the columns (tokens) are split over all
 * tiles, in pairs.
 */
void MAGIA_bias_rope_t_fp16_spatz(float16 *X,
                                  const float16 *B,
                                  const float16 *C,
                                  const float16 *S,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t row0,
                                  uint32_t head_dim,
                                  uint32_t group);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_bias_rope_t_fp16_spatz_l1(float16 *X,
                                     const float16 *B,
                                     const float16 *C,
                                     const float16 *S,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row0,
                                     uint32_t head_dim,
                                     uint32_t group);

/*
 * As MAGIA_bias_rope_t_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_bias_rope_t_fp16_spatz_l1_start(bias_rope_t_fp16_spatz_params_t *p,
                                           float16 *X,
                                           const float16 *B,
                                           const float16 *C,
                                           const float16 *S,
                                           uint32_t rows,
                                           uint32_t cols,
                                           uint32_t row0,
                                           uint32_t head_dim,
                                           uint32_t group,
                                           mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_bias_rope_t_fp16_spatz_l1_start. */
static inline void bias_rope_t_fp16_spatz_params_set(bias_rope_t_fp16_spatz_params_t *p,
                                                     float16 *X,
                                                     const float16 *B,
                                                     const float16 *C,
                                                     const float16 *S,
                                                     uint32_t rows,
                                                     uint32_t cols,
                                                     uint32_t row0,
                                                     uint32_t head_dim,
                                                     uint32_t group)
{
    p->shard_X  = (uintptr_t)X;
    p->bias     = (uintptr_t)B;
    p->cos      = (uintptr_t)C;
    p->sin      = (uintptr_t)S;
    p->dims     = rows | (cols << 16);
    p->row0     = row0;
    p->head_dim = head_dim;
    p->group    = group;
}

#endif /* BIAS_ROPE_T_FP16_SPATZ_H_ */
