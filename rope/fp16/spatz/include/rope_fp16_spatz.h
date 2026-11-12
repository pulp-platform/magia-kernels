// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROPE_FP16_SPATZ_H_
#define ROPE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "rope_fp16_spatz_params.h"

/*
 * Rotary position embedding, rotate-half form (Llama, ONNX RotaryEmbedding with interleaved = 0):
 * X [rows x n_heads * head_dim], one token per row; cos, sin [rows x head_dim / 2], already
 * gathered at each row's position. For every head x = [x1 | x2]:
 *   Y = [x1 * cos - x2 * sin | x2 * cos + x1 * sin]
 * The rows are split over all tiles.
 */
void MAGIA_rope_fp16_spatz(const float16 *X,
                           const float16 *cos,
                           const float16 *sin,
                           uint32_t rows,
                           uint32_t n_heads,
                           uint32_t head_dim,
                           float16 *Y);
/*
 * Same operation on this tile only, with X, cos, sin and Y already in this tile's L1. Y may
 * alias X.
 */
void MAGIA_rope_fp16_spatz_l1(const float16 *X,
                              const float16 *cos,
                              const float16 *sin,
                              uint32_t rows,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              float16 *Y);

/*
 * As MAGIA_rope_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_rope_fp16_spatz_l1_start(rope_fp16_spatz_params_t *p,
                                    const float16 *X,
                                    const float16 *cos,
                                    const float16 *sin,
                                    uint32_t rows,
                                    uint32_t n_heads,
                                    uint32_t head_dim,
                                    float16 *Y,
                                    mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_rope_fp16_spatz_l1_start. */
static inline void rope_fp16_spatz_params_set(rope_fp16_spatz_params_t *p,
                                              const float16 *X,
                                              const float16 *cos,
                                              const float16 *sin,
                                              uint32_t rows,
                                              uint32_t n_heads,
                                              uint32_t head_dim,
                                              float16 *Y)
{
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->cos      = (uintptr_t)cos;
    p->sin      = (uintptr_t)sin;
    p->r_start  = 0;
    p->r_len    = rows;
    p->n_heads  = n_heads;
    p->head_dim = head_dim;
}

#endif /* ROPE_FP16_SPATZ_H_ */
