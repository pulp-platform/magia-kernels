// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWIGLU_FP16_SPATZ_H_
#define SWIGLU_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "swiglu_fp16_spatz_params.h"

/*
 * Y = SiLU(G) * U = G / (1 + exp(-G)) * U (the gating of a SwiGLU FFN, G = X @ W_gate,
 * U = X @ W_up), elementwise over total_elems elements, split over all tiles.
 */
void MAGIA_swiglu_fp16_spatz(const float16 *G, const float16 *U, float16 *Y, uint32_t total_elems);

/*
 * Same operation on this tile only, with G, U and Y already in this tile's L1. Y may alias G or U.
 */
void MAGIA_swiglu_fp16_spatz_l1(const float16 *G, const float16 *U, float16 *Y, uint32_t n);

/*
 * As MAGIA_swiglu_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_swiglu_fp16_spatz_l1_start(swiglu_fp16_spatz_params_t *p,
                                      const float16 *G,
                                      const float16 *U,
                                      float16 *Y,
                                      uint32_t n,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_swiglu_fp16_spatz_l1_start. */
static inline void swiglu_fp16_spatz_params_set(
    swiglu_fp16_spatz_params_t *p, const float16 *G, const float16 *U, float16 *Y, uint32_t n)
{
    p->shard_G     = (uintptr_t)G;
    p->shard_U     = (uintptr_t)U;
    p->shard_Y     = (uintptr_t)Y;
    p->total_elems = n;
    p->elem_start  = 0;
    p->elem_len    = n;
}

#endif /* SWIGLU_FP16_SPATZ_H_ */
