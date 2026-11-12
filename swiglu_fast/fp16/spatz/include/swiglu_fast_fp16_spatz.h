// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWIGLU_FAST_FP16_SPATZ_H_
#define SWIGLU_FAST_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "swiglu_fast_fp16_spatz_params.h"

/*
 * Y[i] = silu(G[i]) * U[i] for len FP16 elements, in FP16 with the Schraudolph exp (see the task).
 * The elements are split over all tiles.
 */
void MAGIA_swiglu_fast_fp16_spatz(float16 *Y, const float16 *G, const float16 *U, uint32_t len);
/* Same operation on this tile only, with all operands in its L1. Y may alias G or U. */
void MAGIA_swiglu_fast_fp16_spatz_l1(float16 *Y, const float16 *G, const float16 *U, uint32_t len);

/*
 * As MAGIA_swiglu_fast_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_swiglu_fast_fp16_spatz_l1_start(swiglu_fast_fp16_spatz_params_t *p,
                                           float16 *Y,
                                           const float16 *G,
                                           const float16 *U,
                                           uint32_t len,
                                           mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_swiglu_fast_fp16_spatz_l1_start. */
static inline void swiglu_fast_fp16_spatz_params_set(swiglu_fast_fp16_spatz_params_t *p,
                                                     float16 *Y,
                                                     const float16 *G,
                                                     const float16 *U,
                                                     uint32_t len)
{
    p->shard_G = (uintptr_t)G;
    p->shard_U = (uintptr_t)U;
    p->shard_Y = (uintptr_t)Y;
    p->len     = len;
}

#endif /* SWIGLU_FAST_FP16_SPATZ_H_ */
