// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SILU_FP16_SPATZ_H_
#define SILU_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "silu_fp16_spatz_params.h"

/*
 * Y = X * sigmoid(X) = X / (1 + exp(-X)) (ONNX Swish with alpha = 1), elementwise over
 * total_elems elements, split over all tiles.
 */
void MAGIA_silu_fp16_spatz(const float16 *X, float16 *Y, uint32_t total_elems);

/* Same operation on this tile only, with X and Y already in this tile's L1. Y may alias X. */
void MAGIA_silu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t n);

/*
 * As MAGIA_silu_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_silu_fp16_spatz_l1_start(
    silu_fp16_spatz_params_t *p, const float16 *X, float16 *Y, uint32_t n, mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_silu_fp16_spatz_l1_start. */
static inline void
silu_fp16_spatz_params_set(silu_fp16_spatz_params_t *p, const float16 *X, float16 *Y, uint32_t n)
{
    p->shard_X     = (uintptr_t)X;
    p->shard_Y     = (uintptr_t)Y;
    p->total_elems = n;
    p->elem_start  = 0;
    p->elem_len    = n;
}

#endif /* SILU_FP16_SPATZ_H_ */
