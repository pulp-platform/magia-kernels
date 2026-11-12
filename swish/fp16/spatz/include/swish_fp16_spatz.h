// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWISH_FP16_SPATZ_H_
#define SWISH_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "swish_fp16_spatz_params.h"

/* Y = swish(X, alpha), with X, Y in L2, split over the tiles. */
void MAGIA_swish_fp16_spatz(const float16 *X, float16 *Y, const float16 alpha, uint32_t size);

/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_swish_fp16_spatz_l1(const float16 *X, float16 *Y, const float16 alpha, uint32_t size);

/*
 * As MAGIA_swish_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_swish_fp16_spatz_l1_start(swish_fp16_spatz_params_t *p,
                                     const float16 *X,
                                     float16 *Y,
                                     const float16 alpha,
                                     uint32_t size,
                                     mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_swish_fp16_spatz_l1_start; alpha is kept in `p`. */
static inline void swish_fp16_spatz_params_set(
    swish_fp16_spatz_params_t *p, const float16 *X, float16 *Y, const float16 alpha, uint32_t size)
{
    p->alpha_l1 = alpha;
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->alpha    = (uintptr_t)&p->alpha_l1;
    p->start    = 0;
    p->end      = size;
    p->len      = size;
}

#endif /* SWISH_FP16_SPATZ_H_ */
