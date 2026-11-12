// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef LEAKYRELU_FP16_SPATZ_H_
#define LEAKYRELU_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "leakyrelu_fp16_spatz_params.h"

/* Y = leakyrelu(X, alpha), with X, Y in L2, split over the tiles. */
void MAGIA_leakyrelu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha);

/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_leakyrelu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t size, float16 alpha);

/*
 * As MAGIA_leakyrelu_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_leakyrelu_fp16_spatz_l1_start(leakyrelu_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         uint32_t size,
                                         float16 alpha,
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_leakyrelu_fp16_spatz_l1_start; alpha is kept in `p`. */
static inline void leakyrelu_fp16_spatz_params_set(
    leakyrelu_fp16_spatz_params_t *p, const float16 *X, float16 *Y, uint32_t size, float16 alpha)
{
    p->alpha_l1 = alpha;
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->alpha    = (uintptr_t)&p->alpha_l1;
    p->start    = 0;
    p->end      = size;
    p->len      = size;
}

#endif /* LEAKYRELU_FP16_SPATZ_H_ */
