// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SELU_FP16_SPATZ_H_
#define SELU_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "selu_fp16_spatz_params.h"

/* Y = selu(X, alpha, gamma), with X, Y in L2, split over the tiles. */
void MAGIA_selu_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma);

/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_selu_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma);

/*
 * As MAGIA_selu_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_selu_fp16_spatz_l1_start(selu_fp16_spatz_params_t *p,
                                    const float16 *X,
                                    float16 *Y,
                                    uint32_t size,
                                    float16 alpha,
                                    float16 gamma,
                                    mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_selu_fp16_spatz_l1_start; alpha, gamma are kept in `p`. */
static inline void selu_fp16_spatz_params_set(selu_fp16_spatz_params_t *p,
                                              const float16 *X,
                                              float16 *Y,
                                              uint32_t size,
                                              float16 alpha,
                                              float16 gamma)
{
    p->alpha_l1 = alpha;
    p->gamma_l1 = gamma;
    p->shard_X  = (uintptr_t)X;
    p->shard_Y  = (uintptr_t)Y;
    p->alpha    = (uintptr_t)&p->alpha_l1;
    p->gamma    = (uintptr_t)&p->gamma_l1;
    p->start    = 0;
    p->end      = size;
    p->len      = size;
}

#endif /* SELU_FP16_SPATZ_H_ */
