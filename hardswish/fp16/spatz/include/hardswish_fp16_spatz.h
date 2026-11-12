// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HARDSWISH_FP16_SPATZ_H_
#define HARDSWISH_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "hardswish_fp16_spatz_params.h"

/* Y = hardswish(X), with X, Y in L2, split over the tiles. */
void MAGIA_hardswish_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);

/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_hardswish_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t size);

/*
 * As MAGIA_hardswish_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_hardswish_fp16_spatz_l1_start(hardswish_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         uint32_t size,
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_hardswish_fp16_spatz_l1_start. */
static inline void hardswish_fp16_spatz_params_set(hardswish_fp16_spatz_params_t *p,
                                                   const float16 *X,
                                                   float16 *Y,
                                                   uint32_t size)
{
    p->shard_X = (uintptr_t)X;
    p->shard_Y = (uintptr_t)Y;
    p->start   = 0;
    p->end     = size;
    p->len     = size;
}

#endif /* HARDSWISH_FP16_SPATZ_H_ */
