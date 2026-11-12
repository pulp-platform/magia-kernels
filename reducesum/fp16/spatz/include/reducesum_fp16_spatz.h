// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef REDUCESUM_FP16_SPATZ_H_
#define REDUCESUM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "reducesum_fp16_spatz_params.h"

/*
 * Y = sum of X over one axis, with X viewed as [outer_dim, reduce_dim, inner_dim] and
 * Y as [outer_dim, inner_dim].
 */
void MAGIA_reducesum_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim);

/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_reducesum_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t outer_len, uint32_t reduce_dim, uint32_t inner_dim);

/*
 * As MAGIA_reducesum_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_reducesum_fp16_spatz_l1_start(reducesum_fp16_spatz_params_t *p,
                                         const float16 *X,
                                         float16 *Y,
                                         uint32_t outer_len,
                                         uint32_t reduce_dim,
                                         uint32_t inner_dim,
                                         mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_reducesum_fp16_spatz_l1_start. */
static inline void reducesum_fp16_spatz_params_set(reducesum_fp16_spatz_params_t *p,
                                                   const float16 *X,
                                                   float16 *Y,
                                                   uint32_t outer_len,
                                                   uint32_t reduce_dim,
                                                   uint32_t inner_dim)
{
    p->shard_X     = (uintptr_t)X;
    p->shard_Y     = (uintptr_t)Y;
    p->reduce_dim  = reduce_dim;
    p->inner_dim   = inner_dim;
    p->outer_start = 0;
    p->outer_len   = outer_len;
}

#endif /* REDUCESUM_FP16_SPATZ_H_ */
