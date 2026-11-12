// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SOFTMAX_FP16_SPATZ_H_
#define SOFTMAX_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "softmax_fp16_spatz_params.h"

/*
 * Softmax over one axis, with input and output viewed as [outer_dim, reduce_dim, inner_dim],
 * split over the tiles along outer_dim.
 */
void MAGIA_softmax_fp16_spatz(const float16 *input,
                              float16 *output,
                              uint32_t outer_dim,
                              uint32_t reduce_dim,
                              uint32_t inner_dim);

/* Same operation on this tile only, with input, output already in this tile's L1. */
void MAGIA_softmax_fp16_spatz_l1(const float16 *input,
                                 float16 *output,
                                 uint32_t outer_dim,
                                 uint32_t reduce_dim,
                                 uint32_t inner_dim);

/*
 * As MAGIA_softmax_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_softmax_fp16_spatz_l1_start(softmax_fp16_spatz_params_t *p,
                                       const float16 *input,
                                       float16 *output,
                                       uint32_t outer_dim,
                                       uint32_t reduce_dim,
                                       uint32_t inner_dim,
                                       mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_softmax_fp16_spatz_l1_start. */
static inline void softmax_fp16_spatz_params_set(softmax_fp16_spatz_params_t *p,
                                                 const float16 *input,
                                                 float16 *output,
                                                 uint32_t outer_dim,
                                                 uint32_t reduce_dim,
                                                 uint32_t inner_dim)
{
    p->shard_input  = (uintptr_t)input;
    p->shard_output = (uintptr_t)output;
    p->reduce_dim   = reduce_dim;
    p->inner_dim    = inner_dim;
    p->outer_start  = 0;
    p->outer_len    = outer_dim;
}

#endif /* SOFTMAX_FP16_SPATZ_H_ */
