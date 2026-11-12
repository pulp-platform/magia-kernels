// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXP_FP16_SPATZ_H_
#define EXP_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "exp_fp16_spatz_params.h"

/* output = exp(input), with input and output in L2, split over the tiles. */
void MAGIA_exp_fp16_spatz(const float16 *input, float16 *output, uint32_t size);

/* Same operation on this tile only, with input and output already in this tile's L1. */
void MAGIA_exp_fp16_spatz_l1(const float16 *input, float16 *output, uint32_t n);

/*
 * As MAGIA_exp_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_exp_fp16_spatz_l1_start(exp_fp16_spatz_params_t *p,
                                   const float16 *input,
                                   float16 *output,
                                   uint32_t n,
                                   mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_exp_fp16_spatz_l1_start. */
static inline void exp_fp16_spatz_params_set(exp_fp16_spatz_params_t *p,
                                             const float16 *input,
                                             float16 *output,
                                             uint32_t n)
{
    p->shard_input  = (uintptr_t)input;
    p->shard_output = (uintptr_t)output;
    p->start        = 0;
    p->end          = n;
    p->len          = n;
}

#endif /* EXP_FP16_SPATZ_H_ */
