// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TANH_FP16_SPATZ_H_
#define TANH_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "tanh_fp16_spatz_params.h"

/* output = tanh(input), with input, output in L2, split over the tiles. */
void MAGIA_tanh_fp16_spatz(const float16 *input, float16 *output, uint32_t size);

/*
 * Same operation on this tile only, with input, output already in this tile's L1. output may alias
 * input.
 */
void MAGIA_tanh_fp16_spatz_l1(const float16 *input, float16 *output, uint32_t size);

/*
 * As MAGIA_tanh_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_tanh_fp16_spatz_l1_start(tanh_fp16_spatz_params_t *p,
                                    const float16 *input,
                                    float16 *output,
                                    uint32_t size,
                                    mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_tanh_fp16_spatz_l1_start. */
static inline void tanh_fp16_spatz_params_set(tanh_fp16_spatz_params_t *p,
                                              const float16 *input,
                                              float16 *output,
                                              uint32_t size)
{
    p->shard_input  = (uintptr_t)input;
    p->shard_output = (uintptr_t)output;
    p->start        = 0;
    p->end          = size;
    p->len          = size;
}

#endif /* TANH_FP16_SPATZ_H_ */
