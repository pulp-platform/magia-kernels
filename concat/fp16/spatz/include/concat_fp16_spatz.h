// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONCAT_FP16_SPATZ_H_
#define CONCAT_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "concat_fp16_spatz_params.h"

/*
 * Concatenate, for each of `iterations` iterations, lens[i] elements of each inputs[i] into
 * concat_result, in L2, with the iterations split over the tiles.
 */
void MAGIA_concat_fp16_spatz(const float16 **inputs,
                             const uint32_t *lens,
                             uint32_t num_inputs,
                             float16 *concat_result,
                             uint32_t iterations);

/* Same operation on this tile only, with the inputs and concat_result already in this tile's L1. */
void MAGIA_concat_fp16_spatz_l1(const float16 **inputs,
                                const uint32_t *lens,
                                uint32_t num_inputs,
                                float16 *concat_result,
                                uint32_t iterations);

/*
 * As MAGIA_concat_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it. The task
 * reads the arrays `inputs` and `lens` too, so they must also be in this tile's L1 and stay valid.
 */
void MAGIA_concat_fp16_spatz_l1_start(concat_fp16_spatz_params_t *p,
                                      const float16 **inputs,
                                      const uint32_t *lens,
                                      uint32_t num_inputs,
                                      float16 *concat_result,
                                      uint32_t iterations,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_concat_fp16_spatz_l1_start. */
static inline void concat_fp16_spatz_params_set(concat_fp16_spatz_params_t *p,
                                                const float16 **inputs,
                                                const uint32_t *lens,
                                                uint32_t num_inputs,
                                                float16 *concat_result,
                                                uint32_t iterations)
{
    p->shard_input  = (uintptr_t)inputs;
    p->len_input    = (uintptr_t)lens;
    p->shard_output = (uintptr_t)concat_result;
    p->iter_start   = 0;
    p->iter_len     = iterations;
    p->num_inputs   = num_inputs;
}

#endif /* CONCAT_FP16_SPATZ_H_ */
