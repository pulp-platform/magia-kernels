// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CLIP_FP16_SPATZ_H_
#define CLIP_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "clip_fp16_spatz_params.h"

/* output = clip(input, min, max), with input, output in L2, split over the tiles. */
void MAGIA_clip_fp16_spatz(
    const float16 *input, float16 *output, float16 min, float16 max, uint32_t size);

/*
 * Same operation on this tile only, with input, output already in this tile's L1. output may alias
 * input.
 */
void MAGIA_clip_fp16_spatz_l1(
    const float16 *input, float16 *output, float16 min, float16 max, uint32_t size);

/*
 * As MAGIA_clip_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_clip_fp16_spatz_l1_start(clip_fp16_spatz_params_t *p,
                                    const float16 *input,
                                    float16 *output,
                                    float16 min,
                                    float16 max,
                                    uint32_t size,
                                    mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_clip_fp16_spatz_l1_start; min and max are kept in `p`. */
static inline void clip_fp16_spatz_params_set(clip_fp16_spatz_params_t *p,
                                              const float16 *input,
                                              float16 *output,
                                              float16 min,
                                              float16 max,
                                              uint32_t size)
{
    p->min_l1       = min;
    p->max_l1       = max;
    p->shard_input  = (uintptr_t)input;
    p->shard_output = (uintptr_t)output;
    p->min          = (uintptr_t)&p->min_l1;
    p->max          = (uintptr_t)&p->max_l1;
    p->start        = 0;
    p->end          = size;
    p->len          = size;
}

#endif /* CLIP_FP16_SPATZ_H_ */
