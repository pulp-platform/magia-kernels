// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MAX_FP16_SPATZ_H_
#define MAX_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "max_fp16_spatz_params.h"

/* C[k] = max(A[k], B[k]) for `size` elements. */
void MAGIA_max_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1 (C may alias A or B). */
void MAGIA_max_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t size);

/*
 * As MAGIA_max_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_max_fp16_spatz_l1_start(max_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t size,
                                   mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_max_fp16_spatz_l1_start. */
static inline void max_fp16_spatz_params_set(max_fp16_spatz_params_t *p,
                                             const float16 *A,
                                             const float16 *B,
                                             float16 *C,
                                             uint32_t size)
{
    p->shard_A = (uintptr_t)A;
    p->shard_B = (uintptr_t)B;
    p->shard_C = (uintptr_t)C;
    p->start   = 0;
    p->len     = size;
    p->end     = size;
}

#endif /* MAX_FP16_SPATZ_H_ */
