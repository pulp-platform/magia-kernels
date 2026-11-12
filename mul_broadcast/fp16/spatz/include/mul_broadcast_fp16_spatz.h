// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MUL_BROADCAST_FP16_SPATZ_H_
#define MUL_BROADCAST_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "mul_broadcast_fp16_spatz_params.h"

/* C = A * B[0]: B points to a single scalar, broadcast over all total_elems elements of A. */
void MAGIA_mul_broadcast_fp16_spatz(const float16 *A,
                                    const float16 *B,
                                    float16 *C,
                                    uint32_t total_elems);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_mul_broadcast_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

/*
 * As MAGIA_mul_broadcast_fp16_spatz_l1, but only starts the Spatz task, with the parameter block
 * `p` (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_mul_broadcast_fp16_spatz_l1_start(mul_broadcast_fp16_spatz_params_t *p,
                                             const float16 *A,
                                             const float16 *B,
                                             float16 *C,
                                             uint32_t n,
                                             mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_mul_broadcast_fp16_spatz_l1_start. */
static inline void mul_broadcast_fp16_spatz_params_set(mul_broadcast_fp16_spatz_params_t *p,
                                                       const float16 *A,
                                                       const float16 *B,
                                                       float16 *C,
                                                       uint32_t n)
{
    p->shard_A     = (uintptr_t)A;
    p->shard_C     = (uintptr_t)C;
    p->scalar      = (uintptr_t)B;
    p->total_elems = n;
    p->elem_start  = 0;
    p->elem_len    = n;
}

#endif /* MUL_BROADCAST_FP16_SPATZ_H_ */
