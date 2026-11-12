// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MUL_FP16_SPATZ_H_
#define MUL_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "mul_fp16_spatz_params.h"

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_mul_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

/*
 * As MAGIA_mul_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_mul_fp16_spatz_l1_start(mul_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t n,
                                   mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_mul_fp16_spatz_l1_start. */
static inline void mul_fp16_spatz_params_set(mul_fp16_spatz_params_t *p,
                                             const float16 *A,
                                             const float16 *B,
                                             float16 *C,
                                             uint32_t n)
{
    p->shard_A     = (uintptr_t)A;
    p->shard_B     = (uintptr_t)B;
    p->shard_C     = (uintptr_t)C;
    p->elem_start  = 0;
    p->elem_len    = n;
    p->total_elems = n;
}

#endif /* MUL_FP16_SPATZ_H_ */
