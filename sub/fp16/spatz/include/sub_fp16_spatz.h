// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SUB_FP16_SPATZ_H_
#define SUB_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "sub_fp16_spatz_params.h"

void MAGIA_sub_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_sub_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

/*
 * As MAGIA_sub_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_sub_fp16_spatz_l1_start(sub_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t size,
                                   mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_sub_fp16_spatz_l1_start. */
static inline void sub_fp16_spatz_params_set(sub_fp16_spatz_params_t *p,
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

#endif /* SUB_FP16_SPATZ_H_ */
