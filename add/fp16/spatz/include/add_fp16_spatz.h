// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ADD_FP16_SPATZ_H_
#define ADD_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "add_fp16_spatz_params.h"

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_add_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

/*
 * As MAGIA_add_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_add_fp16_spatz_l1_start(add_fp16_spatz_params_t *p,
                                   const float16 *A,
                                   const float16 *B,
                                   float16 *C,
                                   uint32_t n,
                                   mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_add_fp16_spatz_l1_start. */
static inline void add_fp16_spatz_params_set(add_fp16_spatz_params_t *p,
                                             const float16 *A,
                                             const float16 *B,
                                             float16 *C,
                                             uint32_t n)
{
    p->shard_A = (uintptr_t)A;
    p->shard_B = (uintptr_t)B;
    p->shard_C = (uintptr_t)C;
    p->start   = 0;
    p->len     = n;
    p->end     = n;
}

#endif /* ADD_FP16_SPATZ_H_ */
