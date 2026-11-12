// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MEMSET_FP16_SPATZ_H_
#define MEMSET_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "memset_fp16_spatz_params.h"

/* C[k] = 0 for `size` elements, with C already in this tile's L1. */
void MAGIA_memset_zero_fp16_spatz_l1(float16 *C, uint32_t size);

/*
 * As MAGIA_memset_zero_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_memset_zero_fp16_spatz_l1_start(memset_fp16_spatz_params_t *p,
                                           float16 *C,
                                           uint32_t size,
                                           mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_memset_zero_fp16_spatz_l1_start. */
static inline void memset_fp16_spatz_params_set(memset_fp16_spatz_params_t *p,
                                                float16 *C,
                                                uint32_t size)
{
    p->shard_C = (uintptr_t)C;
    p->len     = size;
}

#endif /* MEMSET_FP16_SPATZ_H_ */
