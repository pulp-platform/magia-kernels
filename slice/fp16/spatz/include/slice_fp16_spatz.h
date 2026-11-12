// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLICE_FP16_SPATZ_H_
#define SLICE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "slice_fp16_spatz_params.h"

void MAGIA_slice_fp16_spatz(const float16 *data,
                            float16 *sliced,
                            uint32_t outer_dim,
                            uint32_t slice_dim,
                            uint32_t inner_dim,
                            uint32_t start_idx,
                            uint32_t out_slice_dim);
/* Same operation on this tile only, with data and sliced already in this tile's L1. */
void MAGIA_slice_fp16_spatz_l1(const float16 *data,
                               float16 *sliced,
                               uint32_t outer_dim,
                               uint32_t slice_dim,
                               uint32_t inner_dim,
                               uint32_t start_idx,
                               uint32_t out_slice_dim);

/*
 * As MAGIA_slice_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_slice_fp16_spatz_l1_start(slice_fp16_spatz_params_t *p,
                                     const float16 *data,
                                     float16 *sliced,
                                     uint32_t outer_dim,
                                     uint32_t slice_dim,
                                     uint32_t inner_dim,
                                     uint32_t start_idx,
                                     uint32_t out_slice_dim,
                                     mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_slice_fp16_spatz_l1_start. */
static inline void slice_fp16_spatz_params_set(slice_fp16_spatz_params_t *p,
                                               const float16 *data,
                                               float16 *sliced,
                                               uint32_t outer_dim,
                                               uint32_t slice_dim,
                                               uint32_t inner_dim,
                                               uint32_t start_idx,
                                               uint32_t out_slice_dim)
{
    p->shard_X       = (uintptr_t)data;
    p->shard_Y       = (uintptr_t)sliced;
    p->slice_dim     = slice_dim;
    p->out_slice_dim = out_slice_dim;
    p->inner_dim     = inner_dim;
    p->start_idx     = start_idx;
    p->start_outer   = 0;
    p->len_outer     = outer_dim;
}

#endif /* SLICE_FP16_SPATZ_H_ */
