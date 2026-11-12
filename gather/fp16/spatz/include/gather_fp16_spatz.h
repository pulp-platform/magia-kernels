// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GATHER_FP16_SPATZ_H_
#define GATHER_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "gather_fp16_spatz_params.h"

/*
 * Gather slice `index` of axis `axis` (1..3) of the NCHW tensor `data` into `output`, in L2, with
 * the batches (the dimensions before `axis`) split over the tiles.
 */
void MAGIA_gather_fp16_spatz(const float16 *data,
                             uint32_t in_shape[4],
                             float16 *output,
                             uint32_t batch,
                             uint32_t gather_dim_size,
                             uint32_t axis_length,
                             uint32_t index,
                             uint32_t axis);

/* Same operation on this tile only, with data, output already in this tile's L1. */
void MAGIA_gather_fp16_spatz_l1(const float16 *data,
                                uint32_t in_shape[4],
                                float16 *output,
                                uint32_t batch,
                                uint32_t gather_dim_size,
                                uint32_t axis_length,
                                uint32_t index,
                                uint32_t axis);

/*
 * As MAGIA_gather_fp16_spatz_l1 (`axis` must be 1, 2 or 3), but only starts the Spatz task, with
 * the parameter block `p` (which must stay valid until it completes): mg_spatz_wait(mode, event)
 * waits for it.
 */
void MAGIA_gather_fp16_spatz_l1_start(gather_fp16_spatz_params_t *p,
                                      const float16 *data,
                                      uint32_t in_shape[4],
                                      float16 *output,
                                      uint32_t batch,
                                      uint32_t gather_dim_size,
                                      uint32_t axis_length,
                                      uint32_t index,
                                      uint32_t axis,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_gather_fp16_spatz_l1_start. */
static inline void gather_fp16_spatz_params_set(gather_fp16_spatz_params_t *p,
                                                const float16 *data,
                                                uint32_t in_shape[4],
                                                float16 *output,
                                                uint32_t batch,
                                                uint32_t gather_dim_size,
                                                uint32_t axis_length,
                                                uint32_t index,
                                                uint32_t axis)
{
    uint32_t batch_len = 1;

    (void)batch;
    for (uint32_t d = 0; d < axis; d++)
        batch_len *= in_shape[d];

    p->shard_input     = (uintptr_t)data;
    p->shard_output    = (uintptr_t)output;
    p->batch_start     = 0;
    p->batch_len       = batch_len;
    p->gather_dim_size = gather_dim_size;
    p->axis_length     = axis_length;
    p->index           = index;
}

#endif /* GATHER_FP16_SPATZ_H_ */
