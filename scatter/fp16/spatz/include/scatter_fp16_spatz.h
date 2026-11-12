// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SCATTER_FP16_SPATZ_H_
#define SCATTER_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "scatter_fp16_spatz_params.h"

void MAGIA_scatter_fp16_spatz(const float16 *data,
                              const int64_t *indices,
                              const float16 *updates,
                              float16 *output,
                              uint32_t outer_size,
                              uint32_t inner_size,
                              uint32_t axis,
                              uint32_t data_axis_dim,
                              uint32_t indices_axis_dim);
/*
 * Same operation on this tile only, with data, indices, updates and output already in this
 * tile's L1. Here indices are int32. output may alias data.
 */
void MAGIA_scatter_fp16_spatz_l1(const float16 *data,
                                 const int32_t *indices,
                                 const float16 *updates,
                                 float16 *output,
                                 uint32_t outer_size,
                                 uint32_t inner_size,
                                 uint32_t axis,
                                 uint32_t data_axis_dim,
                                 uint32_t indices_axis_dim);

/*
 * As MAGIA_scatter_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_scatter_fp16_spatz_l1_start(scatter_fp16_spatz_params_t *p,
                                       const float16 *data,
                                       const int32_t *indices,
                                       const float16 *updates,
                                       float16 *output,
                                       uint32_t outer_size,
                                       uint32_t inner_size,
                                       uint32_t axis,
                                       uint32_t data_axis_dim,
                                       uint32_t indices_axis_dim,
                                       mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_scatter_fp16_spatz_l1_start. */
static inline void scatter_fp16_spatz_params_set(scatter_fp16_spatz_params_t *p,
                                                 const float16 *data,
                                                 const int32_t *indices,
                                                 const float16 *updates,
                                                 float16 *output,
                                                 uint32_t outer_size,
                                                 uint32_t inner_size,
                                                 uint32_t axis,
                                                 uint32_t data_axis_dim,
                                                 uint32_t indices_axis_dim)
{
    p->shard_data             = (uintptr_t)data;
    p->shard_indices          = (uintptr_t)indices;
    p->shard_updates          = (uintptr_t)updates;
    p->shard_output           = (uintptr_t)output;
    p->outer_per_tile         = outer_size;
    p->elems_per_tile         = outer_size * data_axis_dim * inner_size;
    p->elems_indices_per_tile = outer_size * indices_axis_dim * inner_size;
    p->outer_start            = 0;
    p->inner_size             = inner_size;
    p->data_axis_dim          = data_axis_dim;
    p->indices_axis_dim       = indices_axis_dim;
    p->axis                   = axis;
}

#endif /* SCATTER_FP16_SPATZ_H_ */
