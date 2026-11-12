// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "kernels_profiling_utils.h"

#include "scatter_fp16_spatz.h"
#include "scatter_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "scatter_fp16_spatz"

static int allocate_l1(void **params,
                       uint32_t outer_size,
                       uint32_t inner_size,
                       uint32_t axis,
                       uint32_t data_axis_dim,
                       uint32_t indices_axis_dim)
{
    volatile scatter_fp16_spatz_params_t *scatter_params;
    uintptr_t shard_data;
    uintptr_t shard_indices;
    uintptr_t shard_updates;
    uintptr_t shard_output;

    uint32_t iter_start;
    uint32_t iter_len;
    uint32_t elems_data;
    uint32_t elems_indices;

    mg_split(outer_size, NUM_HARTS, HID, &iter_start, &iter_len);

    elems_data    = iter_len * data_axis_dim * inner_size;
    elems_indices = iter_len * indices_axis_dim * inner_size;

    l1_alloc_init();

    scatter_params = l1_alloc(sizeof(scatter_fp16_spatz_params_t));
    if (!scatter_params)
        return ENOMEM;

    shard_data = l1_alloc(elems_data * sizeof(float16));
    if (!shard_data)
        return ENOMEM;

    shard_indices = l1_alloc(elems_indices * sizeof(uint32_t));
    if (!shard_indices)
        return ENOMEM;

    shard_updates = l1_alloc(elems_indices * sizeof(float16));
    if (!shard_updates)
        return ENOMEM;

    shard_output = l1_alloc(elems_data * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    scatter_params->shard_data    = shard_data;
    scatter_params->shard_indices = shard_indices;
    scatter_params->shard_updates = shard_updates;
    scatter_params->shard_output  = shard_output;

    scatter_params->outer_per_tile         = iter_len;
    scatter_params->elems_per_tile         = elems_data;
    scatter_params->elems_indices_per_tile = elems_indices;
    scatter_params->outer_start            = iter_start;
    scatter_params->inner_size             = inner_size;
    scatter_params->data_axis_dim          = data_axis_dim;
    scatter_params->indices_axis_dim       = indices_axis_dim;
    scatter_params->axis                   = axis;

    *params = (void *)scatter_params;
    return 0;
}

static int init_input_params(void *params,
                             const float16 *input,
                             const int64_t *indices,
                             const float16 *updates)
{
    volatile scatter_fp16_spatz_params_t *scatter_params;
    uintptr_t shard_indices;

    uint32_t elems_per_tile;
    uint32_t elems_indices_per_tile;
    uint32_t global_offset_data;
    uint32_t global_offset_indices;
    uint32_t iter_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;
    uint32_t indices_axis_dim;

    scatter_params = (volatile scatter_fp16_spatz_params_t *)params;

    shard_indices = scatter_params->shard_indices;

    elems_per_tile         = scatter_params->elems_per_tile;
    elems_indices_per_tile = scatter_params->elems_indices_per_tile;
    iter_start             = scatter_params->outer_start;
    inner_size             = scatter_params->inner_size;
    data_axis_dim          = scatter_params->data_axis_dim;
    indices_axis_dim       = scatter_params->indices_axis_dim;

    if (elems_per_tile == 0)
        return 0;

    global_offset_data    = iter_start * data_axis_dim * inner_size;
    global_offset_indices = iter_start * indices_axis_dim * inner_size;

    /* indices are int64 in L2 but int32 in L1 -> narrowing conversion, staged scalar. */
    for (uint32_t i = 0; i < elems_indices_per_tile; i++)
        mmio32(shard_indices + i * sizeof(uint32_t)) = (int32_t)indices[global_offset_indices + i];

    /* data and updates are plain fp16 copies of contiguous L2 blocks. shard_output is not
       zeroed here: the Spatz task's step 1 copies shard_data over it in full. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(input + global_offset_data),
                           scatter_params->shard_data,
                           elems_per_tile * sizeof(float16));
    if (elems_indices_per_tile) {
        mg_idma_memcpy_1d_sync(WFE,
                               0,
                               (uint32_t)(updates + global_offset_indices),
                               scatter_params->shard_updates,
                               elems_indices_per_tile * sizeof(float16));
    }

    return 0;
}

static int store_result(void *params, float16 *output)
{
    volatile scatter_fp16_spatz_params_t *scatter_params;
    uint32_t elems_per_tile;
    uint32_t global_offset;
    uint32_t iter_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;

    scatter_params = (volatile scatter_fp16_spatz_params_t *)params;
    elems_per_tile = scatter_params->elems_per_tile;
    iter_start     = scatter_params->outer_start;
    inner_size     = scatter_params->inner_size;
    data_axis_dim  = scatter_params->data_axis_dim;

    if (elems_per_tile == 0)
        return 0;

    global_offset = iter_start * data_axis_dim * inner_size;

    /* This tile's output slice is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(output + global_offset),
                           scatter_params->shard_output,
                           elems_per_tile * sizeof(float16));

    return 0;
}

void MAGIA_scatter_fp16_spatz(const float16 *data,
                              const int64_t *indices,
                              const float16 *updates,
                              float16 *output,
                              uint32_t outer_size,
                              uint32_t inner_size,
                              uint32_t axis,
                              uint32_t data_axis_dim,
                              uint32_t indices_axis_dim)
{
    int ret;
    void *params;

    PROF_PHASE(
        prof_cyc_alloc,
        ret,
        allocate_l1(
            (void **)&params, outer_size, inner_size, axis, data_axis_dim, indices_axis_dim));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, data, indices, updates));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(SCATTER_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

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
                                       mg_event_t *event)
{
    scatter_fp16_spatz_params_set(p,
                                  data,
                                  indices,
                                  updates,
                                  output,
                                  outer_size,
                                  inner_size,
                                  axis,
                                  data_axis_dim,
                                  indices_axis_dim);
    mg_spatz_run(WFE, SCATTER_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_scatter_fp16_spatz_l1(const float16 *data,
                                 const int32_t *indices,
                                 const float16 *updates,
                                 float16 *output,
                                 uint32_t outer_size,
                                 uint32_t inner_size,
                                 uint32_t axis,
                                 uint32_t data_axis_dim,
                                 uint32_t indices_axis_dim)
{
    scatter_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_scatter_fp16_spatz_l1_start(&params,
                                      data,
                                      indices,
                                      updates,
                                      output,
                                      outer_size,
                                      inner_size,
                                      axis,
                                      data_axis_dim,
                                      indices_axis_dim,
                                      &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
