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

#include "gather_fp16_spatz.h"
#include "gather_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "gather_fp16_spatz"

static int alloc_l1(void **params,
                    uint32_t in_shape[4],
                    uint32_t batch,
                    uint32_t gather_dim_size,
                    uint32_t axis_length,
                    uint32_t index,
                    uint32_t axis)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uint32_t num_batches;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t in_batch_stride;

    if (axis == 1) {
        num_batches = in_shape[0];
    } else if (axis == 2) {
        num_batches = in_shape[0] * in_shape[1];
    } else if (axis == 3) {
        num_batches = in_shape[0] * in_shape[1] * in_shape[2];
    } else {
        return EINVAL;
    }

    mg_split(num_batches, NUM_HARTS, HID, &batch_start, &batch_len);

    in_batch_stride = gather_dim_size * axis_length;

    l1_alloc_init();

    gather_params = l1_alloc(sizeof(gather_fp16_spatz_params_t));
    if (!gather_params)
        return ENOMEM;

    gather_params->shard_input = (uintptr_t)l1_alloc(batch_len * in_batch_stride * sizeof(float16));
    if (!gather_params->shard_input)
        return ENOMEM;

    gather_params->shard_output = (uintptr_t)l1_alloc(batch_len * axis_length * sizeof(float16));
    if (!gather_params->shard_output)
        return ENOMEM;

    gather_params->gather_dim_size = gather_dim_size;
    gather_params->axis_length     = axis_length;
    gather_params->batch_start     = batch_start;
    gather_params->batch_len       = batch_len;
    gather_params->index           = index;

    *params = (void *)gather_params;

    return 0;
}

static int init_input_params(void *params, const float16 *data)
{
    volatile gather_fp16_spatz_params_t *gather_params = params;
    uint32_t in_batch_stride;
    uint32_t batch_start;
    uint32_t batch_len;

    in_batch_stride = gather_params->gather_dim_size * gather_params->axis_length;
    batch_start     = gather_params->batch_start;
    batch_len       = gather_params->batch_len;

    if (batch_len == 0)
        return 0;

    /* This tile's batches [batch_start, batch_start+batch_len) are a contiguous L2 block.
       The Spatz task fully writes shard_output (axis_length per batch), so it is not zeroed
       here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(data + batch_start * in_batch_stride),
                           gather_params->shard_input,
                           batch_len * in_batch_stride * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *gather_result)
{
    volatile gather_fp16_spatz_params_t *gather_params = params;
    uint32_t axis_length;
    uint32_t start;
    uint32_t len;

    axis_length = gather_params->axis_length;
    start       = gather_params->batch_start;
    len         = gather_params->batch_len;

    if (len == 0)
        return 0;

    /* This tile's output batches [batch_start, batch_start+batch_len) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(gather_result + start * axis_length),
                           gather_params->shard_output,
                           len * axis_length * sizeof(float16));

    return 0;
}

void MAGIA_gather_fp16_spatz(const float16 *data,
                             uint32_t in_shape[4],
                             float16 *output,
                             uint32_t batch,
                             uint32_t gather_dim_size,
                             uint32_t axis_length,
                             uint32_t index,
                             uint32_t axis)
{
    int ret;
    volatile gather_fp16_spatz_params_t *params;

    if (axis == 0) {
        printf("[CV32 (%d)] [%s] Axis 0 gather is not supported on Spatz kernel! Handle via flat "
               "L2 DMA copy.\n",
               HID,
               KERNEL_NAME);
        return;
    }

    PROF_PHASE(
        prof_cyc_alloc,
        ret,
        alloc_l1((void **)&params, in_shape, batch, gather_dim_size, axis_length, index, axis));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, data));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(GATHER_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_gather_fp16_spatz_l1_start(gather_fp16_spatz_params_t *p,
                                      const float16 *data,
                                      uint32_t in_shape[4],
                                      float16 *output,
                                      uint32_t batch,
                                      uint32_t gather_dim_size,
                                      uint32_t axis_length,
                                      uint32_t index,
                                      uint32_t axis,
                                      mg_event_t *event)
{
    gather_fp16_spatz_params_set(
        p, data, in_shape, output, batch, gather_dim_size, axis_length, index, axis);
    mg_spatz_run(WFE, GATHER_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_gather_fp16_spatz_l1(const float16 *data,
                                uint32_t in_shape[4],
                                float16 *output,
                                uint32_t batch,
                                uint32_t gather_dim_size,
                                uint32_t axis_length,
                                uint32_t index,
                                uint32_t axis)
{
    gather_fp16_spatz_params_t params;
    mg_event_t event;

    if (axis == 0 || axis > 3) {
        printf("[CV32 (%d)] [%s] Unsupported gather axis: %d\n", HID, KERNEL_NAME, (int)axis);
        return;
    }

    MAGIA_gather_fp16_spatz_l1_start(
        &params, data, in_shape, output, batch, gather_dim_size, axis_length, index, axis, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
