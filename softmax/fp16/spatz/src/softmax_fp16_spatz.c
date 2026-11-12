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

#include "softmax_fp16_spatz.h"
#include "softmax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "softmax_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t block;
    uint32_t outer_start;
    uint32_t outer_len;

    block = reduce_dim * inner_dim;

    mg_split(outer_dim, NUM_HARTS, HID, &outer_start, &outer_len);

    l1_alloc_init();

    softmax_params = l1_alloc(sizeof(softmax_fp16_spatz_params_t));
    if (!softmax_params)
        return ENOMEM;

    shard_input = l1_alloc(outer_len * block * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(outer_len * block * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    softmax_params->shard_input  = shard_input;
    softmax_params->shard_output = shard_output;
    softmax_params->reduce_dim   = reduce_dim;
    softmax_params->inner_dim    = inner_dim;
    softmax_params->outer_start  = outer_start;
    softmax_params->outer_len    = outer_len;

    *params = (void *)softmax_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile softmax_fp16_spatz_params_t *softmax_params = params;
    uint32_t outer_start;
    uint32_t outer_len;
    uint32_t block;

    outer_start = softmax_params->outer_start;
    outer_len   = softmax_params->outer_len;
    block       = softmax_params->reduce_dim * softmax_params->inner_dim;

    if (outer_len == 0)
        return 0;

    /* This tile's outer slices [outer_start, outer_start+outer_len) are contiguous in L2. The
       Spatz task writes every output element, so shard_output is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(input + outer_start * block),
                           softmax_params->shard_input,
                           outer_len * block * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *output)
{
    volatile softmax_fp16_spatz_params_t *softmax_params = params;
    uint32_t outer_start;
    uint32_t outer_len;
    uint32_t block;

    outer_start = softmax_params->outer_start;
    outer_len   = softmax_params->outer_len;
    block       = softmax_params->reduce_dim * softmax_params->inner_dim;

    if (outer_len == 0)
        return 0;

    /* This tile's output slices [outer_start, outer_start+outer_len) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(output + outer_start * block),
                           softmax_params->shard_output,
                           outer_len * block * sizeof(float16));

    return 0;
}

void MAGIA_softmax_fp16_spatz(const float16 *input,
                              float16 *output,
                              uint32_t outer_dim,
                              uint32_t reduce_dim,
                              uint32_t inner_dim)
{
    int ret;
    volatile softmax_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, outer_dim, reduce_dim, inner_dim));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, input));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(SOFTMAX_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_softmax_fp16_spatz_l1_start(softmax_fp16_spatz_params_t *p,
                                       const float16 *input,
                                       float16 *output,
                                       uint32_t outer_dim,
                                       uint32_t reduce_dim,
                                       uint32_t inner_dim,
                                       mg_event_t *event)
{
    softmax_fp16_spatz_params_set(p, input, output, outer_dim, reduce_dim, inner_dim);
    mg_spatz_run(WFE, SOFTMAX_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_softmax_fp16_spatz_l1(const float16 *input,
                                 float16 *output,
                                 uint32_t outer_dim,
                                 uint32_t reduce_dim,
                                 uint32_t inner_dim)
{
    softmax_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_softmax_fp16_spatz_l1_start(
        &params, input, output, outer_dim, reduce_dim, inner_dim, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
