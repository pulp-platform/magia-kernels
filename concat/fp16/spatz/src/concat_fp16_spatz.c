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

#include "concat_fp16_spatz.h"
#include "concat_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "concat_fp16_spatz"

static int alloc_l1(void **params, const uint32_t *lens, uint32_t num_inputs, uint32_t iterations)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uintptr_t *shard_input;
    uint32_t *len_input;
    uintptr_t shard;
    uintptr_t shard_output;

    uint32_t iter_start;
    uint32_t iter_len;
    size_t out_len;

    mg_split(iterations, NUM_HARTS, HID, &iter_start, &iter_len);

    l1_alloc_init();

    concat_params = l1_alloc(sizeof(concat_fp16_spatz_params_t));
    if (!concat_params)
        return ENOMEM;

    shard_input = l1_alloc(num_inputs * sizeof(uintptr_t));
    if (!shard_input)
        return ENOMEM;

    len_input = l1_alloc(num_inputs * sizeof(uint32_t));
    if (!len_input)
        return ENOMEM;

    out_len = 0;
    for (uint32_t i = 0; i < num_inputs; i++) {
        shard = l1_alloc(iter_len * lens[i] * sizeof(float16));
        if (!shard)
            return ENOMEM;

        shard_input[i] = shard;
        len_input[i]   = lens[i];
        out_len += lens[i];
    }

    shard_output = l1_alloc(iter_len * out_len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    concat_params->shard_input  = (uintptr_t)shard_input;
    concat_params->len_input    = (uintptr_t)len_input;
    concat_params->shard_output = shard_output;
    concat_params->num_inputs   = num_inputs;
    concat_params->iter_start   = iter_start;
    concat_params->iter_len     = iter_len;

    *params = (void *)concat_params;

    return 0;
}

static int init_input_params(void *params, const float16 **inputs)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uintptr_t *shard_input;
    uint32_t *len_input;
    uint32_t iter_start;
    uint32_t iter_len;
    uint32_t num_inputs;
    uint32_t len_in;

    concat_params = (volatile concat_fp16_spatz_params_t *)params;

    shard_input = (uintptr_t *)concat_params->shard_input;
    len_input   = (uint32_t *)concat_params->len_input;
    iter_start  = concat_params->iter_start;
    iter_len    = concat_params->iter_len;
    num_inputs  = concat_params->num_inputs;

    if (iter_len == 0)
        return 0;

    /* Each input's iterations are contiguous in L2, so this tile's [iter_start, iter_end)
       slice is a single contiguous block per input. The Spatz task fully writes the output
       shard (in0 || in1 || ... per iteration), so shard_output is not zeroed here. */
    for (uint32_t i = 0; i < num_inputs; i++) {
        len_in = len_input[i];
        if (len_in) {
            mg_idma_memcpy_1d_sync(WFE,
                                   0,
                                   (uint32_t)(inputs[i] + iter_start * len_in),
                                   (uint32_t)shard_input[i],
                                   iter_len * len_in * sizeof(float16));
        }
    }

    return 0;
}

static int store_result(void *params, float16 *concat_result, const uint32_t iterations)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uint32_t *len_input;
    uint32_t shard_output_base;
    uint32_t num_inputs;
    uint32_t out_len;
    uint32_t start;
    uint32_t len;

    concat_params     = (volatile concat_fp16_spatz_params_t *)params;
    len_input         = (uint32_t *)concat_params->len_input;
    shard_output_base = concat_params->shard_output;
    num_inputs        = concat_params->num_inputs;
    start             = concat_params->iter_start;
    len               = concat_params->iter_len;

    if (len == 0)
        return 0;

    out_len = 0;
    for (uint32_t i = 0; i < num_inputs; i++)
        out_len += len_input[i];

    /* This tile's output rows [iter_start, iter_start+iter_len) are contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(concat_result + start * out_len),
                           (uint32_t)shard_output_base,
                           len * out_len * sizeof(float16));

    return 0;
}

void MAGIA_concat_fp16_spatz(const float16 **inputs,
                             const uint32_t *lens,
                             uint32_t num_inputs,
                             float16 *concat_result,
                             uint32_t iterations)
{
    int ret;
    volatile concat_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, lens, num_inputs, iterations));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, inputs));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(CONCAT_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, concat_result, iterations));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_concat_fp16_spatz_l1_start(concat_fp16_spatz_params_t *p,
                                      const float16 **inputs,
                                      const uint32_t *lens,
                                      uint32_t num_inputs,
                                      float16 *concat_result,
                                      uint32_t iterations,
                                      mg_event_t *event)
{
    concat_fp16_spatz_params_set(p, inputs, lens, num_inputs, concat_result, iterations);
    mg_spatz_run(WFE, CONCAT_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_concat_fp16_spatz_l1(const float16 **inputs,
                                const uint32_t *lens,
                                uint32_t num_inputs,
                                float16 *concat_result,
                                uint32_t iterations)
{
    const float16 *shard_input[num_inputs];
    uint32_t len_input[num_inputs];
    concat_fp16_spatz_params_t params;
    mg_event_t event;

    /*
     * The task reads the input addresses and lengths through the params: stage them on this
     * core's stack, which is in L1.
     */
    for (uint32_t i = 0; i < num_inputs; i++) {
        shard_input[i] = inputs[i];
        len_input[i]   = lens[i];
    }

    MAGIA_concat_fp16_spatz_l1_start(
        &params, shard_input, len_input, num_inputs, concat_result, iterations, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
