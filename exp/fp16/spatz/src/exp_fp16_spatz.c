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

#include "exp_fp16_spatz.h"
#include "exp_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "exp_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile exp_fp16_spatz_params_t *exp_params;
    uint32_t start;
    uint32_t len;

    mg_split(size, NUM_HARTS, HID, &start, &len);

    l1_alloc_init();

    exp_params = l1_alloc(sizeof(exp_fp16_spatz_params_t));
    if (!exp_params)
        return ENOMEM;

    exp_params->shard_input = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!exp_params->shard_input)
        return ENOMEM;

    exp_params->shard_output = (uintptr_t)l1_alloc(len * sizeof(float16));
    if (!exp_params->shard_output)
        return ENOMEM;

    exp_params->start = start;
    exp_params->len   = len;
    exp_params->end   = start + len;

    *params = (void *)exp_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile exp_fp16_spatz_params_t *exp_params = params;

    if (exp_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = exp(X)), so shard_output is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(input + exp_params->start),
                           exp_params->shard_input,
                           exp_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile exp_fp16_spatz_params_t *exp_params = params;

    if (exp_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + exp_params->start),
                           exp_params->shard_output,
                           exp_params->len * sizeof(float16));

    return 0;
}

void MAGIA_exp_fp16_spatz(const float16 *input, float16 *output, uint32_t size)
{
    int ret;
    volatile exp_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, size));
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

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(EXP_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_exp_fp16_spatz_l1_start(exp_fp16_spatz_params_t *p,
                                   const float16 *input,
                                   float16 *output,
                                   uint32_t n,
                                   mg_event_t *event)
{
    exp_fp16_spatz_params_set(p, input, output, n);
    mg_spatz_run(WFE, EXP_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_exp_fp16_spatz_l1(const float16 *input, float16 *output, uint32_t n)
{
    exp_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_exp_fp16_spatz_l1_start(&params, input, output, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
