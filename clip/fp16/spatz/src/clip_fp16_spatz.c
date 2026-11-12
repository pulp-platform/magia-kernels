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

#include "clip_fp16_spatz.h"
#include "clip_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "clip_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile clip_fp16_spatz_params_t *clip_params;
    uint32_t shard_start;
    uint32_t shard_len;

    mg_split(size, NUM_HARTS, HID, &shard_start, &shard_len);

    l1_alloc_init();

    clip_params = l1_alloc(sizeof(clip_fp16_spatz_params_t));
    if (!clip_params)
        return ENOMEM;

    clip_params->shard_input = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!clip_params->shard_input)
        return ENOMEM;

    clip_params->shard_output = (uintptr_t)l1_alloc(shard_len * sizeof(float16));
    if (!clip_params->shard_output)
        return ENOMEM;

    clip_params->min = (uintptr_t)l1_alloc(sizeof(float16));
    if (!clip_params->min)
        return ENOMEM;

    clip_params->max = (uintptr_t)l1_alloc(sizeof(float16));
    if (!clip_params->max)
        return ENOMEM;

    clip_params->start = shard_start;
    clip_params->len   = shard_len;
    clip_params->end   = shard_start + shard_len;

    *params = (void *)clip_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input, float16 minimum, float16 maximum)
{
    volatile clip_fp16_spatz_params_t *clip_params = params;

    mmio_fp16(clip_params->min) = minimum;
    mmio_fp16(clip_params->max) = maximum;

    if (clip_params->len == 0)
        return 0;

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = clip(X, min, max)), so shard_output is not zeroed here. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(input + clip_params->start),
                           clip_params->shard_input,
                           clip_params->len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *dst)
{
    volatile clip_fp16_spatz_params_t *clip_params = params;

    if (clip_params->len == 0)
        return 0;

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(dst + clip_params->start),
                           clip_params->shard_output,
                           clip_params->len * sizeof(float16));

    return 0;
}

void MAGIA_clip_fp16_spatz(
    const float16 *input, float16 *output, float16 min, float16 max, uint32_t size)
{
    int ret;
    volatile clip_fp16_spatz_params_t *params;

    PROF_PHASE(prof_cyc_alloc, ret, alloc_l1((void **)&params, size));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_in, ret, init_input_params((void *)params, input, min, max));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    PROF_PHASE(prof_cyc_compute, ret, mg_spatz_call(CLIP_FP16_SPATZ_TASK, (const void *)params));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    PROF_PHASE(prof_cyc_data_out, ret, store_result((void *)params, output));
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_clip_fp16_spatz_l1_start(clip_fp16_spatz_params_t *p,
                                    const float16 *input,
                                    float16 *output,
                                    float16 min,
                                    float16 max,
                                    uint32_t size,
                                    mg_event_t *event)
{
    clip_fp16_spatz_params_set(p, input, output, min, max, size);
    mg_spatz_run(WFE, CLIP_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_clip_fp16_spatz_l1(
    const float16 *input, float16 *output, float16 min, float16 max, uint32_t size)
{
    clip_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_clip_fp16_spatz_l1_start(&params, input, output, min, max, size, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
