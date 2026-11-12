// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_spatz.h"
#include "mg_tile.h"

#include "swiglu_fp16_spatz.h"
#include "swiglu_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "swiglu_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elems)
{
    volatile swiglu_fp16_spatz_params_t *sw_params;
    uint32_t shard_G;
    uint32_t shard_U;
    uint32_t elem_start;
    uint32_t elem_len;

    mg_split(total_elems, NUM_HARTS, HID, &elem_start, &elem_len);

    l1_alloc_init();

    sw_params = l1_alloc(sizeof(swiglu_fp16_spatz_params_t));
    if (!sw_params)
        return ENOMEM;

    /* The task writes Y over G: one shard for G and Y. */
    shard_G = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_U = (uint32_t)l1_alloc(elem_len * sizeof(float16));
    if (!shard_U)
        return ENOMEM;

    sw_params->shard_G     = shard_G;
    sw_params->shard_U     = shard_U;
    sw_params->shard_Y     = shard_G;
    sw_params->elem_start  = elem_start;
    sw_params->elem_len    = elem_len;
    sw_params->total_elems = total_elems;

    *params = (void *)sw_params;

    return 0;
}

static int init_input_params(void *params, const float16 *G, const float16 *U)
{
    volatile swiglu_fp16_spatz_params_t *sw_params = params;

    if (sw_params->elem_len == 0)
        return 0;

    /* This tile's slice [elem_start, elem_start+elem_len) is contiguous in G and U. */
    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(G + sw_params->elem_start),
                           sw_params->shard_G,
                           sw_params->elem_len * sizeof(float16));

    mg_idma_memcpy_1d_sync(WFE,
                           0,
                           (uint32_t)(U + sw_params->elem_start),
                           sw_params->shard_U,
                           sw_params->elem_len * sizeof(float16));

    return 0;
}

static int store_result(void *params, float16 *Y)
{
    volatile swiglu_fp16_spatz_params_t *sw_params = params;

    if (sw_params->elem_len == 0)
        return 0;

    /* This tile's output slice [elem_start, elem_start+elem_len) is contiguous in L2. */
    mg_idma_memcpy_1d_sync(WFE,
                           1,
                           (uint32_t)(Y + sw_params->elem_start),
                           sw_params->shard_Y,
                           sw_params->elem_len * sizeof(float16));

    return 0;
}

void MAGIA_swiglu_fp16_spatz(const float16 *G, const float16 *U, float16 *Y, uint32_t total_elems)
{
    int ret;
    volatile swiglu_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, total_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, G, U);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = mg_spatz_call(SWIGLU_FP16_SPATZ_TASK, (const void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}

void MAGIA_swiglu_fp16_spatz_l1_start(swiglu_fp16_spatz_params_t *p,
                                      const float16 *G,
                                      const float16 *U,
                                      float16 *Y,
                                      uint32_t n,
                                      mg_event_t *event)
{
    swiglu_fp16_spatz_params_set(p, G, U, Y, n);
    mg_spatz_run(WFE, SWIGLU_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_swiglu_fp16_spatz_l1(const float16 *G, const float16 *U, float16 *Y, uint32_t n)
{
    swiglu_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_swiglu_fp16_spatz_l1_start(&params, G, U, Y, n, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
