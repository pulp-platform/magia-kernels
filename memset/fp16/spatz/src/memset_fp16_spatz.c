// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "tile.h"
#include "mg_spatz.h"

#include "memset_fp16_spatz.h"
#include "memset_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "memset_fp16_spatz"

void MAGIA_memset_zero_fp16_spatz_l1_start(memset_fp16_spatz_params_t *p,
                                           float16 *C,
                                           uint32_t size,
                                           mg_event_t *event)
{
    memset_fp16_spatz_params_set(p, C, size);
    mg_spatz_run(WFE, MEMSET_FP16_SPATZ_TASK, p, event, 0);
}

void MAGIA_memset_zero_fp16_spatz_l1(float16 *C, uint32_t size)
{
    memset_fp16_spatz_params_t params;
    mg_event_t event;

    MAGIA_memset_zero_fp16_spatz_l1_start(&params, C, size, &event);
    mg_spatz_wait(WFE, &event);
    if (mg_spatz_exit_code() != 0)
        printf("[CV32 (%d)] [%s] Spatz task failed with exit code: %d\n",
               HID,
               KERNEL_NAME,
               mg_spatz_exit_code());
}
