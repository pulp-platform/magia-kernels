// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernel called (synchronously, on the CV32 control core) by the mgc-generated kernel src/mgc_unfused_attn.c.
// All pointer arguments are L1 addresses.
// Sizes are the kernel's arguments, read through the MGC_GLUE macros of mgc_unfused_attn.h.

#include <stdint.h>

#define MGC_GLUE
#include "mgc_unfused_attn.h"
#include "tile.h"
#include "mg_spatz.h"
#include "mg_tile.h"
#include "eventunit.h"
#include "softmax_rows_fp16_spatz_params.h"
#define SPATZ_BINARY_NO_DEFINE
#include "unfused_attn_spatz_task_bin.h"

/* Per-tile Spatz task parameters, in the per-tile L1. */
static softmax_rows_fp16_spatz_params_t sm_params __attribute__((section(".tile_bss")));

/**
 * x[r, :] = softmax(x[r, :] / sqrt(PROJ_SIZE)) in place, for a `rows` x `cols` fp16 matrix, on
 * this tile's Spatz.
 */
void l1_softmax_rows_fp16_spatz(uint32_t x, uint32_t rows, uint32_t cols)
{
    softmax_rows_fp16_spatz_params_t *p = &sm_params;

    /* The fixed field, on the first call (the scaled exp constants are never 0). */
    if (!p->min_coef) {
        union { float16 f[2]; uint32_t u; } min_coef;
        float16 s = mg_fp16_from_bits(QK_SCALE_FP16);

        min_coef.f[0] = (float16)(SOFTMAX_ROWS_EXP_MIN / (float)s);
        min_coef.f[1] = (float16)(SOFTMAX_ROWS_EXP_COEF * (float)s);
        p->min_coef   = min_coef.u;
    }
    p->shard_X = x;
    p->dims    = rows | (cols << 16);
    mg_spatz_start(SOFTMAX_ROWS_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}
