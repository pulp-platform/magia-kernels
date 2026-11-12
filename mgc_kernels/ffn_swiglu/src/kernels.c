// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated kernel src/mgc_ffn_swiglu.c.
// All pointer arguments are L1 addresses.

#include <stdint.h>

#define MGC_GLUE
#include "mgc_ffn_swiglu.h"
#include "tile.h"
#include "mg_spatz.h"
#include "swiglu_fast_fp16_spatz_params.h"
#include "eventunit.h"
#define SPATZ_BINARY_NO_DEFINE
#include "ffn_swiglu_spatz_task_bin.h"

/* Per-tile Spatz task parameters, preallocated in the per-tile L1. */
static struct {
    swiglu_fast_fp16_spatz_params_t swiglu;
} spatz_params __attribute__((section(".tile_bss")));

/**
 * Starts g[i] = silu(g[i]) * u[i] for `n` fp16 elements, in place, on this tile's Spatz;
 * l1_spatz_wait1(g) waits for it. The parameter block is rewritten at every start (the
 * double-buffered chunks alternate): Spatz reads it only when the task starts, after the previous
 * one has been waited.
 */
void l1_swiglu_start_fp16_spatz(uint32_t g, uint32_t u, uint32_t n)
{
    swiglu_fast_fp16_spatz_params_t *p = &spatz_params.swiglu;

    p->shard_G = g;
    p->shard_U = u;
    p->shard_Y = g;
    p->len     = n;
    mg_spatz_start(SWIGLU_FAST_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel, which writes x.
 */
void l1_spatz_wait1(uint32_t x)
{
    mg_spatz_wait_last(WFE);
}
