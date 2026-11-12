// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated kernel src/mgc_ffn_gelu.c.
// All pointer arguments are L1 addresses.

#include <stdint.h>

#define MGC_GLUE
#include "mgc_ffn_gelu.h"
#include "tile.h"
#include "mg_spatz.h"
#include "gelu_fast_fp16_spatz_params.h"
#include "eventunit.h"
#define SPATZ_BINARY_NO_DEFINE
#include "ffn_gelu_spatz_task_bin.h"

/* Per-tile Spatz task parameters, preallocated in the per-tile L1. */
static struct {
    gelu_fast_fp16_spatz_params_t gelu;
} spatz_params __attribute__((section(".tile_bss")));

/**
 * Starts x[i] = gelu(x[i]) for `n` fp16 elements, in place, on this tile's Spatz;
 * l1_spatz_wait1(x) waits for it. The parameter block is rewritten at every start (the
 * double-buffered chunks alternate): Spatz reads it only when the task starts, after the previous
 * one has been waited.
 */
void l1_gelu_start_fp16_spatz(uint32_t x, uint32_t n)
{
    gelu_fast_fp16_spatz_params_t *p = &spatz_params.gelu;

    p->shard_X = x;
    p->shard_Y = x;
    p->len     = n;
    mg_spatz_start(GELU_FAST_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel, which writes x.
 */
void l1_spatz_wait1(uint32_t x)
{
    mg_spatz_wait_last(WFE);
}
