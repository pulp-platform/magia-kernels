// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated kernel src/mgc_flat_attn_pipeopt4.c.
// All pointer arguments are L1 addresses.
// Sizes are the kernel's arguments, read through the MGC_GLUE macros of mgc_flat_attn_pipeopt4.h.
// The softmax uses the merged Spatz kernels kernels/expminus_rowsum and kernels/colsum_rescale.

#include <stdint.h>

#define MGC_GLUE
#include "mgc_flat_attn_pipeopt4.h"
#include "tile.h"
#include "mg_spatz.h"
#include "mg_tile.h"
#include "colmax_fp16_spatz_params.h"
#include "colsum_rescale_fp16_spatz_params.h"
#include "expminus_rowsum_fp16_spatz.h"
#include "rowmax_fp16_spatz_params.h"
#include "rowscale_fp16_spatz_params.h"
#include "rowslice_sum_div_fp16_spatz_params.h"
#include "eventunit.h"
#define SPATZ_BINARY_NO_DEFINE
#include "flat_attn_pipeopt4_spatz_task_bin.h"

/*
 * Per-tile Spatz task parameter blocks, preallocated in L1 so that each call only stores the
 * fields that change. The fixed fields (sizes and some pointers) are set on the first call,
 * detected by the block's size field still being zero, so they must be the same on every call.
 */
static struct {
    rowscale_fp16_spatz_params_t rowscale;
    rowmax_fp16_spatz_params_t rowmax;
    colmax_fp16_spatz_params_t colmax;
    expminus_rowsum_fp16_spatz_params_t expminus_rowsum;
    colsum_rescale_fp16_spatz_params_t colsum_rescale;
    rowslice_sum_div_fp16_spatz_params_t rowslice_sum_div;
} spatz_params __attribute__((section(".tile_bss")));

/**
 * y[r, c] = x[r, c] * s[r] for a `rows` x `cols` fp16 matrix, i.e. y = diag(s) x, on this
 * tile's Spatz.
 */
void l1_rowscale_fp16_spatz(uint32_t y, uint32_t x, uint32_t s, uint32_t rows, uint32_t cols)
{
    rowscale_fp16_spatz_params_t *p = &spatz_params.rowscale;

    if (!p->cols) {
        p->shard_X = x;
        p->shard_Y = y;
        p->cols    = cols;
        p->row_len = rows;
    }
    p->shard_S = s;
    mg_spatz_start(ROWSCALE_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/**
 * m[r] = max_c x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    rowmax_fp16_spatz_params_t *p = &spatz_params.rowmax;

    if (!p->dims) {
        p->shard_M = m;
        p->dims    = rows | (cols << 16);
    }
    p->shard_X = x;
    p->shard_P = 0;
    mg_spatz_start(ROWMAX_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/**
 * m[r] = max(max_c x[r, c], p[r]) for a `rows` x `cols` fp16 matrix, on this tile's Spatz.
 */
void l1_rowmax_max_fp16_spatz(uint32_t m, uint32_t x, uint32_t pm, uint32_t rows, uint32_t cols)
{
    rowmax_fp16_spatz_params_t *p = &spatz_params.rowmax;

    if (!p->dims) {
        p->shard_M = m;
        p->dims    = rows | (cols << 16);
    }
    p->shard_X = x;
    p->shard_P = pm;
    mg_spatz_start(ROWMAX_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/**
 * m[c] = max_r x[r, c] for a `rows` x `cols` fp16 matrix, on this tile's Spatz
 * (vectorized along the columns).
 */
void l1_colmax_fp16_spatz(uint32_t m, uint32_t x, uint32_t rows, uint32_t cols)
{
    colmax_fp16_spatz_params_t *p = &spatz_params.colmax;

    if (!p->dims) {
        p->shard_X = x;
        p->dims    = rows | (cols << 16);
    }
    p->shard_M = m;
    mg_spatz_start(COLMAX_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/* Sets the fixed fields of the expminus_rowsum block, with the scale 1/sqrt(PROJ_SIZE). */
static inline void expminus_rowsum_init(expminus_rowsum_fp16_spatz_params_t *p,
                                        uint32_t rows,
                                        uint32_t cols)
{
    expminus_rowsum_fp16_spatz_set_scale(
        p, mg_fp16_from_bits(QK_SCALE_FP16), rows, cols);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel. y and s are unused: they only tell
 * mgc which buffers the task writes.
 */
void l1_spatz_wait(uint32_t y, uint32_t s)
{
    mg_spatz_wait_last(WFE);
}

/**
 * Starts y[r, c] = exp((x[r, c] - m[r]) / sqrt(PROJ_SIZE)) and s[r] = sum_c y[r, c] for a
 * `rows` x `cols` fp16 matrix on this tile's Spatz; l1_spatz_wait() waits for it.
 */
void l1_expminus_rowsum_start_fp16_spatz(
    uint32_t y, uint32_t s, uint32_t x, uint32_t m, uint32_t rows, uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t *p = &spatz_params.expminus_rowsum;

    if (!p->dims)
        expminus_rowsum_init(p, rows, cols);
    p->shard_X = x;
    p->shard_M = m;
    p->shard_Y = y;
    p->shard_S = s;
    p->shard_A = 0;
    mg_spatz_start(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, p);
}

/**
 * As l1_expminus_rowsum_start_fp16_spatz, plus a[r] = exp((a[r] - m[r]) / sqrt(PROJ_SIZE)) in
 * place.
 */
void l1_expminus_rowsum_alpha_start_fp16_spatz(
    uint32_t y, uint32_t s, uint32_t a, uint32_t x, uint32_t m, uint32_t rows, uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t *p = &spatz_params.expminus_rowsum;

    if (!p->dims)
        expminus_rowsum_init(p, rows, cols);
    p->shard_X = x;
    p->shard_M = m;
    p->shard_Y = y;
    p->shard_S = s;
    p->shard_A = a;
    mg_spatz_start(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, p);
}

/**
 * s[c] += sum_{r >= 1} g[r, c] for a `rows` x `cols` fp16 matrix g, on this tile's Spatz.
 */
void l1_colsum_acc_fp16_spatz(uint32_t s, uint32_t g, uint32_t rows, uint32_t cols)
{
    colsum_rescale_fp16_spatz_params_t *p = &spatz_params.colsum_rescale;

    if (!p->dims) {
        p->shard_G = g;
        p->dims    = rows | (cols << 16);
    }
    p->shard_S = s;
    p->shard_P = 0;
    p->shard_A = 0;
    mg_spatz_start(COLSUM_RESCALE_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/**
 * As l1_colsum_acc_fp16_spatz, then p[c] = a[c] * p[c] and s[c] = p[c] + s[c].
 */
void l1_colsum_acc_rescale_fp16_spatz(
    uint32_t s, uint32_t g, uint32_t pp, uint32_t a, uint32_t rows, uint32_t cols)
{
    colsum_rescale_fp16_spatz_params_t *p = &spatz_params.colsum_rescale;

    if (!p->dims) {
        p->shard_G = g;
        p->dims    = rows | (cols << 16);
    }
    p->shard_S = s;
    p->shard_P = pp;
    p->shard_A = a;
    mg_spatz_start(COLSUM_RESCALE_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}

/**
 * Reduce-scatter of the per-tile outputs along the mesh row, normalized: this tile (column jj)
 * owns rows [jj * rows, (jj + 1) * rows) of o [n * rows x cols], and
 * y[r, c] = (o[jj * rows + r, c] + sum_{k != jj} g[k, r, c]) / s[jj * rows + r], where g
 * [n x rows x cols] holds the slices sent by the other tiles of the row (g[jj] unused).
 */
void l1_rowslice_sum_div_fp16_spatz(
    uint32_t y, uint32_t o, uint32_t g, uint32_t s, uint32_t n, uint32_t rows, uint32_t cols)
{
    rowslice_sum_div_fp16_spatz_params_t *p = &spatz_params.rowslice_sum_div;
    uint32_t jj = GET_X_ID(get_hartid());

    if (!p->dims) {
        p->shard_Y = y;
        p->shard_O = o + jj * rows * cols * sizeof(float16);
        p->shard_G = g;
        p->dims    = rows | (cols << 16);
        p->slices  = n | (jj << 16);
    }
    p->shard_S = s + jj * rows * sizeof(float16);
    mg_spatz_start(ROWSLICE_SUM_DIV_FP16_SPATZ_TASK, p);
    mg_spatz_wait_last(WFE);
}
