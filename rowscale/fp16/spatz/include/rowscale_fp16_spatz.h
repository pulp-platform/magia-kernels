// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROWSCALE_FP16_SPATZ_H_
#define ROWSCALE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "rowscale_fp16_spatz_params.h"

/*
 * Y[r, c] = X[r, c] * S[r] for a rows x cols matrix X, i.e. Y = diag(S) X: S holds one scale
 * factor per row, broadcast along the columns.
 */
void MAGIA_rowscale_fp16_spatz(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X, S, Y already in this tile's L1. */
void MAGIA_rowscale_fp16_spatz_l1(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols);

/*
 * As MAGIA_rowscale_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_rowscale_fp16_spatz_l1_start(rowscale_fp16_spatz_params_t *p,
                                        const float16 *X,
                                        const float16 *S,
                                        float16 *Y,
                                        uint32_t rows,
                                        uint32_t cols,
                                        mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_rowscale_fp16_spatz_l1_start. */
static inline void rowscale_fp16_spatz_params_set(rowscale_fp16_spatz_params_t *p,
                                                  const float16 *X,
                                                  const float16 *S,
                                                  float16 *Y,
                                                  uint32_t rows,
                                                  uint32_t cols)
{
    p->shard_X   = (uintptr_t)X;
    p->shard_S   = (uintptr_t)S;
    p->shard_Y   = (uintptr_t)Y;
    p->cols      = cols;
    p->row_start = 0;
    p->row_len   = rows;
}

#endif /* ROWSCALE_FP16_SPATZ_H_ */
