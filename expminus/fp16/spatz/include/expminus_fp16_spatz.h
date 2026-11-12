// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXPMINUS_FP16_SPATZ_H_
#define EXPMINUS_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "expminus_fp16_spatz_params.h"

/*
 * Y[r, c] = exp(X[r, c] - M[r]) for a rows x cols matrix X: M holds one value per row, broadcast
 * along the columns. M[r] must be >= every X[r, c] (e.g. the row max).
 */
void MAGIA_expminus_fp16_spatz(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols);

/* Same operation on this tile only, with X, M, Y already in this tile's L1. Y may alias X. */
void MAGIA_expminus_fp16_spatz_l1(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols);

/*
 * As MAGIA_expminus_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_expminus_fp16_spatz_l1_start(expminus_fp16_spatz_params_t *p,
                                        const float16 *X,
                                        const float16 *M,
                                        float16 *Y,
                                        uint32_t rows,
                                        uint32_t cols,
                                        mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_expminus_fp16_spatz_l1_start. */
static inline void expminus_fp16_spatz_params_set(expminus_fp16_spatz_params_t *p,
                                                  const float16 *X,
                                                  const float16 *M,
                                                  float16 *Y,
                                                  uint32_t rows,
                                                  uint32_t cols)
{
    p->shard_X   = (uintptr_t)X;
    p->shard_M   = (uintptr_t)M;
    p->shard_Y   = (uintptr_t)Y;
    p->cols      = cols;
    p->row_start = 0;
    p->row_len   = rows;
}

#endif /* EXPMINUS_FP16_SPATZ_H_ */
