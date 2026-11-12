// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROWMAX_FP16_SPATZ_H_
#define ROWMAX_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "rowmax_fp16_spatz_params.h"

/*
 * M[r] = max(max_c X[r, c], P[r]) for a rows x cols matrix X. P may be NULL, then
 * M[r] = max_c X[r, c]. The rows are split over all tiles.
 */
void MAGIA_rowmax_fp16_spatz(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X, P, M already in this tile's L1. */
void MAGIA_rowmax_fp16_spatz_l1(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols);

/*
 * As MAGIA_rowmax_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_rowmax_fp16_spatz_l1_start(rowmax_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      const float16 *P,
                                      float16 *M,
                                      uint32_t rows,
                                      uint32_t cols,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_rowmax_fp16_spatz_l1_start. */
static inline void rowmax_fp16_spatz_params_set(rowmax_fp16_spatz_params_t *p,
                                                const float16 *X,
                                                const float16 *P,
                                                float16 *M,
                                                uint32_t rows,
                                                uint32_t cols)
{
    p->shard_X = (uintptr_t)X;
    p->shard_P = (uintptr_t)P;
    p->shard_M = (uintptr_t)M;
    p->dims    = rows | (cols << 16);
}

#endif /* ROWMAX_FP16_SPATZ_H_ */
