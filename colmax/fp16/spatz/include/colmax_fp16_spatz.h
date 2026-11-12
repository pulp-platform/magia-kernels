// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef COLMAX_FP16_SPATZ_H_
#define COLMAX_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "colmax_fp16_spatz_params.h"

/*
 * M[c] = max_r X[r, c] for a rows x cols matrix X (cols even). The columns are split over all
 * tiles, in pairs.
 */
void MAGIA_colmax_fp16_spatz(const float16 *X, float16 *M, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X and M already in this tile's L1. */
void MAGIA_colmax_fp16_spatz_l1(const float16 *X, float16 *M, uint32_t rows, uint32_t cols);

/*
 * As MAGIA_colmax_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_colmax_fp16_spatz_l1_start(colmax_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      float16 *M,
                                      uint32_t rows,
                                      uint32_t cols,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_colmax_fp16_spatz_l1_start. */
static inline void colmax_fp16_spatz_params_set(colmax_fp16_spatz_params_t *p,
                                                const float16 *X,
                                                float16 *M,
                                                uint32_t rows,
                                                uint32_t cols)
{
    p->shard_X = (uintptr_t)X;
    p->shard_M = (uintptr_t)M;
    p->dims    = rows | (cols << 16);
}

#endif /* COLMAX_FP16_SPATZ_H_ */
