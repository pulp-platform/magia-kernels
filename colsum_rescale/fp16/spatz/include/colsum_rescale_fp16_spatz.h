// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef COLSUM_RESCALE_FP16_SPATZ_H_
#define COLSUM_RESCALE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "colsum_rescale_fp16_spatz_params.h"

/*
 * S[c] = S[c] + sum_{r >= 1} G[r, c]; if P is not NULL, then P[c] = A[c] * P[c] and
 * S[c] = P[c] + S[c] (cols even). The columns are split over all tiles, in pairs.
 */
void MAGIA_colsum_rescale_fp16_spatz(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols);

/*
 * As MAGIA_colsum_rescale_fp16_spatz_l1, but only starts the Spatz task, with the parameter block
 * `p` (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_colsum_rescale_fp16_spatz_l1_start(colsum_rescale_fp16_spatz_params_t *p,
                                              float16 *S,
                                              const float16 *G,
                                              float16 *P,
                                              const float16 *A,
                                              uint32_t rows,
                                              uint32_t cols,
                                              mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_colsum_rescale_fp16_spatz_l1_start. */
static inline void colsum_rescale_fp16_spatz_params_set(colsum_rescale_fp16_spatz_params_t *p,
                                                        float16 *S,
                                                        const float16 *G,
                                                        float16 *P,
                                                        const float16 *A,
                                                        uint32_t rows,
                                                        uint32_t cols)
{
    p->shard_S = (uintptr_t)S;
    p->shard_G = (uintptr_t)G;
    p->shard_P = (uintptr_t)P;
    p->shard_A = (uintptr_t)A;
    p->dims    = rows | (cols << 16);
}

#endif /* COLSUM_RESCALE_FP16_SPATZ_H_ */
