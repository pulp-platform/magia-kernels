// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROWSLICE_SUM_DIV_FP16_SPATZ_H_
#define ROWSLICE_SUM_DIV_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "rowslice_sum_div_fp16_spatz_params.h"

/*
 * Y[r, c] = (O[r, c] + sum_{k != own} G[k, r, c]) / S[r], for k < n, with G [n x rows x cols].
 * The rows are split over all tiles.
 */
void MAGIA_rowslice_sum_div_fp16_spatz(float16 *Y,
                                       const float16 *O,
                                       const float16 *G,
                                       const float16 *S,
                                       uint32_t n,
                                       uint32_t own,
                                       uint32_t rows,
                                       uint32_t cols);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols);

/*
 * As MAGIA_rowslice_sum_div_fp16_spatz_l1, but only starts the Spatz task, with the parameter block
 * `p` (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_rowslice_sum_div_fp16_spatz_l1_start(rowslice_sum_div_fp16_spatz_params_t *p,
                                                float16 *Y,
                                                const float16 *O,
                                                const float16 *G,
                                                const float16 *S,
                                                uint32_t n,
                                                uint32_t own,
                                                uint32_t rows,
                                                uint32_t cols,
                                                mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_rowslice_sum_div_fp16_spatz_l1_start. */
static inline void rowslice_sum_div_fp16_spatz_params_set(rowslice_sum_div_fp16_spatz_params_t *p,
                                                          float16 *Y,
                                                          const float16 *O,
                                                          const float16 *G,
                                                          const float16 *S,
                                                          uint32_t n,
                                                          uint32_t own,
                                                          uint32_t rows,
                                                          uint32_t cols)
{
    p->shard_Y = (uintptr_t)Y;
    p->shard_O = (uintptr_t)O;
    p->shard_G = (uintptr_t)G;
    p->shard_S = (uintptr_t)S;
    p->dims    = rows | (cols << 16);
    p->slices  = n | (own << 16);
}

#endif /* ROWSLICE_SUM_DIV_FP16_SPATZ_H_ */
