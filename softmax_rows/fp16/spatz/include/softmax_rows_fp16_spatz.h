// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SOFTMAX_ROWS_FP16_SPATZ_H_
#define SOFTMAX_ROWS_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "softmax_rows_fp16_spatz_params.h"

/*
 * X[r, :] = softmax(s * X[r, :]) in place, for a rows x cols matrix X (cols even), with the fast
 * exp (exact scaling for s a power of 2). The rows are split over all tiles.
 */
void MAGIA_softmax_rows_fp16_spatz(float16 *X, float16 s, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X already in this tile's L1. */
void MAGIA_softmax_rows_fp16_spatz_l1(float16 *X, float16 s, uint32_t rows, uint32_t cols);

/* fp16 bits of the exp clamp MIN / s | COEF * s << 16, for the scale s. */
static inline uint32_t softmax_rows_fp16_spatz_min_coef(float16 s)
{
    union {
        float16 f[2];
        uint32_t u;
    } mc;

    mc.f[0] = (float16)(SOFTMAX_ROWS_EXP_MIN / (float)s);
    mc.f[1] = (float16)(SOFTMAX_ROWS_EXP_COEF * (float)s);

    return mc.u;
}

/*
 * As MAGIA_softmax_rows_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_softmax_rows_fp16_spatz_l1_start(softmax_rows_fp16_spatz_params_t *p,
                                            float16 *X,
                                            float16 s,
                                            uint32_t rows,
                                            uint32_t cols,
                                            mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_softmax_rows_fp16_spatz_l1_start. */
static inline void softmax_rows_fp16_spatz_params_set(softmax_rows_fp16_spatz_params_t *p,
                                                      float16 *X,
                                                      float16 s,
                                                      uint32_t rows,
                                                      uint32_t cols)
{
    p->shard_X  = (uintptr_t)X;
    p->dims     = rows | (cols << 16);
    p->min_coef = softmax_rows_fp16_spatz_min_coef(s);
}

#endif /* SOFTMAX_ROWS_FP16_SPATZ_H_ */
