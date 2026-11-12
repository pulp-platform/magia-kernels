// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RMSNORM_FP16_SPATZ_H_
#define RMSNORM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "rmsnorm_fp16_spatz_params.h"

/*
 * Y = X / sqrt(mean(X^2) + epsilon) * scale over the last axis of X [rows x w_len] (ONNX
 * RMSNormalization, axis = -1): the rows are split over all tiles.
 */
void MAGIA_rmsnorm_fp16_spatz(const float16 *X,
                              const float16 *scale,
                              const float16 epsilon,
                              uint32_t rows,
                              uint32_t w_len,
                              float16 *Y);

/*
 * Same operation on this tile only, with X, scale and Y already in this tile's L1. Y may alias X.
 */
void MAGIA_rmsnorm_fp16_spatz_l1(const float16 *X,
                                 const float16 *scale,
                                 const float16 *epsilon,
                                 uint32_t rows,
                                 uint32_t w_len,
                                 float16 *Y);

/*
 * As MAGIA_rmsnorm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_rmsnorm_fp16_spatz_l1_start(rmsnorm_fp16_spatz_params_t *p,
                                       const float16 *X,
                                       const float16 *scale,
                                       const float16 *epsilon,
                                       uint32_t rows,
                                       uint32_t w_len,
                                       float16 *Y,
                                       mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_rmsnorm_fp16_spatz_l1_start. */
static inline void rmsnorm_fp16_spatz_params_set(rmsnorm_fp16_spatz_params_t *p,
                                                 const float16 *X,
                                                 const float16 *scale,
                                                 const float16 *epsilon,
                                                 uint32_t rows,
                                                 uint32_t w_len,
                                                 float16 *Y)
{
    p->shard_X = (uintptr_t)X;
    p->shard_Y = (uintptr_t)Y;
    p->gamma   = (uintptr_t)scale;
    p->eps     = (uintptr_t)epsilon;
    p->r_start = 0;
    p->r_len   = rows;
    p->w_len   = w_len;
}

#endif /* RMSNORM_FP16_SPATZ_H_ */
