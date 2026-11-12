// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GLOBALAVERAGEPOOL_FP16_SPATZ_H_
#define GLOBALAVERAGEPOOL_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "globalaveragepool_fp16_spatz_params.h"

/* GlobalAveragePool of the NCHW tensor X into Y, in L2, with the channels split over the tiles. */
void MAGIA_globalaveragepool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4]);

/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_globalaveragepool_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t input_shape[4]);

/*
 * As MAGIA_globalaveragepool_fp16_spatz_l1, but only starts the Spatz task, with the parameter
 * block `p` (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_globalaveragepool_fp16_spatz_l1_start(globalaveragepool_fp16_spatz_params_t *p,
                                                 const float16 *X,
                                                 float16 *Y,
                                                 uint32_t input_shape[4],
                                                 mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_globalaveragepool_fp16_spatz_l1_start. */
static inline void globalaveragepool_fp16_spatz_params_set(globalaveragepool_fp16_spatz_params_t *p,
                                                           const float16 *X,
                                                           float16 *Y,
                                                           uint32_t input_shape[4])
{
    p->shard_X = (uintptr_t)X;
    p->shard_Y = (uintptr_t)Y;
    p->start   = 0;
    p->len     = input_shape[0] * input_shape[1];
    p->hw_len  = input_shape[2] * input_shape[3];
}

#endif /* GLOBALAVERAGEPOOL_FP16_SPATZ_H_ */
