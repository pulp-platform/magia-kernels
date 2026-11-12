// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GEMM_FP16_SPATZ_H_
#define GEMM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "gemm_fp16_spatz_params.h"

/*
 * Y = alpha * A' @ B' + beta * C (A', B' optionally transposed), in L2, with the larger output
 * dimension split over the tiles.
 */
void MAGIA_gemm_fp16_spatz(const float16 *A,
                           const float16 *B,
                           const float16 *C,
                           float16 alpha,
                           float16 beta,
                           int transA,
                           int transB,
                           uint32_t A_shape[2],
                           uint32_t B_shape[2],
                           uint32_t C_shape[2],
                           uint32_t Y_shape[2],
                           float16 *Y);

/* Same operation on this tile only, with A, B, C and Y already in this tile's L1. */
void MAGIA_gemm_fp16_spatz_l1(const float16 *A,
                              const float16 *B,
                              const float16 *C,
                              float16 alpha,
                              float16 beta,
                              int transA,
                              int transB,
                              uint32_t A_shape[2],
                              uint32_t B_shape[2],
                              uint32_t C_shape[2],
                              uint32_t Y_shape[2],
                              float16 *Y);

/*
 * As MAGIA_gemm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_gemm_fp16_spatz_l1_start(gemm_fp16_spatz_params_t *p,
                                    const float16 *A,
                                    const float16 *B,
                                    const float16 *C,
                                    float16 alpha,
                                    float16 beta,
                                    int transA,
                                    int transB,
                                    uint32_t A_shape[2],
                                    uint32_t B_shape[2],
                                    uint32_t C_shape[2],
                                    uint32_t Y_shape[2],
                                    float16 *Y,
                                    mg_event_t *event);

/*
 * Fill the parameter block `p` for MAGIA_gemm_fp16_spatz_l1_start; alpha and beta are kept in
 * `p`.
 */
static inline void gemm_fp16_spatz_params_set(gemm_fp16_spatz_params_t *p,
                                              const float16 *A,
                                              const float16 *B,
                                              const float16 *C,
                                              float16 alpha,
                                              float16 beta,
                                              int transA,
                                              int transB,
                                              uint32_t A_shape[2],
                                              uint32_t B_shape[2],
                                              uint32_t C_shape[2],
                                              uint32_t Y_shape[2],
                                              float16 *Y)
{
    (void)B_shape;
    (void)C_shape;
    p->alpha_l1  = alpha;
    p->beta_l1   = beta;
    p->shard_A   = (uintptr_t)A;
    p->shard_B   = (uintptr_t)B;
    p->shard_C   = (uintptr_t)C;
    p->shard_Y   = (uintptr_t)Y;
    p->alpha     = (uintptr_t)&p->alpha_l1;
    p->beta      = (uintptr_t)&p->beta_l1;
    p->transA    = transA;
    p->transB    = transB;
    p->shard_dim = 0;
    p->m_start   = 0;
    p->m_len     = Y_shape[0];
    p->M         = Y_shape[0];
    p->N         = Y_shape[1];
    p->K         = (!transA) ? A_shape[1] : A_shape[0];
}

#endif /* GEMM_FP16_SPATZ_H_ */
