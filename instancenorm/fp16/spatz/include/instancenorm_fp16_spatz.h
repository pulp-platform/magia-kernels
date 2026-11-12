// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef INSTANCENORM_FP16_SPATZ_H_
#define INSTANCENORM_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "instancenorm_fp16_spatz_params.h"

void MAGIA_instancenorm_fp16_spatz(const float16 *input,
                                   float16 *output,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 epsilon,
                                   uint32_t input_shape[4]);
/* Same operation on this tile only, with input, output, scale and B already in this tile's L1. */
void MAGIA_instancenorm_fp16_spatz_l1(const float16 *input,
                                      float16 *output,
                                      const float16 *scale,
                                      const float16 *B,
                                      const float16 epsilon,
                                      uint32_t input_shape[4]);

/*
 * As MAGIA_instancenorm_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_instancenorm_fp16_spatz_l1_start(instancenorm_fp16_spatz_params_t *p,
                                            const float16 *input,
                                            float16 *output,
                                            const float16 *scale,
                                            const float16 *B,
                                            const float16 epsilon,
                                            uint32_t input_shape[4],
                                            mg_event_t *event);

/*
 * Fill the parameter block `p` for MAGIA_instancenorm_fp16_spatz_l1_start; epsilon is kept in `p`.
 */
static inline void instancenorm_fp16_spatz_params_set(instancenorm_fp16_spatz_params_t *p,
                                                      const float16 *input,
                                                      float16 *output,
                                                      const float16 *scale,
                                                      const float16 *B,
                                                      const float16 epsilon,
                                                      uint32_t input_shape[4])
{
    p->eps_l1       = epsilon;
    p->shard_input  = (uintptr_t)input;
    p->shard_output = (uintptr_t)output;
    p->gamma        = (uintptr_t)scale;
    p->beta         = (uintptr_t)B;
    p->eps          = (uintptr_t)&p->eps_l1;
    p->inst_start   = 0;
    p->inst_len     = input_shape[0] * input_shape[1];
    p->hw_len       = input_shape[2] * input_shape[3];
    p->num_channels = input_shape[1];
}

#endif /* INSTANCENORM_FP16_SPATZ_H_ */
