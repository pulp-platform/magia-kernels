// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RESIZE_FP16_SPATZ_H_
#define RESIZE_FP16_SPATZ_H_

#include <stdint.h>
#include "mg_event.h"
#include "resize_fp16_spatz_params.h"

void MAGIA_resize_fp16_spatz(const float16 *X,
                             float16 *Y,
                             uint32_t batch_size,
                             uint32_t channels,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t out_h,
                             uint32_t out_w);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_resize_fp16_spatz_l1(const float16 *X,
                                float16 *Y,
                                uint32_t batch_size,
                                uint32_t channels,
                                uint32_t in_h,
                                uint32_t in_w,
                                uint32_t out_h,
                                uint32_t out_w);

/*
 * As MAGIA_resize_fp16_spatz_l1, but only starts the Spatz task, with the parameter block `p`
 * (which must stay valid until it completes): mg_spatz_wait(mode, event) waits for it.
 */
void MAGIA_resize_fp16_spatz_l1_start(resize_fp16_spatz_params_t *p,
                                      const float16 *X,
                                      float16 *Y,
                                      uint32_t batch_size,
                                      uint32_t channels,
                                      uint32_t in_h,
                                      uint32_t in_w,
                                      uint32_t out_h,
                                      uint32_t out_w,
                                      mg_event_t *event);

/* Fill the parameter block `p` for MAGIA_resize_fp16_spatz_l1_start. */
static inline void resize_fp16_spatz_params_set(resize_fp16_spatz_params_t *p,
                                                const float16 *X,
                                                float16 *Y,
                                                uint32_t batch_size,
                                                uint32_t channels,
                                                uint32_t in_h,
                                                uint32_t in_w,
                                                uint32_t out_h,
                                                uint32_t out_w)
{
    p->shard_X         = (uintptr_t)X;
    p->shard_Y         = (uintptr_t)Y;
    p->in_h            = in_h;
    p->in_w            = in_w;
    p->out_h           = out_h;
    p->out_w           = out_w;
    p->iteration_start = 0;
    p->iteration_len   = batch_size * channels;
}

#endif /* RESIZE_FP16_SPATZ_H_ */
