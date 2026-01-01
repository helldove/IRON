// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "lut_based_ops.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

 
#define SM_VEC_LEN 16   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;


static v16bfloat16 CalcExpBf16(v16bfloat16 x) {
  return to_v16bfloat16(getExpBf16Lut(x));
}

static bfloat16 GetInvBf16(float x) {
  return getInvBf16Lut(x);
}

void softmax_simple_bf16(bfloat16 *restrict input_vector, bfloat16 *restrict output_vector, const int32_t vector_size)
{
    event0();

    int num_elems = vector_size;
    float accum_exp_val;
    auto it_exp_in = aie::cbegin_vector<16>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_vector<16>((bfloat16 *)output_vector);
    auto it_scale = aie::cbegin_restrict_vector<16>((bfloat16 *)output_vector);
    auto it_soft_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

    bfloat16 col_sum_inv;
    aie::vector<bfloat16, 16> in_elems, va;
    aie::accum<accfloat, 16> out_vals;
    int col_iters = num_elems >> 4;
    accum_exp_val = 0;

    /////////////////////
    //// Compute exp ////
    /////////////////////
    aie::vector<bfloat16, 16> exp_val;
    aie::vector<float, 16> input_fp32;

    const int elem_iters = num_elems / 16;
    aie::vector<bfloat16, 16> input_bf16;
    aie::accum<accfloat, 16> exp_val_accum;
    exp_val_accum = aie::zeros<accfloat, 16>();
    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_exp_in++;
        // exp_val = to_v16bfloat16(getExpBf16(input_bf16));
        exp_val = CalcExpBf16(input_bf16);
        exp_val_accum = add(exp_val_accum, exp_val);
        *it_exp_out++ = exp_val;
    }
    aie::vector<float, 16> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);
    /////////////////////

    col_sum_inv = (bfloat16)aie::inv(accum_exp_val);
    for (int c = 0; c < col_iters; c++) {
        in_elems = *it_scale++;
        out_vals = aie::mul(in_elems, col_sum_inv);
        *it_soft_out++ = out_vals.to_vector<bfloat16>();
    }

    event1();

    return;
}

void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                bfloat16 *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx,
                                const int32_t num_rows,
                                const bfloat16 scale)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // VJUNG: We do 3 passes on the vector:
    // 1. Find the max value scaled by log2e in the vector
    // 2. Calculate the exponentials of the scaled values minus the maximum
    // 3. Calculate the softmax by dividing each exponential by the sum of all exponentials
    // Note: The multiplication by log2e is very sensitive, casting it to bf16 before exponentiation leads to wrong
    // output.

    auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
    aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

    float max_val = 0;
    float accum_exp_val = 0;
    float running_max = 0;
    float col_sum_inv;
    const int elem_iters = vector_size / SM_VEC_LEN;

    exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

    log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)scale);

    // First pass
    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_log_in++;
        scaled_accum = aie::mul(input_bf16, log2e_vec);
        running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (running_max > max_val) {
            max_val = running_max;
        }
    }

    // Compute m_{i}
    if (max_val > scale_buffer[row_idx]) {
        scale_buffer[num_rows + row_idx] = max_val;
    } else {
        scale_buffer[num_rows + row_idx] = scale_buffer[row_idx];
        max_val = scale_buffer[row_idx];
    }

    max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

    // Second pass
    for (int i = 0; i < elem_iters; i++) {

        input_bf16 = *it_exp_in++;

        scaled_accum = aie::mul(input_bf16, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        // exp_val = to_v16bfloat16(getExpBf16(exp_in_accum.to_vector<bfloat16>()));
        exp_val = CalcExpBf16(exp_in_accum.to_vector<bfloat16>());
        exp_val_accum = add(exp_val_accum, exp_val);

        *it_exp_out++ = exp_val;
    }

    aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);

    scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

    event1();

    return;
}
extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output, const int32_t input_size)
{
    softmax_simple_bf16(input, output, input_size);
}

void partial_softmax_bf16(bfloat16 *restrict input,
                          bfloat16 *restrict output,
                          bfloat16 *restrict scale_buffer,
                          const int32_t input_size,
                          const int32_t row_idx,
                          const int32_t num_rows,
                          const bfloat16 scale)
{
    partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx, num_rows, scale);
}
} // extern "C"
