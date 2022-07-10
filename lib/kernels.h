#ifndef _JLP_KERNELS_H_
#define _JLP_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace jlp {
struct JlpDescriptor {
  std::int64_t size;
};

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Stochastic Rounding.
 **/
// void float_quantize_stochastic_cuda(cudaStream_t stream, void** buffers, const char* opaque,
//                     std::size_t opaque_len);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Nearest Rounding.
 **/
void float_quantize_nearest_cuda(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

__global__ void float_kernel_nearest(const float * a, float *o, int size,
                                     int man_bits, int exp_bits,
                                     bool subnormal_support, bool saturate);

}  // namespace jlp

#endif