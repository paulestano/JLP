#ifndef _KEPLER_JAX_KERNELS_H_
#define _KEPLER_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kepler_jax {
struct KeplerDescriptor {
  std::int64_t size;
};

void gpu_kepler_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_kepler_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
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

}  // namespace kepler_jax

#endif