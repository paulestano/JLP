// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kepler.h"
#include "kernel_helpers.h"
#include "kernels.h"
#include "bit_helper.cu"
#include <stdio.h>

namespace kepler_jax {

namespace {

template <typename T>
__global__ void kepler_kernel(std::int64_t size, const T *mean_anom, const T *ecc, T *sin_ecc_anom,
                              T *cos_ecc_anom) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    compute_eccentric_anomaly<T>(mean_anom[idx], ecc[idx], sin_ecc_anom + idx, cos_ecc_anom + idx);
  }
}

__device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                 bool subnormal_support = true, bool saturate = true) {
  unsigned int target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23);

  if (noquantize) {
    quantized = origin_float;
  } else {
    if (subnormal && subnormal_support) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val = origin_float + shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_fp_stochastic(float origin_float, unsigned int rand_prob,
                                    int man_bits, int exp_bits,
                                    bool subnormal_support = true, bool saturate = true) {
  unsigned int target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);

  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  } else {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (unsigned int)r[index], man_bits,
                                  exp_bits, subnormal_support, saturate);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(const float * a, float *o, int size,
                                     int man_bits, int exp_bits,
                                     bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_nearest(a[index], man_bits, exp_bits, subnormal_support, saturate);
}

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T>
inline void apply_kepler(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  const KeplerDescriptor &d = *UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const T *mean_anom = reinterpret_cast<const T *>(buffers[0]);
  const T *ecc = reinterpret_cast<const T *>(buffers[1]);
  T *sin_ecc_anom = reinterpret_cast<T *>(buffers[2]);
  T *cos_ecc_anom = reinterpret_cast<T *>(buffers[3]);

  const int block_dim = 128;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);
  kepler_kernel<T>
      <<<grid_dim, block_dim, 0, stream>>>(size, mean_anom, ecc, sin_ecc_anom, cos_ecc_anom);

  ThrowIfError(cudaGetLastError());
}
//Tensor a, int man_bits, int exp_bits, bool subnormals, bool saturate
inline void apply_float_quantize_nearest_cuda(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
// use external random number right now
  const KeplerDescriptor &d = *UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const float *a =  reinterpret_cast<const float *>(buffers[0]);
  const std::int64_t man_bits =  *reinterpret_cast<const std::int64_t *>(buffers[1]);
  const std::int64_t exp_bits =  *reinterpret_cast<const std::int64_t *>(buffers[2]);
  const bool subnormals = *reinterpret_cast<const bool *>(buffers[3]);
  const bool saturate = *reinterpret_cast<const bool *>(buffers[4]);
  float *o = reinterpret_cast<float *>(buffers[5]);

  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(
      a, o, size, man_bits, exp_bits, 
      subnormals, saturate);
  ThrowIfError(cudaGetLastError()); 
}

}  // namespace

void gpu_kepler_f32(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_kepler<float>(stream, buffers, opaque, opaque_len);
}

void gpu_kepler_f64(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  apply_kepler<double>(stream, buffers, opaque, opaque_len);
}

 void float_quantize_nearest_cuda(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len) {
  printf("my kernel");
  apply_float_quantize_nearest_cuda( stream, buffers, opaque, opaque_len);
}

}  // namespace kepler_jax
