// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include <curand.h>
#include "bit_helper.cu"
#include "kernel_helpers.h"
#include "kernels.h"

namespace jlp {

namespace {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void GPU_fill_rand(float *A, int size) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

  curandGenerateUniform(prng, A, size);
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

__device__ float cast_fp_stochastic(float origin_float, unsigned int rand_prob, int man_bits,
                                    int exp_bits, bool subnormal_support = true,
                                    bool saturate = true) {
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
__global__ void float_kernel_stochastic(const float *__restrict__ a, int *__restrict__ r, float *o,
                                        const int size, const std::int64_t *format_,
                                        const bool *features_) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (unsigned int)r[index], format_[0], format_[1],
                                  format_[0], format_[1]);
}

inline void apply_float_quantize_stochastic_cuda(cudaStream_t stream, void **buffers, const char *opaque,
                                           std::size_t opaque_len) {
  // use external random number right now
  // use external random number right now
  const JlpDescriptor &d = *UnpackDescriptor<JlpDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const float *a = reinterpret_cast<const float *>(buffers[0]);
  const std::int64_t *format_ = reinterpret_cast<const std::int64_t *>(buffers[1]);
  const bool *features_ = reinterpret_cast<const bool *>(buffers[2]);

  float *o = reinterpret_cast<float *>(buffers[3]);
  int *rand_ints;
  cudaMallocManaged(&rand_ints, sizeof(int) * size);

  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize, 0, stream>>>(a, rand_ints, o, size, format_,
                                                               features_);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(const float *a, float *o, int size,
                                     const std::int64_t *format_, const bool *features_) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_nearest(a[index], format_[0], format_[1], features_[0], features_[1]);
}

inline void apply_float_quantize_nearest_cuda(cudaStream_t stream, void **buffers,
                                              const char *opaque, std::size_t opaque_len) {
  // use external random number right now
  const JlpDescriptor &d = *UnpackDescriptor<JlpDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  const float *a = reinterpret_cast<const float *>(buffers[0]);
  const std::int64_t *format_ = reinterpret_cast<const std::int64_t *>(buffers[1]);
  const bool *features_ = reinterpret_cast<const bool *>(buffers[2]);

  float *o = reinterpret_cast<float *>(buffers[3]);

  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize, 0, stream>>>(a, o, size, format_, features_);
  ThrowIfError(cudaGetLastError());
}

}  // namespace

void float_quantize_nearest_cuda(cudaStream_t stream, void **buffers, const char *opaque,
                                 std::size_t opaque_len) {
  apply_float_quantize_nearest_cuda(stream, buffers, opaque, opaque_len);
}

void float_quantize_stochastic_cuda(cudaStream_t stream, void **buffers, const char *opaque,
                                 std::size_t opaque_len) {
  apply_float_quantize_stochastic_cuda(stream, buffers, opaque, opaque_len);
}


}  // namespace jlp
