#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace jlp;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["float_quantize_nearest_cuda"] = EncapsulateFunction(float_quantize_nearest_cuda);
  dict["float_quantize_stochastic_cuda"] = EncapsulateFunction(float_quantize_stochastic_cuda);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("registrations", &Registrations);
  m.def("build_jlp_descriptor",
        [](std::int64_t size) { return PackDescriptor(JlpDescriptor{size}); });
}
}  // namespace
