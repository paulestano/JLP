# -*- coding: utf-8 -*-

__all__ = ["float_quantize_nearest_cuda"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops


def float_quantize_nearest_cuda(a, man_bits, exp_bits, subnormals, saturate):
    res = _float_quantize_nearest_cuda_prim.bind(
        a,
        np.array([man_bits, exp_bits], dtype=np.int64),
        np.array([subnormals, saturate], dtype=bool),
    )
    return res


def _float_quantize_nearest_cuda_abstract(a, format, features):
    shape = a.shape
    dtype = dtypes.canonicalize_dtype(a.dtype)
    return ShapedArray(shape, dtype)


def _float_quantize_nearest_cuda_translation(c, a, format, features, *, platform="gpu"):
    assert platform == "gpu"
    # The inputs have "shapes" that provide both the shape and the dtype
    a_shape = c.get_shape(a)
    format_shape = c.get_shape(format)
    feature_shape = c.get_shape(features)

    # Extract the dtype and shape
    dtype = a_shape.element_type()
    dims = a_shape.dimensions()

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # The inputs and outputs all have the same shape so let's predefine this
    # specification
    shape = xla_client.Shape.array_shape(
        np.dtype(dtype), dims, tuple(range(len(dims) - 1, -1, -1))
    )

    scalar_shape = xla_client.Shape.array_shape(
        np.dtype(format_shape.element_type()),
        format_shape.dimensions(),
        tuple(range(len(format_shape.dimensions()) - 1, -1, -1)),
    )
    bool_shape = xla_client.Shape.array_shape(
        np.dtype(feature_shape.element_type()),
        feature_shape.dimensions(),
        tuple(range(len(feature_shape.dimensions()) - 1, -1, -1)),
    )

    op_name = "float_quantize_nearest_cuda".encode()
    if gpu_ops is None:
        raise ValueError("The 'jlp' module was not compiled with CUDA support")

    # On the GPU, we do things a little differently and encapsulate the
    # dimension using the 'opaque' parameter
    opaque = gpu_ops.build_jlp_descriptor(size)
    return xops.CustomCallWithLayout(
        c,
        op_name,
        operands=(a, format, features),
        operand_shapes_with_layout=(shape, scalar_shape, bool_shape),
        shape_with_layout=shape,
        opaque=opaque,
    )


_float_quantize_nearest_cuda_prim = core.Primitive("float_quantize_nearest_cuda")
_float_quantize_nearest_cuda_prim.def_impl(
    partial(xla.apply_primitive, _float_quantize_nearest_cuda_prim)
)
_float_quantize_nearest_cuda_prim.def_abstract_eval(
    _float_quantize_nearest_cuda_abstract
)

xla.backend_specific_translations["gpu"][_float_quantize_nearest_cuda_prim] = partial(
    _float_quantize_nearest_cuda_translation, platform="gpu"
)
