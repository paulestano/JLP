from jlp import float_quantize_stochastic_cuda, float_quantize_nearest_cuda
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax import random
from jax.config import config

def build_compile_quantize(func, man_bits, exp_bits, subnormals, saturate):
    compiled=lambda x:func(x,man_bits, exp_bits, subnormals, saturate)
    return jax.jit(compiled)

key = random.PRNGKey(0)
size=3
x = random.normal(key, (size, size), dtype=jnp.float32)
print(x)
quantize_nearest = build_compile_quantize(float_quantize_nearest_cuda, 2,3,True,True)
quantize_stochastic = build_compile_quantize(float_quantize_stochastic_cuda, 2,3,True,True)
qx = quantize_nearest(x)
qx2 = quantize_stochastic(x)
print(qx)
print(qx2)

