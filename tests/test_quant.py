from jlp import float_quantize_stochastic_cuda, float_quantize_nearest_cuda
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.config import config

config.update("jax_enable_x64", True)

key = random.PRNGKey(0)
size=3
x = random.normal(key, (size, size), dtype=jnp.float32)
qx = float_quantize_nearest_cuda(x, 8,7,True, True)
qx2 = float_quantize_stochastic_cuda(x, 8,7,True, True)
