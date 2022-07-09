from kepler_jax import float_quantize_nearest_cuda
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.config import config

config.update("jax_enable_x64", True)

key = random.PRNGKey(0)
size=3
print("generating number")
x = random.normal(key, (size, size), dtype=jnp.float32)
print(x)
qx = float_quantize_nearest_cuda(x, 1,2,False, False)
print(abs(x-qx)/x)