import warp as wp
from . import math
from . import types

from . import broadphase


def collision(m: types.Model, d: types.Data):
  """Forward kinematics."""

  broadphase.collision2(m, d, 1e9, 12, 12, 12, 8, 1.0)

  print("Hello")