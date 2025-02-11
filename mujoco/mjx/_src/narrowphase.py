import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from absl.testing import absltest
from jax import numpy as jp
from . import types

import warp as wp

# wp.set_device("cpu")

# wp.config.verify_cuda = True
# wp.config.verify_fp = True
# wp.clear_kernel_cache()


wp.config.enable_backward = False
wp.set_module_options(
    {
        "enable_backward": False,
        "max_unroll": 1,
    }
)

mjxGEOM_PLANE = 0
mjxGEOM_HFIELD = 1
mjxGEOM_SPHERE = 2
mjxGEOM_CAPSULE = 3
mjxGEOM_ELLIPSOID = 4
mjxGEOM_CYLINDER = 5
mjxGEOM_BOX = 6
mjxGEOM_CONVEX = 7
mjxGEOM_size = 8

mjMINVAL = 1e-15

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30

kGjkMultiContactCount = 4
kMaxEpaBestCount = 12
kMaxMultiPolygonCount = 8


@wp.struct
class GeomType_PLANE:
    pos: wp.vec3
    rot: wp.mat33


@wp.struct
class GeomType_SPHERE:
    pos: wp.vec3
    rot: wp.mat33
    radius: float


@wp.struct
class GeomType_CAPSULE:
    pos: wp.vec3
    rot: wp.mat33
    radius: float
    halfsize: float


@wp.struct
class GeomType_ELLIPSOID:
    pos: wp.vec3
    rot: wp.mat33
    size: wp.vec3


@wp.struct
class GeomType_CYLINDER:
    pos: wp.vec3
    rot: wp.mat33
    radius: float
    halfsize: float


@wp.struct
class GeomType_BOX:
    pos: wp.vec3
    rot: wp.mat33
    size: wp.vec3


@wp.struct
class GeomType_CONVEX:
    pos: wp.vec3
    rot: wp.mat33
    vert_offset: int
    vert_count: int