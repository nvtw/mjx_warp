# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import warp as wp

from .types import Model
from .types import Data
from .types import GeomType
from .types import NUM_GEOM_TYPES
from .math import make_frame
from .math import normalize_with_norm
from .support import mat33_from_cols


@wp.struct
class GeomPlane:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3


@wp.struct
class GeomSphere:
  pos: wp.vec3
  rot: wp.mat33
  radius: float


@wp.struct
class GeomCapsule:
  pos: wp.vec3
  rot: wp.mat33
  radius: float
  halfsize: float


@wp.struct
class GeomEllipsoid:
  pos: wp.vec3
  rot: wp.mat33
  size: wp.vec3


@wp.struct
class GeomCylinder:
  pos: wp.vec3
  rot: wp.mat33
  radius: float
  halfsize: float


@wp.struct
class GeomBox:
  pos: wp.vec3
  rot: wp.mat33
  size: wp.vec3


@wp.struct
class GeomConvex:
  pos: wp.vec3
  rot: wp.mat33
  vert_offset: int
  vert_count: int


def get_info(t):
  @wp.func
  def _get_info(
    gid: int,
    dataid: int,
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    size: wp.vec3,
    convex_vert_offset: wp.array(dtype=int),
  ):
    pos = geom_xpos[gid]
    rot = geom_xmat[gid]
    if wp.static(t == GeomType.SPHERE.value):
      sphere = GeomSphere()
      sphere.pos = pos
      sphere.rot = rot
      sphere.radius = size[0]
      return sphere
    elif wp.static(t == GeomType.BOX.value):
      box = GeomBox()
      box.pos = pos
      box.rot = rot
      box.size = size
      return box
    elif wp.static(t == GeomType.PLANE.value):
      plane = GeomPlane()
      plane.pos = pos
      plane.rot = rot
      plane.normal = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
      return plane
    elif wp.static(t == GeomType.CAPSULE.value):
      capsule = GeomCapsule()
      capsule.pos = pos
      capsule.rot = rot
      capsule.radius = size[0]
      capsule.halfsize = size[1]
      return capsule
    elif wp.static(t == GeomType.ELLIPSOID.value):
      ellipsoid = GeomEllipsoid()
      ellipsoid.pos = pos
      ellipsoid.rot = rot
      ellipsoid.size = size
      return ellipsoid
    elif wp.static(t == GeomType.CYLINDER.value):
      cylinder = GeomCylinder()
      cylinder.pos = pos
      cylinder.rot = rot
      cylinder.radius = size[0]
      cylinder.halfsize = size[1]
      return cylinder
    elif wp.static(t == GeomType.MESH.value):
      convex = GeomConvex()
      convex.pos = pos
      convex.rot = rot
      if convex_vert_offset and dataid >= 0:
        convex.vert_offset = convex_vert_offset[dataid]
        convex.vert_count = convex_vert_offset[dataid + 1] - convex.vert_offset
      else:
        convex.vert_offset = 0
        convex.vert_count = 0
      return convex
    else:
      wp.static(RuntimeError("Unsupported type", t))

  return _get_info


@wp.func
def _plane_sphere(
  plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float
):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(plane: GeomPlane, sphere: GeomSphere, worldid: int, d: Data):
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.radius)

  index = wp.atomic_add(d.ncon, 0, 1)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = make_frame(plane.normal)
  d.contact.worldid[index] = worldid


@wp.func
def sphere_sphere(sphere1: GeomSphere, sphere2: GeomSphere, worldid: int, d: Data):
  dir = sphere1.pos - sphere2.pos
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (sphere1.radius + sphere2.radius)
  pos = sphere1.pos + n * (sphere1.radius + 0.5 * dist)

  index = wp.atomic_add(d.ncon, 0, 1)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = make_frame(n)
  d.contact.worldid[index] = worldid


@wp.func
def plane_capsule(plane: GeomPlane, cap: GeomCapsule, worldid: int, d: Data):
  """Calculates two contacts between a capsule and a plane."""
  n = plane.normal
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  frame = mat33_from_cols(n, b, wp.cross(n, b))
  segment = axis * cap.halfsize

  index = wp.atomic_add(d.ncon, 0, 2)
  dist, pos = _plane_sphere(n, plane.pos, cap.pos + segment, cap.radius)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = frame
  d.contact.worldid[index] = worldid
  index += 1

  dist, pos = _plane_sphere(n, plane.pos, cap.pos - segment, cap.radius)
  d.contact.dist[index] = dist
  d.contact.pos[index] = pos
  d.contact.frame[index] = frame
  d.contact.worldid[index] = worldid


_collision_functions = {
  (GeomType.PLANE.value, GeomType.SPHERE.value): plane_sphere,
  (GeomType.SPHERE.value, GeomType.SPHERE.value): sphere_sphere,
  (GeomType.PLANE.value, GeomType.CAPSULE.value): plane_capsule,
}


def create_collision_function_kernel(type1, type2):
  key = group_key(type1, type2)

  @wp.kernel
  def _collision_function_kernel(
    m: Model,
    d: Data,
  ):
    tid = wp.tid()
    num_candidate_contacts = d.narrowphase_candidate_group_count[key]
    if tid >= num_candidate_contacts:
      return

    geoms = d.narrowphase_candidate_geom[key, tid]
    worldid = d.narrowphase_candidate_worldid[key, tid]

    geom1 = wp.static(get_info(type1))(
      geoms[0],
      worldid,
      d.geom_xpos[worldid],
      d.geom_xmat[worldid],
      m.geom_size[worldid],
      m.convex_vert_offset,
    )
    geom2 = wp.static(get_info(type2))(
      geoms[1],
      worldid,
      d.geom_xpos[worldid],
      d.geom_xmat[worldid],
      m.geom_size[worldid],
      m.convex_vert_offset,
    )

    wp.static(_collision_functions[(type1, type2)])(geom1, geom2, worldid, d)

  return _collision_function_kernel


@wp.kernel
def plane_convex_kernel(m: Model, d: Data, group_key: int):
  """Calculates contacts between a plane and a convex object."""
  tid = wp.tid()
  num_candidate_contacts = d.narrowphase_candidate_group_count[group_key]
  if tid >= num_candidate_contacts:
    return

  geoms = d.narrowphase_candidate_geom[group_key, tid]
  worldid = d.narrowphase_candidate_worldid[group_key, tid]

  # plane is always first, convex could be box/mesh.
  plane_geom = geoms[0]
  convex_geom = geoms[1]

  convex_type = m.geom_type[convex_geom]
  # if convex_type == wp.static(GeomType.BOX.value):
  #  pass # box-specific stuff - many things can be hardcoded here
  # else:
  #  pass # mesh-specific stuff

  # if contact
  index = wp.atomic_add(d.ncon, 0, 1)
  # d.contact.dist[index] = dist
  # d.contact.pos[index] = pos
  # d.contact.frame[index] = frame
  # d.contact.worldid[index] = worldid


def plane_capsule(m: Model, d: Data, group_key: int):
  pass


def plane_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def plane_cylinder(m: Model, d: Data, group_key: int):
  pass


def plane_convex(m: Model, d: Data, group_key: int):
  wp.launch(
    kernel=plane_convex_kernel,
    dim=(d.nconmax),
    inputs=[m, d, group_key],
  )


def hfield_sphere(m: Model, d: Data, group_key: int):
  pass


def hfield_capsule(m: Model, d: Data, group_key: int):
  pass


def hfield_convex(m: Model, d: Data, group_key: int):
  pass


def sphere_sphere(m: Model, d: Data, group_key: int):
  pass


def sphere_capsule(m: Model, d: Data, group_key: int):
  pass


def sphere_cylinder(m: Model, d: Data, group_key: int):
  pass


def sphere_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def sphere_convex(m: Model, d: Data, group_key: int):
  pass


def capsule_capsule(m: Model, d: Data, group_key: int):
  pass


def capsule_convex(m: Model, d: Data, group_key: int):
  pass


def capsule_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def capsule_cylinder(m: Model, d: Data, group_key: int):
  pass


def ellipsoid_ellipsoid(m: Model, d: Data, group_key: int):
  pass


def ellipsoid_cylinder(m: Model, d: Data, group_key: int):
  pass


def cylinder_cylinder(m: Model, d: Data, group_key: int):
  pass


def box_box(m: Model, d: Data, group_key: int):
  pass


def convex_convex(m: Model, d: Data, group_key: int):
  pass


@wp.func
def group_key(type1: wp.int32, type2: wp.int32) -> wp.int32:
  return type1 + type2 * NUM_GEOM_TYPES


# same order as in MJX - collision function and group key.
# _COLLISION_FUNCS = [
#   (plane_sphere, group_key(GeomType.PLANE.value, GeomType.SPHERE.value)),
#   (plane_capsule, group_key(GeomType.PLANE.value, GeomType.CAPSULE.value)),
#   (plane_convex, group_key(GeomType.PLANE.value, GeomType.BOX.value)),
#   (plane_ellipsoid, group_key(GeomType.PLANE.value, GeomType.ELLIPSOID.value)),
#   (plane_cylinder, group_key(GeomType.PLANE.value, GeomType.CYLINDER.value)),
#   (plane_convex, group_key(GeomType.PLANE.value, GeomType.MESH.value)),
#   (hfield_sphere, group_key(GeomType.HFIELD.value, GeomType.SPHERE.value)),
#   (hfield_capsule, group_key(GeomType.HFIELD.value, GeomType.CAPSULE.value)),
#   (hfield_convex, group_key(GeomType.HFIELD.value, GeomType.BOX.value)),
#   (hfield_convex, group_key(GeomType.HFIELD.value, GeomType.MESH.value)),
#   (sphere_sphere, group_key(GeomType.SPHERE.value, GeomType.SPHERE.value)),
#   (sphere_capsule, group_key(GeomType.SPHERE.value, GeomType.CAPSULE.value)),
#   (sphere_cylinder, group_key(GeomType.SPHERE.value, GeomType.CYLINDER.value)),
#   (sphere_ellipsoid, group_key(GeomType.SPHERE.value, GeomType.ELLIPSOID.value)),
#   (sphere_convex, group_key(GeomType.SPHERE.value, GeomType.BOX.value)),
#   (sphere_convex, group_key(GeomType.SPHERE.value, GeomType.MESH.value)),
#   (capsule_capsule, group_key(GeomType.CAPSULE.value, GeomType.CAPSULE.value)),
#   (capsule_convex, group_key(GeomType.CAPSULE.value, GeomType.BOX.value)),
#   (capsule_ellipsoid, group_key(GeomType.CAPSULE.value, GeomType.ELLIPSOID.value)),
#   (capsule_cylinder, group_key(GeomType.CAPSULE.value, GeomType.CYLINDER.value)),
#   (capsule_convex, group_key(GeomType.CAPSULE.value, GeomType.MESH.value)),
#   (ellipsoid_ellipsoid, group_key(GeomType.ELLIPSOID.value, GeomType.ELLIPSOID.value)),
#   (ellipsoid_cylinder, group_key(GeomType.ELLIPSOID.value, GeomType.CYLINDER.value)),
#   (cylinder_cylinder, group_key(GeomType.CYLINDER.value, GeomType.CYLINDER.value)),
#   (box_box, group_key(GeomType.BOX.value, GeomType.BOX.value)),
#   (convex_convex, group_key(GeomType.BOX.value, GeomType.MESH.value)),
#   (convex_convex, group_key(GeomType.MESH.value, GeomType.MESH.value)),
# ]


_collision_kernels = {}

def narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.

  # we run the collision functions in increasing condim order to get the grouping
  # right from the get-go.

  # TODO only generate collision kernels we actually need
  if len(_collision_kernels) == 0:
    for type1, type2 in _collision_functions.keys():
      _collision_kernels[(type1, type2)] = create_collision_function_kernel(
        type1, type2
      )

  # for i in range(len(_COLLISION_FUNCS)):
  #   # this will lead to a bunch of unnecessary launches, but we don't want to sync at this point
  #   func, group_key = _COLLISION_FUNCS[i]
  #   func(m, d, group_key)

  for collision_kernel in _collision_kernels.values():
    wp.launch(collision_kernel, dim=d.nconmax, inputs=[m, d])
