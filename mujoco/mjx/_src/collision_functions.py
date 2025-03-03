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
from .types import Contact
from .types import GeomType
from .types import NUM_GEOM_TYPES


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


def plane_sphere(m: Model, d: Data, group_key: int):
  pass


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


def capsule_convex(m: Model, d: Data, group_key: int):
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


def convex_convex(m: Model, d: Data, group_key: int):
  pass


@wp.func
def group_key(type1: wp.int32, type2: wp.int32) -> wp.int32:
  return type1 + type2 * NUM_GEOM_TYPES


# same order as in MJX - collision function and group key.
_COLLISION_FUNCS = [
  (plane_sphere, group_key(GeomType.PLANE.value, GeomType.SPHERE.value)),
  (plane_capsule, group_key(GeomType.PLANE.value, GeomType.CAPSULE.value)),
  (plane_convex, group_key(GeomType.PLANE.value, GeomType.BOX.value)),
  (plane_ellipsoid, group_key(GeomType.PLANE.value, GeomType.ELLIPSOID.value)),
  (plane_cylinder, group_key(GeomType.PLANE.value, GeomType.CYLINDER.value)),
  (plane_convex, group_key(GeomType.PLANE.value, GeomType.MESH.value)),
  (hfield_sphere, group_key(GeomType.HFIELD.value, GeomType.SPHERE.value)),
  (hfield_capsule, group_key(GeomType.HFIELD.value, GeomType.CAPSULE.value)),
  (hfield_convex, group_key(GeomType.HFIELD.value, GeomType.BOX.value)),
  (hfield_convex, group_key(GeomType.HFIELD.value, GeomType.MESH.value)),
  (sphere_sphere, group_key(GeomType.SPHERE.value, GeomType.SPHERE.value)),
  (sphere_capsule, group_key(GeomType.SPHERE.value, GeomType.CAPSULE.value)),
  (sphere_cylinder, group_key(GeomType.SPHERE.value, GeomType.CYLINDER.value)),
  (sphere_ellipsoid, group_key(GeomType.SPHERE.value, GeomType.ELLIPSOID.value)),
  (sphere_convex, group_key(GeomType.SPHERE.value, GeomType.BOX.value)),
  (sphere_convex, group_key(GeomType.SPHERE.value, GeomType.MESH.value)),
  (capsule_capsule, group_key(GeomType.CAPSULE.value, GeomType.CAPSULE.value)),
  (capsule_convex, group_key(GeomType.CAPSULE.value, GeomType.BOX.value)),
  (capsule_ellipsoid, group_key(GeomType.CAPSULE.value, GeomType.ELLIPSOID.value)),
  (capsule_cylinder, group_key(GeomType.CAPSULE.value, GeomType.CYLINDER.value)),
  (capsule_convex, group_key(GeomType.CAPSULE.value, GeomType.MESH.value)),
  (ellipsoid_ellipsoid, group_key(GeomType.ELLIPSOID.value, GeomType.ELLIPSOID.value)),
  (ellipsoid_cylinder, group_key(GeomType.ELLIPSOID.value, GeomType.CYLINDER.value)),
  (cylinder_cylinder, group_key(GeomType.CYLINDER.value, GeomType.CYLINDER.value)),
  (box_box, group_key(GeomType.BOX.value, GeomType.BOX.value)),
  (convex_convex, group_key(GeomType.BOX.value, GeomType.MESH.value)),
  (convex_convex, group_key(GeomType.MESH.value, GeomType.MESH.value)),
]


def narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.

  # we run the collision functions in increasing condim order to get the grouping
  # right from the get-go.

  for i in range(len(_COLLISION_FUNCS)):
    # this will lead to a bunch of unnecessary launches, but we don't want to sync at this point
    func, group_key = _COLLISION_FUNCS[i]
    func(m, d, group_key)
