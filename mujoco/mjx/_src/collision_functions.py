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
from .support import where

@wp.kernel
def plane_convex_kernel(
  m: Model,
  d: Data,
  group_key: int
):
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
  #if convex_type == wp.static(GeomType.BOX.value):
  #  pass # box-specific stuff - many things can be hardcoded here
  #else:
  #  pass # mesh-specific stuff

  # if contact
  index = wp.atomic_add(d.contact_counter, worldid, 1)
  #d.contact.dist[worldid, index] = dist
  #d.contact.pos[worldid, index] = pos
  #d.contact.frame[worldid, index] = frame


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
      dim=(d.nworld * d.max_num_overlaps_per_world),
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

