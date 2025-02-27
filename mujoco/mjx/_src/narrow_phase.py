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

BoxType = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


@wp.func
def where(condition: bool, onTrue: float, onFalse: float) -> float:
    if condition:
        return onTrue
    else:
        return onFalse

@wp.func
def orthogonals(a: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
  """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = where(-0.5 < a[1] and a[1] < 0.5, y, z)
  b = b - a * wp.dot(a, b)
  # normalize b. however if a is a zero vector, zero b as well.
  b = wp.normalize(b) * float(wp.length(a) > 0.0)
  return b, wp.cross(a, b)


@wp.func
def make_frame(a: wp.vec3) -> wp.mat33:
  """Makes a right-handed 3D frame given a direction."""
  a = wp.normalize(a)
  b, c = orthogonals(a)
  return wp.mat33(a, b, c)




@wp.func
def _manifold_points(
  worldId : int,
  poly: wp.array(dtype=wp.vec3, ndim=2),
  poly_start: int,
  poly_count: int,
  poly_norm: wp.vec3,
  plane_pos: wp.vec3,
  n: wp.vec3,
  max_support: float,
) -> wp.vec4i:
  """Chooses four points on the polygon with approximately maximal area."""
  max_val = float(-1e6)
  a_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      a_idx = i
  a = poly[worldId, poly_start + a_idx]
  # choose point b furthest from a
  max_val = float(-1e6)
  b_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = wp.length_sq(a - poly[worldId, poly_start + i]) + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      b_idx = i
  b = poly[worldId, poly_start + b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = wp.cross(poly_norm, a - b)
  # ap = a - poly
  max_val = float(-1e6)
  c_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = wp.abs(wp.dot(a - poly[worldId, poly_start + i], ab)) + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    if val > max_val:
      max_val = val
      c_idx = i
  c = poly[worldId, poly_start + c_idx]
  # choose point d furthest from the other two triangle edges
  ac = wp.cross(poly_norm, a - c)
  bc = wp.cross(poly_norm, b - c)
  # bp = b - poly
  max_val = float(-1e6)
  d_idx = int(0)
  for i in range(poly_count):
    support = wp.dot(plane_pos - poly[worldId, poly_start+i], n)
    val = (
      wp.abs(wp.dot(b - poly[worldId, poly_start + i], bc))
      + wp.abs(wp.dot(a - poly[worldId, poly_start + i], ac))
      + where(support > wp.max(0.0, max_support - 1e-3), 0.0, -1e6)
    )
    if val > max_val:
      max_val = val
      d_idx = i
  return wp.vec4i(a_idx, b_idx, c_idx, d_idx)


@wp.func
def plane_convex(
  m: Model,
  d: Data,
  worldId : int,
  planeIndex: int,
  convexIndex: int,
  outBaseIndex: int,
  result: Contact,
):
  """Calculates contacts between a plane and a convex object."""
  vert_start = m.geom_vert_addr[worldId, convexIndex]
  vert_count = m.geom_vert_num[worldId, convexIndex]

  convexPos = d.geom_pos[worldId, convexIndex]
  convexMat = d.geom_mat[worldId, convexIndex]

  planePos = d.geom_pos[worldId, planeIndex]
  planeMat = d.geom_mat[worldId, planeIndex]

  # get points in the convex frame
  plane_pos = wp.transpose(convexMat) @ (planePos - convexPos)
  n = (
    wp.transpose(convexMat) @ planeMat[2]
  )  # TODO: Does [2] indeed return the last column of the matrix?

  max_support = float(-100000)
  for i in range(vert_count):
    max_support = wp.max(max_support, wp.dot(plane_pos - m.vert[worldId, vert_start+i], n))

  # search for manifold points within a 1mm skin depth
  idx = wp.vec4i(0)
  idx = _manifold_points(worldId, m.vert, vert_start, vert_count, n, plane_pos, n, max_support)
  frame = make_frame(
    wp.vec3(
      planeMat[0, 2], planeMat[1, 2], planeMat[2, 2]
    )
  )


  for i in range(4):
    # Get vertex position and convert to world frame
    id = int(idx[i])
    pos_i = m.vert[worldId, id]
    pos_i = convexPos + pos_i @ wp.transpose(convexMat)

    # Compute uniqueness by comparing with previous indices
    count = int(0)
    for j in range(i + 1):
      if idx[i] == idx[j]:
        count += 1
    unique = where(count == 1, 1.0, 0.0)

    # Compute distance and final position
    dist_i = where(unique > 0.0, -wp.dot(plane_pos - m.vert[worldId, vert_start+i], n), 1.0)
    pos_i = pos_i - 0.5 * dist_i * frame[2]

    # Store results
    result.dist[outBaseIndex + i] = dist_i
    result.pos[outBaseIndex + i] = pos_i
    result.frame[outBaseIndex + i] = frame

  # return ret




@wp.func
def pack_key(condim: int, g_min: int, g_max: int, local_id: int) -> int:
    return (condim << 28) | (g_min << 16) | (g_max << 3) | local_id

@wp.func
def decompose_key(key: int) -> wp.vec4i:
    # Extract components from key using bit masks and shifts
    condim = (key >> 28) & 0xF # 4 bits for condim
    g_min = (key >> 16) & 0xFFF # 12 bits for g_min
    g_max = (key >> 3) & 0x1FFF # 13 bits for g_max
    local_id = key & 7
    return wp.vec3i(condim, g_min, g_max, local_id)


@wp.kernel
def create_batched_contact_arrays(
    m: Model, 
    d: Data,
    overlap_pairs: wp.array(dtype=wp.vec2i, ndim=2),
    overlap_pairs_count: wp.array(dtype=int, ndim=1),
    sort_keys: wp.array(dtype=int, ndim=2), 
    sort_indexer: wp.array(dtype=int, ndim=2), 
    contact_indexer : wp.array(dtype=int)):
    """Exports contact data in a batched structure.
    
    The contact data follows this shape convention: (batch_size, n_contacts, ...)
    
    Example structure:
    - dist: (batch_size, n_contacts)         # penetration depths
    - pos: (batch_size, n_contacts, 3)       # 3D contact positions
    - frame: (batch_size, n_contacts, 3, 3)  # contact frames (rotation matrices)
    - geom1: (batch_size, n_contacts)        # first geom indices
    - geom2: (batch_size, n_contacts)        # second geom indices
    
    Example values for a batch_size=2 with 3 contacts:
    dist = [
        [-0.01, -0.005, -0.002],  # batch 1: penetration depths
        [-0.015, -0.008, -0.003]  # batch 2: penetration depths
    ]
    pos = [
        [[1.0, 0.0, 0.0],         # batch 1: 3D positions
          [0.0, 1.0, 0.0],
          [0.5, 0.5, 0.0]],
        [[1.2, 0.1, 0.0],         # batch 2: 3D positions
          [0.1, 1.1, 0.0],
          [0.6, 0.4, 0.0]]
    ]
    """
    world_id, contact_id = wp.tid()


    num_generated_contacts = 4 # info depending on collision type (e. g. 4 for plane_convex)

    num_generated_contacts = overlap_pairs_count[world_id]
    if(contact_id >= num_generated_contacts):      
      return

    geom = overlap_pairs[world_id, contact_id]
    geom1 = geom[0]
    geom2 = geom[1]

    condim = 3 # TODO: where to get this from?

    base_contact_id = wp.atomic_add(contact_indexer, 0, num_generated_contacts)

    # Generate the contact(s) here
    #dist, pos, frame = plane_convex(base_contact_id)
    plane_convex(m, d, world_id, geom1, geom2, base_contact_id, d.contact) # should write dist, pos and frame directly

    

    # write_dist_pos_frame(worldId, base_contact_id + 0, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 1, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 2, ...)
    # write_dist_pos_frame(worldId, base_contact_id + 3, ...)

    # Write codim
    # Write contact key (generated from g1 and g2)
    # Write contact sub id (local contact id)

    g_min = wp.min(geom1, geom2) 
    g_max = wp.max(geom1, geom2)
    local_id = int(3)  
    
    # Sort key
    #base_key = (condim << 28) | (g_min << 16) | (g_max << 3) # | local_id
    sort_keys[world_id, base_contact_id + 0] = pack_key(condim, g_min, g_max, 0) # base_key | 0
    sort_keys[world_id, base_contact_id + 1] = pack_key(condim, g_min, g_max, 1) # base_key | 1
    sort_keys[world_id, base_contact_id + 2] = pack_key(condim, g_min, g_max, 2) # base_key | 2
    sort_keys[world_id, base_contact_id + 3] = pack_key(condim, g_min, g_max, 3) # base_key | 3
    
    sort_indexer[world_id, base_contact_id + 0] = base_contact_id + 0
    sort_indexer[world_id, base_contact_id + 1] = base_contact_id + 1
    sort_indexer[world_id, base_contact_id + 2] = base_contact_id + 2
    sort_indexer[world_id, base_contact_id + 3] = base_contact_id + 3

    pass


@wp.kernel
def duplicate_properties_per_contact_point(m: Model, d: Data, sort_keys: wp.array(dtype=int, ndim=2)):
    """Exports contact properties, repeating them when necessary to match contact structure.
    
    When a collision function returns multiple contact points (ncon > 1) for a single
    geometry pair, this function repeats the properties to match the number of contacts.
    
    Example:
    For a box-plane collision with 4 contact points (corners):
    Original properties:
        friction = [1.0]           # single friction value
        includemargin = [0.01]     # single margin value
    
    Repeated properties (ncon=4):
        friction = [1.0, 1.0, 1.0, 1.0]
        includemargin = [0.01, 0.01, 0.01, 0.01]
    
    This ensures each contact point has its own set of parameters, even though
    they're identical for contacts from the same geometry pair.
    """
    
    world_id, contact_id = wp.tid()
    if(contact_id >= d.contact_count[world_id]):
      return

    sort_key = sort_keys[world_id, contact_id]

    sort_key_parts = decompose_key(sort_key)
    condim = sort_key_parts[0]
    g_min = sort_key_parts[1]
    g_max = sort_key_parts[2]
    # local_id = sort_key_parts[3]

    geom1 = g_min
    geom2 = g_max
    
    eps = mujoco.mjMINVAL

    margin = wp.maximum(m.geom_margin[geom1], m.geom_margin[geom2])
    gap = wp.maximum(m.geom_gap[geom1], m.geom_gap[geom2])
    solmix1, solmix2 = m.geom_solmix[geom1], m.geom_solmix[geom2]
    mix = solmix1 / (solmix1 + solmix2)
    mix = where((solmix1 < eps) & (solmix2 < eps), 0.5, mix)
    mix = where((solmix1 < eps) & (solmix2 >= eps), 0.0, mix)
    mix = where((solmix1 >= eps) & (solmix2 < eps), 1.0, mix)
    # friction: max
    friction = wp.maximum(m.geom_friction[geom1], m.geom_friction[geom2])
    solref1, solref2 = m.geom_solref[geom1], m.geom_solref[geom2]
    # reference standard: mix
    solref_standard = mix * solref1 + (1 - mix) * solref2
    # reference direct: min
    solref_direct = wp.minimum(solref1, solref2)
    is_standard = (solref1[0] > 0) & (solref2[0] > 0)
    solref = wp.where(is_standard, solref_standard, solref_direct)
    # solreffriction = jp.zeros(geom1.shape + (mujoco.mjNREF,))
    # impedance: mix
    solimp = mix * m.geom_solimp[geom1] + (1 - mix) * m.geom_solimp[geom2]

    pri = m.geom_priority[geom1] != m.geom_priority[geom2]
    # if pri.any():
    if pri:
        # use priority geom when specified instead of mixing
        gp1, gp2 = m.geom_priority[geom1], m.geom_priority[geom2]
        gp = where(gp1 > gp2, geom1, geom2)[pri]
        friction = friction.at[pri].set(m.geom_friction[gp])
        solref = solref.at[pri].set(m.geom_solref[gp])
        solimp = solimp.at[pri].set(m.geom_solimp[gp])

    d.contact.dist[world_id, contact_id] = margin - gap
    d.contact.friction[world_id, contact_id, 0] = friction[0] 
    d.contact.friction[world_id, contact_id, 1] = friction[0]
    d.contact.friction[world_id, contact_id, 2] = friction[1]
    d.contact.friction[world_id, contact_id, 3] = friction[2]
    d.contact.friction[world_id, contact_id, 4] = friction[2]
    d.contact.solref[world_id, contact_id] = solref
    d.contact.solimp[world_id, contact_id] = solimp
    d.contact.includemargin[world_id, contact_id] = margin
    d.contact.dim[world_id, contact_id] = condim
    d.contact.geom[world_id, contact_id, 0] = geom1
    d.contact.geom[world_id, contact_id, 1] = geom2
    d.contact.solreffriction[world_id, contact_id] = wp.zeros(types.MJ_NREF, dtype=wp.float32)

@wp.kernel
def contact_count_to_sort_sections(max_contacts_per_world: int, contact_count_per_world: wp.array(dtype=int), sort_sections_starts: wp.array(dtype=int), sort_sections_ends: wp.array(dtype=int)):
    world_id = wp.tid()

    count = contact_count_per_world[world_id]

    sort_sections_starts[world_id] = world_id * max_contacts_per_world
    sort_sections_ends[world_id] = sort_sections_starts[world_id] + count

    
def organize_contacts_by_constraint_dimension(
        m: Model, 
        d: Data, 
        sort_keys: wp.array(dtype=int, ndim=2), 
        sort_indexer: wp.array(dtype=int, ndim=2), 
        num_contacts_per_world : wp.array(dtype=int),
        sort_sections_starts : wp.array(dtype=int),
        sort_sections_ends : wp.array(dtype=int)):
    """Groups contacts by their contact dimension (condim).
    
    Contacts are grouped by their degrees of freedom:
    - condim=1: normal force only (frictionless)
    - condim=3: normal force + friction in 2 tangential directions
    - condim=6: normal force + friction + torsional friction
    
    Example grouping:
    condim_groups = {
        1: [sphere_plane_contacts],  # frictionless contacts
        3: [box_plane_contacts],     # friction contacts
        6: [gear_contacts]           # torsional friction contacts
    }
    
    The groups are later concatenated in order of increasing condim to ensure
    consistent contact ordering for the constraint solver.
    """

    wp.launch(
        kernel=contact_count_to_sort_sections,
        dim=d.nworld,
        inputs=[
            m.max_num_overlaps_per_world,
            num_contacts_per_world,
        ],
        outputs=[
            sort_sections_starts,
            sort_sections_ends,
        ],
    )

    wp.utils.segmented_sort_pairs_start_end(
        sort_keys.reshape(-1),
        sort_indexer.reshape(-1),
        m.max_num_overlaps_per_world * d.nworld,
        sort_sections_starts,
        sort_sections_ends,
        d.nworld,
    )

# TODO: Not strictly required for now. Implement later
@wp.func
def select_deepest_penetrating_contacts():
    """Limits the number of contacts per condim group based on penetration depth.
    
    When max_contact_points is set, this function:
    1. Processes each condim group separately
    2. Selects the contacts with deepest penetration
    3. Discards less important contacts
    
    Example:
    With max_contact_points = 2 and these penetration depths:
        dist = [-0.05, -0.03, -0.01, -0.02]
    
    Result after filtering:
        dist = [-0.05, -0.03]  # kept two deepest penetrations
    
    This is important because:
    1. Reduces simulation computational cost
    2. Keeps the most physically significant contacts
    3. Maintains contact type diversity by limiting each condim group separately
    
    Example with max_contact_points = 4:
    - 6 contacts with condim=1 → reduced to 4
    - 3 contacts with condim=3 → stays 3
    - 5 contacts with condim=6 → reduced to 4
    Final result: 11 total contacts (4 + 3 + 4)
    """
    pass



def find_collisions(
  m: Model, 
  d: Data
):

  wp.launch(
    kernel=create_batched_contact_arrays,
    dim=(d.nworld, m.max_num_overlaps_per_world),
    inputs=[
      m,
      d,
      d.broadphase_pairs,
      d.result_count,
      d.narrow_phase_sort_keys,
      d.narrow_phase_sort_indexer,
      d.narrow_phase_contact_indexer,
    ],
  )
  
  organize_contacts_by_constraint_dimension(m, d, d.narrow_phase_sort_keys, d.narrow_phase_sort_indexer, d.narrow_phase_contact_count, 
    d.narrow_phase_sort_sections_starts, d.narrow_phase_sort_sections_ends)
  
  wp.launch(
    kernel=duplicate_properties_per_contact_point,
    dim=(d.nworld, d.max_num_contacts_per_world),
    inputs=[m, d, d.narrow_phase_sort_keys],
  )
  
  # select_deepest_penetrating_contacts(m, d, d.narrow_phase_sort_keys, d.narrow_phase_sort_indexer, d.narrow_phase_sort_sections) # Optional
  pass

def narrow_phase(m: Model, d: Data) -> Data:
  """Narrow phase collision detection."""
  pass