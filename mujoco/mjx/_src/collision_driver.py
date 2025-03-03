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
from .types import MJ_MINVAL
from .types import MJ_NREF
from .types import MJ_NIMP
from .types import array2df
from .types import array3df
from .types import NUM_GEOM_TYPES
from .types import vec5
from .collision_functions import _COLLISION_FUNCS
from .support import where
from .support import group_key


#####################################################################################
# BROADPHASE
#####################################################################################

# # use this kernel to get the AAMM for each body
# @wp.kernel
# def get_dyn_body_aamm(
#   m: Model,
#   d: Data,
# ):
#   env_id, bid = wp.tid()

#   # Initialize AAMM with extreme values
#   aamm_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
#   aamm_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)

#   # Iterate over all geometries associated with the body
#   for i in range(m.body_geomnum[bid]):
#     g = m.body_geomadr[bid] + i

#     for j in range(3):
#       pos = d.geom_xpos[env_id, g][j]
#       rbound = m.geom_rbound[g]
#       margin = m.geom_margin[g]

#       min_val = pos - rbound - margin
#       max_val = pos + rbound + margin

#       aamm_min[j] = wp.min(aamm_min[j], min_val)
#       aamm_max[j] = wp.max(aamm_max[j], max_val)

#   # Write results to output
#   d.dyn_body_aamm[env_id, bid, 0] = aamm_min
#   d.dyn_body_aamm[env_id, bid, 1] = aamm_max

@wp.kernel
def get_dyn_geom_aabb(
  m: Model,
  d: Data,
):
  env_id, gid = wp.tid()

  pos = d.geom_xpos[env_id, gid]
  ori = d.geom_xmat[env_id, gid]

  aabb = m.geom_aabb[gid, 0]
  aabb_pos = m.geom_aabb[gid, 1]

  aabb_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)

  for i in range(8):
    corner = wp.vec3(aabb.x, aabb.y, aabb.z)
    if i % 2 == 0:
      corner.x = -corner.x
    if (i // 2) % 2 == 0:
      corner.y = -corner.y
    if i < 4:
      corner.z = -corner.z
    corner_world = (
      ori * (corner + aabb_pos) + pos
    )
    wp.printf("%f %f %f\n", corner_world.x, corner_world.y, corner_world.z)
    aabb_max = wp.max(aabb_max, corner_world)
    aabb_min = wp.min(aabb_min, corner_world)

  # Write results to output
  d.dyn_geom_aabb[env_id, gid, 0] = aabb_min
  d.dyn_geom_aabb[env_id, gid, 1] = aabb_max


@wp.kernel
def init_contact_kernel(
  contact: Contact,
):
  contact_id = wp.tid()

  contact.dist[contact_id] = 1e12
  contact.pos[contact_id] = wp.vec3(0.0)
  contact.frame[contact_id] = wp.mat33f(0.0)
  contact.geom[contact_id] = wp.vec2i(-1, -1)
  contact.includemargin[contact_id] = 0.0
  contact.solref[contact_id].x = 0.02
  contact.solref[contact_id].y = 1.0
  contact.solimp[contact_id] = vec5(0.9, 0.95, 0.001, 0.5, 2.0)
  contact.friction[contact_id] = vec5(1.0, 1.0, 0.005, 0.0001, 0.0001)
  contact.solreffriction[contact_id] = wp.vec2(0.0, 0.0)


@wp.func
def overlap(
  world_id: int,
  a: int,
  b: int,
  boxes: wp.array(dtype=wp.vec3, ndim=3),
) -> bool:
  # Extract centers and sizes
  a_min = boxes[world_id, a, 0]
  a_max = boxes[world_id, a, 1]
  b_min = boxes[world_id, b, 0]
  b_max = boxes[world_id, b, 1]

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )


@wp.kernel
def broadphase_project_boxes_onto_sweep_direction_kernel(
  m: Model,
  d: Data,
):
  worldId, i = wp.tid()

  box_min = d.dyn_geom_aabb[worldId, i, 0]
  box_max = d.dyn_geom_aabb[worldId, i, 1]
  c = (box_min + box_max) * 0.5
  box_half_size = (box_max - box_min) * 0.5
  
  # Use fixed direction vector and its absolute values
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)
  abs_dir = wp.vec3(abs(direction.x), abs(direction.y), abs(direction.z))
  
  center = wp.dot(direction, c)
  d_val = wp.dot(box_half_size, abs_dir)
  f = center - d_val

  # Store results in the data arrays
  d.data_start[worldId, i] = f
  d.data_end[worldId, i] = center + d_val
  d.data_indexer[worldId, i] = i

  if i == 0:
    d.result_count[worldId] = 0  # Initialize result count to 0


@wp.kernel
def reorder_bounding_boxes_kernel(
  m: Model,
  d: Data,
):
  worldId, i = wp.tid()

  # Get the index from the data indexer
  mapped = d.data_indexer[worldId, i]

  # Get the box from the original boxes array
  box_min = d.dyn_geom_aabb[worldId, mapped, 0]
  box_max = d.dyn_geom_aabb[worldId, mapped, 1]

  # box = transform_aabb(
  #   box, box_translations[worldId, mapped], box_rotations[worldId, mapped]
  # )

  # Reorder the box into the sorted array
  d.boxes_sorted[worldId, i, 0] = box_min
  d.boxes_sorted[worldId, i, 1] = box_max


@wp.func
def find_first_greater_than(
  worldId: int,
  starts: wp.array(dtype=wp.float32, ndim=2),
  value: wp.float32,
  low: int,
  high: int,
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[worldId, mid] > value:
      high = mid
    else:
      low = mid + 1
  return low


@wp.kernel
def broadphase_sweep_and_prune_prepare_kernel(
  m: Model,
  d: Data,
):
  worldId, i = wp.tid()  # Get the thread ID

  # Get the index of the current bounding box
  idx1 = d.data_indexer[worldId, i]

  end = d.data_end[worldId, idx1]
  limit = find_first_greater_than(worldId, d.data_start, end, i + 1, m.ngeom)
  limit = wp.min(m.ngeom - 1, limit)

  # Calculate the range of boxes for the sweep and prune process
  count = limit - i

  # Store the cumulative sum for the current box
  d.ranges[worldId, i] = count


@wp.func
def find_right_most_index_int(
  starts: wp.array(dtype=wp.int32, ndim=1), value: wp.int32, low: int, high: int
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[mid] > value:
      high = mid
    else:
      low = mid + 1
  return high


@wp.func
def find_indices(
  id: int, cumulative_sum: wp.array(dtype=wp.int32, ndim=1), length: int
) -> wp.vec2i:
  # Perform binary search to find the right most index
  i = find_right_most_index_int(cumulative_sum, id, 0, length)

  # Get the baseId, and compute the offset and j
  if i > 0:
    base_id = cumulative_sum[i - 1]
  else:
    base_id = 0
  offset = id - base_id
  j = i + offset + 1

  return wp.vec2i(i, j)


@wp.kernel
def broadphase_sweep_and_prune_kernel(
  m: Model,
  d: Data,
  num_threads : int
):
  threadId = wp.tid()  # Get thread ID
  if d.cumulative_sum.shape[0] > 0:
    total_num_work_packages = d.cumulative_sum[d.cumulative_sum.shape[0] - 1]
  else:
    total_num_work_packages = 0

  while threadId < total_num_work_packages:
    # Get indices for current and next box pair
    ij = find_indices(threadId, d.cumulative_sum, d.cumulative_sum.shape[0])
    i = ij.x
    j = ij.y

    worldId = i // m.ngeom
    i = i % m.ngeom

    # world_id_j = j // m.ngeom
    j = j % m.ngeom

    # assert worldId == world_id_j, "Only boxes in the same world can be compared"
    # TODO: Remove print if debugging is done
    # if worldId != world_id_j:
    #     print("Only boxes in the same world can be compared")

    idx1 = d.data_indexer[worldId, i]

    # box1 = boxes_sorted[worldId, i]

    idx2 = d.data_indexer[worldId, j]

    body1 = wp.min(idx1, idx2)
    body2 = wp.max(idx1, idx2)

    # Collision filtering start
    """
    if (body_contype[body1] == 0 and body_conaffinity[body1] == 0) or (
      body_contype[body2] == 0 and body_conaffinity[body2] == 0
    ):
      threadId += num_threads
      continue

    signature = (body1 << 16) + body2
    filtered = bool(False)
    for i in range(nexclude):
      if exclude_signature[i] == signature:
        filtered = True
        break

    if filtered:
      threadId += num_threads
      continue

    w1 = body_weldid[body1]
    w2 = body_weldid[body2]
    if w1 == w2:
      threadId += num_threads
      continue

    # Filter parent not supported yet
    # w1_p = body_weldid[body_parentid[w1]]
    # w2_p = body_weldid[body_parentid[w2]]
    # if filter_parent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
          threadId += num_threads
    #     continue
    # Collision filtering end
    """

    body1 = m.geom_bodyid[i]
    body2 = m.geom_bodyid[j]
    if body1 == body2:
      threadId += num_threads
      continue

    # Check if the boxes overlap
    if body1 != body2 and overlap(worldId, i, j, d.boxes_sorted):
      # if not (body_has_plane[body1] or body_has_plane[body2]):
      #  return

      pair = wp.vec2i(body1, body2)

      id = wp.atomic_add(d.result_count, worldId, 1)

      if id < d.max_num_overlaps_per_world:
        d.broadphase_pairs[worldId, id] = pair

    threadId += num_threads


@wp.kernel
def get_contact_solver_params_kernel(
  m: Model,
  d: Data,
):
  tid = wp.tid()

  n_contact_pts = d.ncon[0]
  if tid >= n_contact_pts:
    return

  g1 = d.narrowphase_candidate_geom[tid, 0]
  g2 = d.narrowphase_candidate_geom[tid, 1]

  margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])
  gap = wp.max(m.geom_gap[g1], m.geom_gap[g2])
  solmix1 = m.geom_solmix[g1]
  solmix2 = m.geom_solmix[g2]
  mix = solmix1 / (solmix1 + solmix2)
  mix = where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
  mix = where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
  mix = where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)

  p1 = m.geom_priority[g1]
  p2 = m.geom_priority[g2]
  mix = where(p1 == p2, mix, where(p1 > p2, 1.0, 0.0))
  is_standard = (m.geom_solref[g1, 0] > 0) and (m.geom_solref[g2, 0] > 0)

  solref_ = wp.vec(0.0, length=MJ_NREF, dtype=wp.float32)
  for i in range(MJ_NREF):
    solref_[i] = mix * m.geom_solref[g1, i] + (1.0 - mix) * m.geom_solref[g2, i]
    solref_[i] = where(
      is_standard, solref_[i], wp.min(m.geom_solref[g1, i], m.geom_solref[g2, i])
    )

  # solimp_ = wp.zeros(mjNIMP, dtype=float)
  # for i in range(mjNIMP):
  #     solimp_[i] = mix * geom_solimp[i + g1 * mjNIMP] + (1 - mix) * geom_solimp[i + g2 * mjNIMP]

  friction_ = wp.vec3(0.0, 0.0, 0.0)
  for i in range(3):
    friction_[i] = wp.max(m.geom_friction[g1, i], m.geom_friction[g2, i])

  friction5 = vec5(friction_[0], friction_[0], friction_[1], friction_[2], friction_[2])

  d.contact.includemargin[tid] = margin - gap
  d.contact.friction[tid] = friction5

  for i in range(2):
    d.contact.solref[tid][i] = solref_[i]

  for i in range(MJ_NIMP):
    d.contact.solimp[tid][i] = (
      mix * d.contact.geom_solimp[g1, i] + (1.0 - mix) * d.contact.geom_solimp[g2, i]
    )  # solimp_[i]


@wp.kernel
def group_contacts_by_type_kernel(
  m: Model,
  d: Data,
):
  worldid, tid = wp.tid()
  if tid >= d.result_count[worldid]:
    return

  geoms = d.broadphase_pairs[worldid, tid]
  geom1 = geoms[0]
  geom2 = geoms[1]

  type1 = m.geom_type[geom1]
  type2 = m.geom_type[geom2]
  key = group_key(type1, type2)

  n_type_pair = wp.atomic_add(d.narrowphase_candidate_group_count, key, 1)
  d.narrowphase_candidate_worldid[key, n_type_pair] = worldid
  d.narrowphase_candidate_geom[key, n_type_pair] = wp.vec2i(geom1, geom2)


def broadphase_sweep_and_prune(m: Model, d: Data):
  """Broad-phase collision detection via sweep-and-prune."""

  # Directional vectors for sweep
  # TODO: Improve picking of direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)
  abs_dir = wp.vec3(abs(direction.x), abs(direction.y), abs(direction.z))

  print("broadphase_project_boxes_onto_sweep_direction")
  wp.launch(
    kernel=broadphase_project_boxes_onto_sweep_direction_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )
  wp.synchronize()
  segmented_sort_available = hasattr(wp.utils, "segmented_sort_pairs")

  if segmented_sort_available:
    print("Using segmented sort")
    wp.utils.segmented_sort_pairs(
      d.data_start, d.data_indexer, m.ngeom * d.nworld, d.segment_indices, d.nworld
    )
    wp.synchronize()
  else:
    # Sort each world's segment separately
    for world_id in range(d.nworld):
      start_idx = world_id * m.ngeom

      # Create temporary arrays for sorting
      temp_data_start = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_start.dtype,
      )
      temp_data_indexer = wp.zeros(
        m.ngeom * 2,
        dtype=d.data_indexer.dtype,
      )

      # Copy data to temporary arrays
      wp.copy(
        temp_data_start,
        d.data_start,
        0,
        start_idx,
        m.ngeom,
      )
      wp.copy(
        temp_data_indexer,
        d.data_indexer,
        0,
        start_idx,
        m.ngeom,
      )

      # Sort the temporary arrays
      wp.utils.radix_sort_pairs(temp_data_start, temp_data_indexer, m.ngeom)

      # Copy sorted data back
      wp.copy(
        d.data_start,
        temp_data_start,
        start_idx,
        0,
        m.ngeom,
      )
      wp.copy(
        d.data_indexer,
        temp_data_indexer,
        start_idx,
        0,
        m.ngeom,
      )

  print("reorder_bounding_boxes_kernel")
  wp.launch(
    kernel=reorder_bounding_boxes_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )
  wp.synchronize()

  print("broadphase_sweep_and_prune_prepare_kernel")
  wp.launch(
    kernel=broadphase_sweep_and_prune_prepare_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )
  wp.synchronize()
  # The scan (scan = cumulative sum, either inclusive or exclusive depending on the last argument) is used for load balancing among the threads
  wp.utils.array_scan(d.ranges.reshape(-1), d.cumulative_sum, True)

  # Estimate how many overlap checks need to be done - assumes each box has to be compared to 5 other boxes (and batched over all worlds)
  num_sweep_threads = 5 * d.nworld * m.ngeom
  print("broadphase_sweep_and_prune_kernel")
  wp.launch(
    kernel=broadphase_sweep_and_prune_kernel,
    dim=num_sweep_threads,
    inputs=[m, d, num_sweep_threads],
  )
  wp.synchronize()

###########################################################################################3


def init_contact(m: Model, d: Data):
  # initialize output data
  wp.launch(
    kernel=init_contact_kernel,
    dim=(d.nconmax),
    inputs=[d.contact],
  )


def broadphase(m: Model, d: Data):
  # broadphase collision detection

  # generate body AAMMs
  # generate body pairs
  # get geom AABBs in global frame
  # get geom pairs

  # get geom AABBs in global frame
  # wp.launch(
  #   kernel=get_dyn_body_aamm,
  #   dim=(d.nworld, m.nbody),
  #   inputs=[m, d],
  # )

  print("get_dyn_geom_aabb")
  wp.launch(
    kernel=get_dyn_geom_aabb,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )
  wp.synchronize()

  print(d.dyn_geom_aabb)

  broadphase_sweep_and_prune(m, d)

def group_contacts_by_type(m: Model, d: Data):
  # initialize type pair count & group contacts by type

  # Initialize type pair count 
  d.narrowphase_candidate_group_count.zero_()

  wp.launch(
    group_contacts_by_type_kernel,
    dim=(d.nworld, d.max_num_overlaps_per_world),
    inputs=[m, d],
  )

  # Initialize the env contact counter
  d.ncon.zero_()


def get_contact_solver_params(m: Model, d: Data):
  wp.launch(
    get_contact_solver_params_kernel,
    dim=[d.nconmax],
    inputs=[m, d],
  )

  # TODO(team): do we need condim sorting, deepest penetrating contact here?


def collision(m: Model, d: Data):
  """Collision detection."""

  # AD: based on engine_collision_driver.py in Eric's warp fork/mjx-collisions-dev
  # which is further based on the CUDA code here:
  # https://github.com/btaba/mujoco/blob/warp-collisions/mjx/mujoco/mjx/_src/cuda/engine_collision_driver.cu.cc#L458-L583

  init_contact(m, d)
  broadphase(m, d)
  # filtering?
  group_contacts_by_type(m, d)
  # XXX switch between collision functions and GJK/EPA here
  if True:
    from .collision_functions import narrowphase
  else:
    from .collision_convex import narrowphase
  narrowphase(m, d)
  get_contact_solver_params(m, d)
