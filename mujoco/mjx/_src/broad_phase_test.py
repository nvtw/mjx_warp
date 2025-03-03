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

"""Tests for broad phase functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

from . import test_util

BoxType = wp.types.matrix(shape=(2, 3), dtype=wp.float32)


class AABB:
  min: wp.vec3
  max: wp.vec3


def transform_aabb(aabb_pos, aabb_size, pos: wp.vec3, ori: wp.mat33) -> AABB:
  
  aabb = AABB()
  aabb.max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb.min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)

  for i in range(8):
    corner = wp.vec3(aabb_size[0], aabb_size[1], aabb_size[2])
    if i % 2 == 0:
      corner.x = -corner.x
    if (i // 2) % 2 == 0:
      corner.y = -corner.y
    if i < 4:
      corner.z = -corner.z
    corner_world = (
      ori @ (corner + wp.vec3(aabb_pos[0], aabb_pos[1], aabb_pos[2])) + wp.vec3(pos[0], pos[1], pos[2])
    )
    aabb.max = wp.max(aabb.max, corner_world)
    aabb.min = wp.min(aabb.min, corner_world)
    
  return aabb


def overlap(
  a: AABB,
  b: AABB,
) -> bool:
  # Extract centers and sizes
  a_min = a.min
  a_max = a.max
  b_min = b.min
  b_max = b.max

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )


def find_overlaps_brute_force(worldId: int, num_boxes_per_world: int, boxes, pos, rot):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.
  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """
  overlaps = []

  for i in range(num_boxes_per_world):
    aabb_i = transform_aabb(boxes[i][0], boxes[i][1], pos[worldId][i], rot[worldId][i])

    for j in range(i + 1, num_boxes_per_world):
      aabb_j = transform_aabb(boxes[j][0], boxes[j][1], pos[worldId][j], rot[worldId][j])
      
      if overlap(aabb_i, aabb_j):
        overlaps.append((i, j))  # Store indices of overlapping boxes

  return overlaps


def find_overlaps_brute_force_batched(
  num_worlds: int, num_boxes_per_world: int, boxes, pos, rot
):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.
  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """

  overlaps = []

  for worldId in range(num_worlds):
    overlaps.append(
      find_overlaps_brute_force(worldId, num_boxes_per_world, boxes, pos, rot)
    )

  return overlaps


class MultiIndexList:
  def __init__(self):
    self.data = {}

  def __setitem__(self, key, value):
    worldId, i = key
    if worldId not in self.data:
      self.data[worldId] = []
    if i >= len(self.data[worldId]):
      self.data[worldId].extend([None] * (i - len(self.data[worldId]) + 1))
    self.data[worldId][i] = value

  def __getitem__(self, key):
    worldId, i = key
    return self.data[worldId][i]  # Raises KeyError if not found


class BroadPhaseTest(parameterized.TestCase):
  def test_broad_phase(self):
    """Tests broad phase."""
    _, mjd, m, d = test_util.fixture("cube.xml")

   
    aabbs = m.geom_aabb.numpy()
    pos = d.geom_xpos.numpy()
    rot = d.geom_xmat.numpy()

    aabbs = aabbs.reshape((m.ngeom, 2, 3))
    pos = pos.reshape((d.nworld, m.ngeom, 3))
    rot = rot.reshape((d.nworld, m.ngeom, 3, 3))

    brute_force_overlaps = find_overlaps_brute_force_batched(d.nworld, m.ngeom, aabbs, pos, rot)


    mjx.broadphase(m, d)

    result = d.broadphase_pairs
    result_count = d.result_count

    # Get numpy arrays from result and result_count
    result_np = result.numpy()
    result_count_np = result_count.numpy()

    # Iterate over each world
    for world_idx in range(d.nworld):
      # Get number of collisions for this world
      num_collisions = result_count_np[world_idx]
      print(f"Number of collisions for world {world_idx}: {num_collisions}")

      list = brute_force_overlaps[world_idx]
      assert len(list) == num_collisions, "Number of collisions does not match"

      # Print each collision pair
      for i in range(num_collisions):
        pair = result_np[world_idx][i]

        # Convert pair to tuple for comparison
        pair_tuple = (int(pair[0]), int(pair[1]))
        assert pair_tuple in list, (
          f"Collision pair {pair_tuple} not found in brute force results"
        )

    return



    # Create some test boxes
    num_worlds = d.nworld
    num_boxes_per_world = m.ngeom
    print(f"num_worlds: {num_worlds}, num_boxes_per_world: {num_boxes_per_world}")

    # Parameters for random box generation
    sample_space_origin = wp.vec3(-10.0, -10.0, -10.0)  # Origin of the bounding volume
    sample_space_size = wp.vec3(20.0, 20.0, 20.0)  # Size of the bounding volume
    min_edge_length = 0.5  # Minimum edge length of random boxes
    max_edge_length = 5.0  # Maximum edge length of random boxes

    boxes_list = []

    # Set random seed for reproducibility
    import random

    random.seed(11)

    # Generate random boxes for each world
    for _ in range(num_boxes_per_world):
      # Generate random position within bounding volume
      pos_x = sample_space_origin.x + random.random() * sample_space_size.x
      pos_y = sample_space_origin.y + random.random() * sample_space_size.y
      pos_z = sample_space_origin.z + random.random() * sample_space_size.z

      # Generate random box dimensions between min and max edge lengths
      size_x = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_y = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_z = min_edge_length + random.random() * (max_edge_length - min_edge_length)

      # Create box with random position and size
      box_min, box_max = init_box(pos_x, pos_y, pos_z, pos_x + size_x, pos_y + size_y, pos_z + size_z)
      boxes_list.append([box_min, box_max])

    # Generate random positions and orientations for each box
    pos = []
    rot = []
    for _ in range(num_worlds * num_boxes_per_world):
      # Random position within bounding volume
      pos_x = sample_space_origin.x + random.random() * sample_space_size.x
      pos_y = sample_space_origin.y + random.random() * sample_space_size.y
      pos_z = sample_space_origin.z + random.random() * sample_space_size.z
      pos.append(wp.vec3(pos_x, pos_y, pos_z))
      # pos.append(wp.vec3(0, 0, 0))

      # Random rotation matrix
      rx = random.random() * 6.28318530718  # 2*pi
      ry = random.random() * 6.28318530718
      rz = random.random() * 6.28318530718
      axis = wp.vec3(rx, ry, rz)
      axis = axis / wp.length(axis)  # normalize axis
      angle = random.random() * 6.28318530718  # random angle between 0 and 2*pi
      rot.append(wp.quat_to_matrix(wp.quat_from_axis_angle(axis, angle)))
      # rot.append(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(1, 0, 0), float(0))))

    # Convert pos and rot to MultiIndexList format
    pos_multi = MultiIndexList()
    rot_multi = MultiIndexList()

    # Populate the MultiIndexLists using pos and rot data
    idx = 0
    for world_idx in range(num_worlds):
      for i in range(num_boxes_per_world):
        pos_multi[world_idx, i] = pos[idx]
        rot_multi[world_idx, i] = rot[idx]
        idx += 1

    brute_force_overlaps = find_overlaps_brute_force_batched(
      num_worlds, num_boxes_per_world, boxes_list, pos_multi, rot_multi
    )

    # Test the broad phase by setting custom aabb data
    d.dyn_geom_aabb = wp.array(
      boxes_list, dtype=wp.vec3, ndim=2
    )
    d.dyn_geom_aabb = d.dyn_geom_aabb.reshape((num_worlds, num_boxes_per_world, 2))
    d.geom_xpos = wp.array(pos, dtype=wp.vec3)
    d.geom_xpos = d.geom_xpos.reshape((num_worlds, num_boxes_per_world))
    d.geom_xmat = wp.array(rot, dtype=wp.mat33)
    d.geom_xmat = d.geom_xmat.reshape((num_worlds, num_boxes_per_world))

    print("mjx.broadphase")
    mjx.broadphase(m, d)

    result = d.broadphase_pairs
    result_count = d.result_count

    # Get numpy arrays from result and result_count
    result_np = result.numpy()
    result_count_np = result_count.numpy()

    # Iterate over each world
    for world_idx in range(num_worlds):
      # Get number of collisions for this world
      num_collisions = result_count_np[world_idx]
      print(f"Number of collisions for world {world_idx}: {num_collisions}")

      list = brute_force_overlaps[world_idx]
      assert len(list) == num_collisions, "Number of collisions does not match"

      # Print each collision pair
      for i in range(num_collisions):
        pair = result_np[world_idx][i]

        # Convert pair to tuple for comparison
        pair_tuple = (int(pair[0]), int(pair[1]))
        assert pair_tuple in list, (
          f"Collision pair {pair_tuple} not found in brute force results"
        )


  def test_broadphase_simple(self):
    """Tests the broadphase"""

    # create a model with a few intersecting bodies
    _MODEL = """
    <mujoco>
      <worldbody>
        <geom size="40 40 40" type="plane"/>   <!- (0) intersects with nothing -->
        <body pos="0 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (1) intersects with 2 -->
        </body>
        <body pos="0.1 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (2) intersects with 1 -->
        </body>

        <body pos="1.8 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (3) intersects with 4  -->
        </body>
        <body pos="1.6 0 0.7">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (4) intersects with 3 -->
        </body>

        <body pos="0 0 1.8">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (5) intersects with 7 -->
          <geom size="0.5 0.5 0.5" type="box" pos="0 0 -1"/> <!- (6) intersects with 7, 2, 1 -->
        </body>
        <body pos="0 0.5 1.2">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/> <!- (7) intersects with 5 -->
        </body>
        
      </worldbody>
    </mujoco>
    """

    m = mujoco.MjModel.from_xml_string(_MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    mjx.broadphase(mx, dx)

    assert(dx.result_count.numpy()[0] == 9)


if __name__ == "__main__":
  wp.init()
  absltest.main()
