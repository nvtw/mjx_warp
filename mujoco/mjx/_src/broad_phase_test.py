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



# Helper function to initialize a box
def init_box(min_x, min_y, min_z, max_x, max_y, max_z):
  center = wp.vec3((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
  size = wp.vec3(max_x - min_x, max_y - min_y, max_z - min_z)
  box = wp.types.matrix(shape=(2, 3), dtype=wp.float32)(
    [center.x, center.y, center.z, size.x, size.y, size.z]
  )
  return box


def overlap(
  a: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
  b: wp.types.matrix(shape=(2, 3), dtype=wp.float32),
) -> bool:
  # Extract centers and sizes
  a_center = a[0]
  a_size = a[1]
  b_center = b[0]
  b_size = b[1]

  # Calculate min/max from center and size
  a_min = a_center - 0.5 * a_size
  a_max = a_center + 0.5 * a_size
  b_min = b_center - 0.5 * b_size
  b_max = b_center + 0.5 * b_size

  return not (
    a_min.x > b_max.x
    or b_min.x > a_max.x
    or a_min.y > b_max.y
    or b_min.y > a_max.y
    or a_min.z > b_max.z
    or b_min.z > a_max.z
  )


def find_overlaps_brute_force(worldId: int, num_boxes_per_world: int, boxes):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.

  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """
  overlaps = []

  for i in range(num_boxes_per_world):
    box_a = boxes[worldId, i]

    for j in range(i + 1, num_boxes_per_world):
      box_b = boxes[worldId, j]

      # Use the overlap function to check for overlap
      if overlap(box_a, box_b):
        overlaps.append((i, j))  # Store indices of overlapping boxes

  return overlaps


def find_overlaps_brute_force_batched(num_worlds: int, num_boxes_per_world: int, boxes):
  """
  Finds overlapping bounding boxes using the brute-force O(n^2) algorithm.

  Returns:
      List of tuples [(idx1, idx2)] where idx1 and idx2 are indices of overlapping boxes.
  """

  overlaps = []

  for worldId in range(num_worlds):
    overlaps.append(find_overlaps_brute_force(worldId, num_boxes_per_world, boxes))

  # Show progress bar for brute force computation
  # from tqdm import tqdm

  # for worldId in tqdm(range(num_worlds), desc="Computing overlaps"):
  #    overlaps.append(find_overlaps_brute_force(worldId, num_boxes_per_world, boxes))

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
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml")

    # Create some test boxes
    num_worlds = 5
    num_boxes_per_world = 40

    # Parameters for random box generation
    box_origin = wp.vec3(-10.0, -10.0, -10.0)  # Origin of the bounding volume
    box_size = wp.vec3(20.0, 20.0, 20.0)  # Size of the bounding volume
    min_edge_length = 0.5  # Minimum edge length of random boxes
    max_edge_length = 5.0  # Maximum edge length of random boxes

    boxes_list = []

    # Set random seed for reproducibility
    import random
    random.seed(11)

    # Generate random boxes for each world
    for _ in range(num_worlds * num_boxes_per_world):
      # Generate random position within bounding volume
      pos_x = box_origin.x + random.random() * box_size.x
      pos_y = box_origin.y + random.random() * box_size.y
      pos_z = box_origin.z + random.random() * box_size.z

      # Generate random box dimensions between min and max edge lengths
      size_x = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_y = min_edge_length + random.random() * (max_edge_length - min_edge_length)
      size_z = min_edge_length + random.random() * (max_edge_length - min_edge_length)

      # Create box with random position and size
      boxes_list.append(
        init_box(pos_x, pos_y, pos_z, pos_x + size_x, pos_y + size_y, pos_z + size_z)
      )

    # Convert boxes_list to MultiIndexList format
    boxes = MultiIndexList()

    # Populate the MultiIndexList using boxes_list data
    box_idx = 0
    for world_idx in range(num_worlds):
      for i in range(num_boxes_per_world):
        boxes[world_idx, i] = boxes_list[box_idx]
        box_idx += 1

    brute_force_overlaps = find_overlaps_brute_force_batched(
      num_worlds, num_boxes_per_world, boxes
    )

    # Test the broad phase by setting custom aabb data
    d.geom_aabb = wp.array(
      boxes_list, dtype=wp.types.matrix(shape=(2, 3), dtype=wp.float32)
    )
    d.geom_aabb = d.geom_aabb.reshape((num_worlds, num_boxes_per_world))

    mjx.broad_phase(m, d)

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
