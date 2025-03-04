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
import itertools
from typing import Any

import warp as wp
import numpy as np

from .types import Contact, Data, Model
import math


LARGE_VAL = 1e6

BOX_VERTS = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))), dtype=float)

BOX_FACES = np.array([
      0, 4, 5, 1,
      0, 2, 6, 4,
      6, 7, 5, 4,
      2, 3, 7, 6,
      1, 5, 7, 3,
      0, 1, 3, 2,
  ]).reshape((-1, 4))

BOX_NORMALS = np.array(
    [[ 0, -1,  0],
     [ 0,  0, -1],
     [ 1,  0,  0],
     [ 0,  1,  0],
     [ 0,  0,  1],
     [-1,  0,  0]]).reshape((-1, 3))


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
    pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
    pass


class mat38f(wp.types.matrix(shape=(3, 8), dtype=wp.float32)):
    pass

@wp.struct
class Box:
    verts: mat83f


@wp.func
def box(R: wp.mat33) -> Box:
  """Get a transformed box"""
  x, y, z = m.geom_size[geom_idx, 0], m.geom_size[geom_idx, 1], m.geom_size[geom_idx, 2]
  t = wp.vec3(0.0)
  m = mat38f(
      R @ wp.vec3(-x, -y, -z) + t,
      R @ wp.vec3(-x, -y, +z) + t,
      R @ wp.vec3(-x, +y, -z) + t,
      R @ wp.vec3(-x, +y, +z) + t,
      R @ wp.vec3(+x, -y, -z) + t,
      R @ wp.vec3(+x, -y, +z) + t,
      R @ wp.vec3(+x, +y, -z) + t,
      R @ wp.vec3(+x, +y, +z) + t,
  )
  return Box(wp.transpose(m))


@wp.func
def get_axis(
    axis_idx: int,
    R: wp.mat33,
) -> tuple[wp.vec3, bool]:
    """Get the axis at index axis_idx.
    R: rotation matrix from a to b
    Axes 0-12 are face normals of boxes a & b
    Axes 12-21 are edge cross products."""
    if axis_idx < 6: # a faces
        axis = R @ wp.vec3(BOX_NORMALS[axis_idx])
        is_degenerate = False
    elif axis_idx < 12: # b faces
        axis = wp.vec3(BOX_NORMALS[axis_idx-6])
        is_degenerate = False
    else: # edges cross products
        assert axis_idx < 21
        edges = axis_idx - 12
        axis_a, axis_b = edges / 3, edges % 3
        edge_a = R[axis_a]
        if axis_b == 0:
            axis = wp.vec3(0.0, -edge_a[2], edge_a[1])
        elif axis_b == 1:
            axis = wp.vec3(edge_a[2], 0.0, -edge_a[0])
        else:
            axis = wp.vec3(-edge_a[1], edge_a[0], 0.0)
        is_degenerate = wp.length_sq(axis) < 1e-6
    return wp.normalize(axis), is_degenerate


@wp.func
def get_box_axis_support(axis: wp.vec3, degenerate_axis: bool, a: Box, b: Box) -> tuple[wp.float32, bool]:
    """Get the overlap (or separating distance if negative) along `axis`, and the sign."""
    axis_d = wp.vec3d(axis)
    support_a_max, support_b_max = wp.float32(-LARGE_VAL), wp.float32(-LARGE_VAL)
    support_a_min, support_b_min = wp.float32(LARGE_VAL), wp.float32(LARGE_VAL)
    for i in range(8):
      vert_a = wp.vec3d(a.verts[i])
      vert_b = wp.vec3d(b.verts[i])
      proj_a = wp.float32(wp.dot(vert_a, axis_d))
      proj_b = wp.float32(wp.dot(vert_b, axis_d))
      support_a_max = wp.max(support_a_max, proj_a)
      support_b_max = wp.max(support_b_max, proj_b)
      support_a_min = wp.min(support_a_min, proj_a)
      support_b_min = wp.min(support_b_min, proj_b)
    dist1 = support_a_max - support_b_min
    dist2 = support_b_max - support_a_min
    dist = wp.select(degenerate_axis, wp.min(dist1, dist2), LARGE_VAL)
    sign = wp.select(dist1 > dist2, 1, -1)
    return dist, sign


@wp.struct
class AxisSupport:
    best_dist: wp.float32
    best_sign: wp.int8
    best_idx: wp.int8


@wp.func
def reduce_support_axis(a: AxisSupport, b: AxisSupport):
  return wp.select(a.best_dist > b.best_dist, a, b)


@wp.func
def collision_axis_tiled(
    m: Model,
    d: Data,
    a: Box,
    b: Box,
    R: wp.array(dtype=wp.mat33),
    axis_idx: int,
) -> tuple[wp.vec3, wp.int8]:
  """Finds the axis of minimum separation or maximum overlap.
  Returns:
    best_axis: vec3
    best_idx: int8
  """
  # launch tiled with block_dim=21

  axis, degenerate_axis = get_axis(axis_idx, R)
  axis_dist, axis_sign = get_box_axis_support(axis, degenerate_axis, a, b)
  
  supports = wp.tile(AxisSupport(axis_dist, wp.int8(axis_sign), wp.int8(axis_idx)))

  face_supports = wp.tile_view(supports, offset=(0,), shape=(12,))
  edge_supports = wp.tile_view(supports, offset=(12,), shape=(9,))

  # TODO(ca): handle untile & tile usage
  best_face = wp.untile(wp.tile_broadcast(
      wp.tile_reduce(reduce_support_axis, face_supports),
      shape=(1,21)))
  best_edge = wp.untile(wp.tile_broadcast(
      wp.tile_reduce(reduce_support_axis, edge_supports),
      shape=(1,21)))
  
  if axis_idx > 0:
      return
  
  face_axis = get_axis(best_face.best_idx)
  best_axis = wp.vec3(face_axis)
  if best_edge.dist < best_face.dist:
      edge_axis = get_axis(best_edge.idx)
      if wp.abs(wp.dot(face_axis, edge_axis)) < 0.99:
          best_axis = edge_axis


  # get the (reference) face most aligned with the separating axis
  return wp.vec3(0.0)


@wp.func
def create_contact_manifold(
  m: Model,
  d: Data,
  ) -> tuple[wp.float32, wp.vec3, wp.mat33]:
  pass


@wp.kernel
def box_box_kernel(
  m: Model,
  d: Data,
  group_key: int,
  num_kernels: int,
):
  """Calculates contacts between pairs of boxes."""
  thread_idx, axis_idx = wp.tid()

  num_candidate_contacts = d.narrowphase_candidate_group_count[group_key]

  for bp_idx in range(thread_idx, num_candidate_contacts, num_kernels):
    a_idx, b_idx = d.narrowphase_candidate_geom[group_key, bp_idx]
    world_id = d.narrowphase_candidate_worldid[group_key, bp_idx]

    # transformations
    a_pos, b_pos = d.geom_xpos[world_id, a_idx], d.geom_xpos[world_id, b_idx]
    a_mat, b_mat = d.geom_xmat[world_id, a_idx], d.geom_xmat[world_id, b_idx]
    b_mat_inv = wp.transpose(b_mat)
    trans_atob = b_mat_inv @ (a_pos - b_pos)
    rot_atob = b_mat_inv @ a_mat

    a = box(m, a_idx, rot_atob, trans_atob)
    b = box(m, a_idx, wp.identity(3), wp.vec3(0.0))


    # TODO(ca):
    # - faces compute from verts

    # box-box implementation
    sep_axis, sep_axis_idx = collision_axis_tiled(m, d, rot_atob, a, b, axis_idx)
    dist, pos, normal = create_contact_manifold(m, d, axis_idx)

    # generate contact w/ 1 thread per pair
    if axis_idx != 0:
      return

    # if contact
    if dist < 0:
      contact_idx = wp.atomic_add(d.contact_counter, 0, 1)
      # TODO(ca): multi-dimensional contacts
      d.contact.dist[contact_idx] = dist
      d.contact.pos[contact_idx] = b_pos + b_mat @ pos
      d.contact.frame[contact_idx] = make_frame(b_mat @ normal)
      d.contact.worldid[contact_idx] = world_id


def box_box(
  m: Model,
  d: Data,
  group_key: int,
):
  """Calculates contacts between pairs of boxes."""
  kernel_ratio = 16
  num_kernels = math.ceil(d.nconmax / kernel_ratio)  # parallel kernels excluding tile dim
  wp.launch_tiled(
    kernel=box_box_kernel,
    dim=num_kernels,
    inputs=[m, d, group_key, num_kernels],
    block_dim=21,
  )


@wp.func
def _closest_segment_point_plane(
    a: wp.vec3, b: wp.vec3, p0: wp.vec3, plane_normal: wp.vec3
) -> wp.vec3:
  """Gets the closest point between a line segment and a plane.

  Args:
    a: first line segment point
    b: second line segment point
    p0: point on plane
    plane_normal: plane normal

  Returns:
    closest point between the line segment and the plane
  """
  # Parametrize a line segment as S(t) = a + t * (b - a), plug it into the plane
  # equation dot(n, S(t)) - d = 0, then solve for t to get the line-plane
  # intersection. We then clip t to be in [0, 1] to be on the line segment.
  n = plane_normal
  d = wp.dot(p0, n)  # shortest distance from origin to plane
  denom = wp.dot(n, (b - a))
  t = (d - wp.dot(n, a)) / (denom + 1e-6 * wp.select(denom == 0.0, 1.0, 0.0))
  t = wp.clamp(t, 0.0, 1.0)
  segment_point = a + t * (b - a)

  return segment_point


@wp.kernel
def _clip_edge_to_poly(
    subject_poly: wp.array(dtype=wp.vec3, ndim=2),
    subject_poly_length: wp.array(dtype=wp.int32, ndim=1),
    clipping_poly: wp.array(dtype=wp.vec3, ndim=2),
    clipping_poly_length: wp.array(dtype=wp.int32, ndim=1),
    clipping_normal: wp.array(dtype=wp.vec3, ndim=1),
    clipped_points_offset: wp.array(dtype=wp.int32, ndim=1),
    # outputs
    clipped_points: wp.array(dtype=wp.vec3, ndim=2),
    mask: wp.array(dtype=wp.int32, ndim=2),
    clipped_points_length: wp.array(dtype=wp.int32, ndim=1),
):
    assert clipped_points.shape[1] == subject_poly.shape[1] + clipping_poly.shape[1]
    assert clipped_points.shape[1] == mask.shape[1]

    poly_idx, edge_idx = wp.tid()
    assert subject_poly_length[poly_idx] > 1, "subject_poly_length must be > 1"
    assert clipping_poly_length[poly_idx] > 1, "clipping_poly_length must be > 1"
    n_subj = subject_poly_length[poly_idx]
@wp.func
def _clip_edge_to_quad(
    subject_poly: mat43f,
    clipping_poly: mat43f,
    clipping_normal: wp.vec3,
):

  for edge_idx in range(4):
    subject_p0 = subject_poly[(edge_idx + 3) % 4]
    subject_p1 = subject_poly[edge_idx]

    any_both_in_front = wp.int32(0)
    clipped0_dist_max = wp.float32(-1e6)
    clipped1_dist_max = wp.float32(-1e6)
    clipped_p0_distmax = wp.vec3(0.0)
    clipped_p1_distmax = wp.vec3(0.0)

    for clipping_edge_idx in range(4):
      clipping_p0 = clipping_poly[(clipping_edge_idx + 3) % 4]
      clipping_p1 = clipping_poly[clipping_edge_idx]
      edge_normal = wp.cross(clipping_p1 - clipping_p0, clipping_normal)

      p0_in_front = wp.dot(subject_p0 - clipping_p0, edge_normal) > 1e-6
      p1_in_front = wp.dot(subject_p1 - clipping_p0, edge_normal) > 1e-6
      wp.printf("Edge %d plane %d: p0if: %d, p1if: %d\n", edge_idx, clipping_edge_idx, p0_in_front, p1_in_front)
      candidate_clipped_p = _closest_segment_point_plane(subject_p0, subject_p1, clipping_p1, edge_normal)
      clipped_p0 = wp.select(p0_in_front, subject_p0, candidate_clipped_p)
      clipped_p1 = wp.select(p1_in_front, subject_p1, candidate_clipped_p)
      clipped_dist_p0 = wp.dot(clipped_p0 - subject_p0, subject_p1 - subject_p0)
      clipped_dist_p1 = wp.dot(clipped_p1 - subject_p1, subject_p0 - subject_p1)
      any_both_in_front |= wp.int32(p0_in_front and p1_in_front)

      if clipped_dist_p0 > clipped0_dist_max:
        clipped0_dist_max = clipped_dist_p0
        clipped_p0_distmax = clipped_p0

      if clipped_dist_p1 > clipped1_dist_max:
        clipped1_dist_max = clipped_dist_p1
        clipped_p1_distmax = clipped_p1
    # new_p0 = wp.select(any_both_in_front, clipped_p0_distmax, subject_p0)
    # new_p1 = wp.select(any_both_in_front, clipped_p1_distmax, subject_p1)
    new_p0 = clipped_p0_distmax
    new_p1 = clipped_p1_distmax

    mask_val = wp.int8(wp.select(
        wp.dot(subject_p0 - subject_p1, new_p0 - new_p1) < 0,
        wp.int32(not any_both_in_front),
        0))

    if edge_idx == 0:
       clipped_p0_0 = new_p0
       clipped_p1_0 = new_p1
       mask_p0_0 = mask_val
    elif edge_idx == 1:
       clipped_p0_1 = new_p0
       clipped_p1_1 = new_p1
       mask_p0_1 = mask_val
    elif edge_idx == 2:
       clipped_p0_2 = new_p0
       clipped_p1_2 = new_p1
       mask_p0_2 = mask_val
    else:
       clipped_p0_3 = new_p0
       clipped_p1_3 = new_p1
       mask_p0_3 = mask_val
  return (
      wp.transpose(mat34f(clipped_p0_0, clipped_p0_1, clipped_p0_2, clipped_p0_3)),
      wp.transpose(mat34f(clipped_p1_0, clipped_p1_1, clipped_p1_2, clipped_p1_3)),
      wp.vec4b(mask_p0_0, mask_p0_1, mask_p0_2, mask_p0_3),
      )
  


@wp.kernel
def _project_poly_onto_poly_plane(
    poly1: wp.array(dtype=wp.vec3, ndim=2),
    poly1_length: wp.array(dtype=wp.int32, ndim=1),
    poly1_norm: wp.array(dtype=wp.vec3, ndim=1),
    poly2: wp.array(dtype=wp.vec3, ndim=2),
    poly2_norm: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    new_poly: wp.array(dtype=wp.vec3, ndim=2),
):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  poly_idx = wp.tid()
  d = wp.dot(poly2[poly_idx, 0], poly2_norm[poly_idx])
  denom = wp.dot(poly1_norm[poly_idx], poly2_norm[poly_idx])
  for i in range(poly1_length[poly_idx]):
    t = (d - wp.dot(poly1[poly_idx, i], poly2_norm[poly_idx])) / (denom + wp.select(denom == 0.0, 0.0, 1e-6))
    new_poly[poly_idx, i] = poly1[poly_idx, i] + t * poly1_norm[poly_idx]


@wp.kernel
def _project_points_onto_plane(
    points: wp.array(dtype=wp.vec3, ndim=2),
    n_points: wp.array(dtype=wp.int32, ndim=1),
    plane_point: wp.array(dtype=wp.vec3, ndim=1),
    plane_normal: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    projected_points: wp.array(dtype=wp.vec3, ndim=2),
):
  """Projects points onto a plane using the plane normal."""
  row_idx = wp.tid()
  for i in range(n_points[row_idx]):
    dist = wp.dot(points[row_idx, i] - plane_point[row_idx], plane_normal[row_idx])
    projected_points[row_idx, i] = points[row_idx, i] - dist * plane_normal[row_idx]


@wp.kernel
def _points_in_front_of_plane(
    points: wp.array(dtype=wp.vec3, ndim=2),
    n_points: wp.array(dtype=wp.int32, ndim=1),
    plane_pt: wp.array(dtype=wp.vec3, ndim=1),
    plane_normal: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    mask: wp.array(dtype=wp.int32, ndim=2),
):
  """Checks if a set of points are strictly in front of a plane."""
  idx = wp.tid()
  for i in range(n_points[idx]):
    dist = wp.dot(points[idx, i] - plane_pt[idx], plane_normal[idx])
    mask[idx, i] = wp.int32(dist > 1e-6)


def _clip(
    clipping_poly: wp.array(dtype=wp.vec3, ndim=2),
    clipping_poly_length: wp.array(dtype=wp.int32, ndim=1),
    clipping_normal: wp.array(dtype=wp.vec3, ndim=1),
    subject_poly: wp.array(dtype=wp.vec3, ndim=2),
    subject_poly_length: wp.array(dtype=wp.int32, ndim=1),
    subject_normal: wp.array(dtype=wp.vec3, ndim=1),
) -> tuple[wp.array(dtype=wp.vec3, ndim=2), wp.array(dtype=wp.int32, ndim=2), wp.array(dtype=wp.int32, ndim=1)]:
  """Clips a subject polygon against a clipping polygon.

  A parallelized clipping algorithm for convex polygons. The result is a sequence of
  vertices on the clipped subject polygon in the subject polygon plane.

  Args:
    clipping_poly: the polygon that we use to clip the subject polygon against
    clipping_poly_length: number of vertices in the clipping polygon
    clipping_normal: normal of the clipping polygon
    subject_poly: the polygon that gets clipped
    subject_poly_length: number of vertices in the subject polygon
    subject_normal: normal of the subject polygon

  Returns:
    clipped_pts: points on the clipped polygon
    mask: True if a point is in the clipping polygon, False otherwise
    clipped_poly_length: number of points in the clipped poly
  """
  n_polys = subject_poly.shape[0]
  n_subject_verts = subject_poly.shape[1]
  n_clipping_verts = clipping_poly.shape[1]
  n_total_verts = n_subject_verts + n_clipping_verts

  # Clip all edges of the subject poly against clipping side planes.
  clipped_points = wp.empty((n_polys, 2*n_total_verts), dtype=wp.vec3)

  clipped_poly_length = wp.empty_like(clipping_poly_length)
  masks = wp.empty((n_polys, 2*n_total_verts), dtype=wp.int32)
  clipped_points_offset_zero = wp.zeros((1), dtype=wp.int32)
  wp.launch(
      kernel=_clip_edge_to_poly,
      dim=subject_poly.shape,
      inputs=[subject_poly, subject_poly_length, clipping_poly, clipping_poly_length, clipping_normal, clipped_points_offset_zero],
      outputs=[clipped_points, masks, clipped_poly_length],
  )

  # Project the clipping poly onto the subject plane.
  clipping_poly_s = wp.empty_like(clipping_poly)
  wp.launch(
      kernel=_project_poly_onto_poly_plane,
      dim=subject_poly.shape[0],
      inputs=[clipping_poly, clipped_poly_length, clipping_normal, subject_poly, subject_normal],
      outputs=[clipping_poly_s],
  )

  # Clip all edges of the clipping poly against subject planes.
  wp.launch(
      kernel=_clip_edge_to_poly,
      dim=subject_poly.shape,
      inputs=[clipping_poly_s, clipping_poly_length, subject_poly, subject_poly_length, subject_normal, subject_poly_length],
      outputs=[clipped_points, masks, clipped_poly_length],
  )

  return clipped_points, masks, clipped_poly_length


