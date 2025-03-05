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


class mat16_3f(wp.types.matrix(shape=(16, 3), dtype=wp.float32)):
    pass


class mat34f(wp.types.matrix(shape=(3, 4), dtype=wp.float32)):
    pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
    pass


class mat38f(wp.types.matrix(shape=(3, 8), dtype=wp.float32)):
    pass


class vec8b(wp.types.vector(length=8, dtype=wp.int8)):
    pass


class vec16b(wp.types.vector(length=16, dtype=wp.int8)):
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


@wp.func
def _project_quad_onto_plane(
    quad: mat43f,
    quad_n: wp.vec3,
    plane_n: wp.vec3,
    plane_pt: wp.vec3,
):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  d = wp.dot(plane_pt, plane_n)
  denom = wp.dot(quad_n, plane_n)
  qn_scaled = quad_n / (denom + wp.select(denom == 0.0, 1e-6, 0.0))

  for i in range(4):
    quad[i] = quad[i] + (d - wp.dot(quad[i], plane_n)) * qn_scaled
  return quad


@wp.func
def _project_poly_onto_plane(
    poly: Any,
    poly_n: wp.vec3,
    plane_n: wp.vec3,
    plane_pt: wp.vec3,
):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  d = wp.dot(plane_pt, plane_n)
  denom = wp.dot(poly_n, plane_n)
  qn_scaled = poly_n / (denom + wp.select(denom == 0.0, 1e-6, 0.0))

  for i in range(len(poly)):
    poly[i] = poly[i] + (d - wp.dot(poly[i], plane_n)) * qn_scaled
  return poly


@wp.func
def _clip_edge_to_quad(
    subject_poly: mat43f,
    clipping_poly: mat43f,
    clipping_normal: wp.vec3,
):

  p0 = mat43f()
  p1 = mat43f()
  mask = wp.vec4b()
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
    new_p0 = wp.select(any_both_in_front, clipped_p0_distmax, subject_p0)
    new_p1 = wp.select(any_both_in_front, clipped_p1_distmax, subject_p1)

    mask_val = wp.int8(wp.select(
        wp.dot(subject_p0 - subject_p1, new_p0 - new_p1) < 0,
        wp.int32(not any_both_in_front),
        0))

    p0[edge_idx] = new_p0
    p1[edge_idx] = new_p1
    mask[edge_idx] = mask_val
  return p0, p1, mask


@wp.func
def _clip_quad(
    subject_quad: mat43f,
    subject_normal: wp.vec3,
    clipping_quad: mat43f,
    clipping_normal: wp.vec3,
): #  -> tuple[mat83f, vec8b]
  """Clips a subject quad against a clipping quad.
  Serial implementation.
  """

  subject_clipped_p0, subject_clipped_p1, subject_mask = _clip_edge_to_quad(subject_quad, clipping_quad, clipping_normal)
  clipping_proj = _project_poly_onto_plane(clipping_quad, clipping_normal, subject_normal, subject_quad[0])
  clipping_clipped_p0, clipping_clipped_p1, clipping_mask = _clip_edge_to_quad(clipping_proj, subject_quad, subject_normal)

  clipped = mat16_3f()
  mask = vec16b()
  for i in range(4):
    clipped[i  ] = subject_clipped_p0[i]
    clipped[i+4] = clipping_clipped_p0[i]
    clipped[i+8] = subject_clipped_p1[i]
    clipped[i+12] = clipping_clipped_p1[i]
    mask[i] = subject_mask[i]
    mask[i+4] = clipping_mask[i]
    mask[i+8] = subject_mask[i]
    mask[i+8+4] = clipping_mask[i]

  return clipped, mask


# @wp.kernel
# def _manifold_points_kernel(
#     poly: wp.array(dtype=wp.vec3, ndim=2),
#     poly_mask: wp.array(dtype=wp.int32, ndim=2),
#     poly_norm: wp.array(dtype=wp.vec3, ndim=1),
#     n_poly_verts: wp.array(dtype=wp.int32, ndim=1),
#     dist_mask: wp.array(dtype=wp.float32, ndim=2),
#     # outputs
#     points: wp.array(dtype=wp.int32, ndim=2),
#   ):
#   poly_idx = wp.tid()
#   polygon = poly[poly_idx]
# 
#   n_points = n_poly_verts[poly_idx]
# 
#   for i in range(n_points):
#     dist_mask[poly_idx, i] = wp.select(poly_mask[poly_idx, i], -1e6, 0.0)
# 
#   a_idx = _argmax(dist_mask[poly_idx])
#   a = polygon[a_idx]
# 
#   # choose point b furthest from a
#   b_dist = wp.float32(-1e6)
#   b_idx = wp.int32(0)
#   for i in range(n_points):
#     b_cand_dist = wp.length_sq(a-polygon[i]) + dist_mask[poly_idx, i]
#     if b_cand_dist > b_dist:
#         b_idx = i
#         b_dist = b_cand_dist
# 
#   b = polygon[b_idx]
# 
#   # choose point c furthest along the axis orthogonal to (a-b)
#   c_dist = wp.float32(-1e6)
#   c_idx = wp.int32(0)
#   ab = wp.cross(poly_norm[poly_idx], a - b)
#   for i in range(n_points):
#     ap = a - polygon[i]
#     c_cand_dist = wp.abs(wp.dot(ap, ab)) + dist_mask[poly_idx, i]
#     if c_cand_dist > c_dist:
#         c_idx = i
#         c_dist = c_cand_dist
# 
#   c = polygon[c_idx]
# 
#   # choose point d furthest from the other two triangle edges
#   d_dist = wp.float32(-1e6)
#   d_idx = wp.int32(0)
#   ac = wp.cross(poly_norm[poly_idx], a - c)
#   bc = wp.cross(poly_norm[poly_idx], b - c)
#   for i in range(n_points):
#     dist_bp = wp.abs(wp.dot(b-polygon[i], bc)) + dist_mask[poly_idx, i]
#     dist_ap = wp.abs(wp.dot(a-polygon[i], ac)) + dist_mask[poly_idx, i]
#     d_cand_dist = dist_bp + dist_ap
#     if d_cand_dist > d_dist:
#         d_idx = i
#         d_dist = d_cand_dist
# 
#   points[poly_idx, 0] = a_idx
#   points[poly_idx, 1] = b_idx
#   points[poly_idx, 2] = c_idx
#   points[poly_idx, 3] = d_idx


# TODO(ca): sparse, tiling variant for large point counts
@wp.func
def _manifold_points(
    poly: Any,
    mask: Any,
    clipping_norm: wp.vec3,
) -> wp.vec4b:
  """Chooses four points on the polygon with approximately maximal area. Return the indices"""
  n = len(poly)

  a_idx = wp.int32(0)
  a_mask = wp.int8(mask[0]);
  for i in range(n):
    if mask[i] >= a_mask:
      a_idx = i
      a_mask = mask[i]
  a = poly[a_idx]

  b_idx = wp.int32(0)
  b_dist = wp.float32(-1e6)
  for i in range(n):
    dist = wp.length_sq(poly[i] - a) + wp.select(mask[i], -1e6, 0.0)
    if dist >= b_dist:
      b_idx = i
      b_dist = dist
  b = poly[b_idx]

  ab = wp.cross(clipping_norm, a-b)

  c_idx = wp.int32(0)
  c_dist = wp.float32(-1e6)
  for i in range(n):
    ap = a - poly[i]
    dist = wp.abs(wp.dot(ap, ab)) + wp.select(mask[i], -1e6, 0.0)
    if dist >= c_dist:
      c_idx = i
      c_dist = dist
  c = poly[c_idx]

  ac = wp.cross(clipping_norm, a - c)
  bc = wp.cross(clipping_norm, b - c)

  d_idx = wp.int32(-2e6)
  d_dist = wp.float32(0)
  for i in range(n):
    ap = a - poly[i]
    dist_ap = wp.abs(wp.dot(ap, bc)) + wp.select(mask[i], -1e6, 0.0)
    bp = b - poly[i]
    dist_bp = wp.abs(wp.dot(bp, bc)) + wp.select(mask[i], -1e6, 0.0)
    if dist_ap + dist_bp >= d_dist:
      d_idx = i
      d_dist = dist_ap + dist_bp
  d = poly[d_idx]
  return wp.vec4b(wp.int8(a_idx), wp.int8(b_idx), wp.int8(c_idx), wp.int8(d_idx))


@wp.func
def _create_contact_manifold(
    clipping_quad: mat43f,
    clipping_normal: wp.vec3,
    subject_quad: mat43f,
    subject_normal: wp.vec3,
    sep_axis: wp.vec3,
    ): # -> tuple[ wp.vec3, wp.vec3, wp.vec3]
  # Clip the subject (incident) face onto the clipping (reference) face.
  # The incident points are clipped points on the subject polygon.
  incident, mask = _clip_quad(subject_quad, subject_normal, clipping_quad, clipping_normal)
  
  clipping_normal_neg = -clipping_normal
  d = wp.dot(clipping_quad[0], clipping_normal_neg) + 1e-6

  for i in range(16):
    if wp.dot(incident[i], clipping_normal_neg) > d:
      mask[i] = wp.int8(0)
    
  ref = _project_poly_onto_plane(incident, clipping_normal, clipping_normal, clipping_quad[0])

  # # Choose four contact points.
  best = _manifold_points(ref, mask, clipping_normal)
  contact_pts = mat43f()
  dist = wp.vec4f()

  for i in range(4):
    idx = wp.int32(best[i])
    contact_pt = ref[idx]
    contact_pts[i] = contact_pt
    penetration_dir = incident[idx] - contact_pt
    penetration = wp.dot(penetration_dir, clipping_normal_neg)
    dist[i] = wp.select(mask[idx], 1.0, -penetration)

  return dist, contact_pts


@wp.kernel
def _and_mask(
    a: wp.array(dtype=wp.int32, ndim=2),
    b: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    result: wp.array(dtype=wp.int32, ndim=2),
):
  """Computes the element-wise AND of two masks. Output mask can be the same as one of the inputs."""
  row = wp.tid()
  for j in range(a.shape[1]):
    result[row, j] = wp.int32(a[row, j] and b[row, j])


@wp.kernel
def _create_manifold_post_process(
    poly_ref: wp.array(dtype=wp.vec3, ndim=2),
    poly_incident: wp.array(dtype=wp.vec3, ndim=2),
    clipping_normal: wp.array(dtype=wp.vec3, ndim=2),
    best: wp.array(dtype=wp.int32, ndim=2),
    mask: wp.array(dtype=wp.int32, ndim=2),
    n_poly_verts: wp.array(dtype=wp.int32, ndim=1),
    sep_axis: wp.array(dtype=wp.vec3, ndim=1),
    # outputs
    contact_pts: wp.array(dtype=wp.vec3, ndim=2),
    dist: wp.array(dtype=wp.float32, ndim=2),
    normal: wp.array(dtype=wp.vec3, ndim=2),
):
  poly_idx = wp.tid()
  n_verts = n_poly_verts[poly_idx]
  ref = poly_ref[poly_idx, 0]
  for i in range(4):
    point_idx = best[poly_idx, i]
    contact_pts[poly_idx, i] = poly_ref[poly_idx, point_idx]
    penetration_dir = poly_incident[poly_idx, point_idx] - poly_ref[poly_idx, point_idx]
    penetration = wp.dot(penetration_dir, -clipping_normal[poly_idx, point_idx])

    dist[poly_idx, i] = wp.select(mask[poly_idx, point_idx], 1.0, -penetration)
    normal[poly_idx, i] = -sep_axis[poly_idx]
