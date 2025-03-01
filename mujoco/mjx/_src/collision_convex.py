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
from typing import Any

import warp as wp

from .types import Contact, Data, Model


@wp.func
def get_axis(
    axis_idx: int,
    normals_a: wp.array(dtype=wp.vec3, ndim=1),
    normals_b: wp.array(dtype=wp.vec3, ndim=1),
    R: wp.mat33,
) -> tuple[wp.vec3, bool]:
    """Get the axis at index axis_idx.
    R: rotation matrix between box a and b
    Axes 0-12 are face normals of boxes a & b
    Axes 12-21 are edge cross products."""
    if axis_idx < 6: # a faces
        axis = normals_a[axis_idx]
        is_degenerate = False
    elif axis_idx < 12: # b faces
        axis = normals_b[axis_idx-6]
        is_degenerate = False
    else: # edges
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


def box_box(
  m: Model,
  d: Data,
  worldId: wp.array(dtype=wp.int32, ndim=1),
  planeIndex: wp.array(dtype=wp.int32, ndim=1),
  convexIndex: wp.array(dtype=wp.int32, ndim=1),
  outBaseIndex: wp.array(dtype=wp.int32, ndim=1),
  result: Contact,
):
  """Calculates contacts between pairs of boxes."""
@wp.func
def _argmax(a: wp.array(dtype=Any)) -> wp.int32:
    m = type(a[0])(a[0])
    am = wp.int32(0)
    for i in range(a.shape[0]):
      if a[i] > m:
          m = a[i]
          am = i
    return am


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

    any_both_in_front = wp.int32(0)
    clipped0_dist_max = wp.float32(-1e6)
    clipped1_dist_max = wp.float32(-1e6)
    clipped_p0_distmax = wp.vec3(0.0)
    clipped_p1_distmax = wp.vec3(0.0)

    if edge_idx < subject_poly_length[poly_idx]:
      subject_p0 = subject_poly[poly_idx, (edge_idx - 1 + n_subj) % n_subj]
      subject_p1 = subject_poly[poly_idx, edge_idx]
      for clipping_edge_idx in range(clipping_poly_length[poly_idx]):
        n_clip = clipping_poly_length[poly_idx]
        clipping_p0 = clipping_poly[poly_idx, (clipping_edge_idx - 1 + n_clip) % n_clip]
        clipping_p1 = clipping_poly[poly_idx, clipping_edge_idx]
        edge_normal = wp.cross(clipping_p1 - clipping_p0, clipping_normal[poly_idx])

        p0_in_front = wp.dot(subject_p0 - clipping_p0, edge_normal) > 1e-6
        p1_in_front = wp.dot(subject_p1 - clipping_p0, edge_normal) > 1e-6
        candidate_clipped_p = _closest_segment_point_plane(subject_p0, subject_p1, clipping_p0, edge_normal)
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
      if poly_idx < clipping_poly.shape[0]:
        offset = clipped_points_offset[poly_idx]
      else:
        offset = 0

      clipped_point_index = 2*(edge_idx + offset)
      clipped_points[poly_idx, clipped_point_index] = new_p0
      clipped_points[poly_idx, clipped_point_index+1] = new_p1
      mask_val = wp.select(
          wp.dot(subject_p0 - subject_p1, new_p0 - new_p1) < 0,
          wp.int32(not any_both_in_front),
          0)
      mask[poly_idx, clipped_point_index] = mask_val
      mask[poly_idx, clipped_point_index+1] = mask_val
      if edge_idx == 0:
        clipped_points_length[poly_idx] = 2*(offset + subject_poly_length[poly_idx])


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


