import numpy as np
from jax import numpy as jp
import jax
import warp as wp

def _manifold_points(
    poly: jax.Array, poly_mask: jax.Array, poly_norm: jax.Array
) -> jax.Array:
  """Chooses four points on the polygon with approximately maximal area."""
  dist_mask = jp.where(poly_mask, 0.0, -1e6)
  a_idx = jp.argmax(dist_mask)
  a = poly[a_idx]
  # choose point b furthest from a
  b_idx = (((a - poly) ** 2).sum(axis=1) + dist_mask).argmax()
  b = poly[b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = jp.cross(poly_norm, a - b)
  ap = a - poly
  c_idx = (jp.abs(ap.dot(ab)) + dist_mask).argmax()
  c = poly[c_idx]
  # choose point d furthest from the other two triangle edges
  ac = jp.cross(poly_norm, a - c)
  bc = jp.cross(poly_norm, b - c)
  bp = b - poly
  dist_bp = jp.abs(bp.dot(bc)) + dist_mask
  dist_ap = jp.abs(ap.dot(ac)) + dist_mask
  d_idx = (dist_bp + dist_ap).argmax() % poly.shape[0]
  return jp.array([a_idx, b_idx, c_idx, d_idx])

@wp.func
def sel(condition: bool, onTrue: float, onFalse: float) -> float:
  """Returns onTrue if condition is true, otherwise returns onFalse."""
  if condition:
    return onTrue
  else:
    return onFalse

@wp.func
def _manifold_points_wp(
    poly: wp.array(dtype=wp.vec3),
    poly_mask: wp.array(dtype=wp.float32),
    poly_count: int,
    poly_norm: wp.vec3
) -> wp.vec4i:
  """Chooses four points on the polygon with approximately maximal area."""
  max_val = float(-1e6)
  a_idx = int(0)
  for i in range(poly_count):
    val = sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      a_idx = i
  a = poly[a_idx]
  # choose point b furthest from a
  max_val = float(-1e6)
  b_idx = int(0)
  for i in range(poly_count):
    val = wp.length_sq(a - poly[i]) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      b_idx = i
  b = poly[b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = wp.cross(poly_norm, a - b)
  #ap = a - poly
  max_val = float(-1e6)
  c_idx = int(0)
  for i in range(poly_count):
    val = wp.abs(wp.dot(a - poly[i], ab)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      c_idx = i
  c = poly[c_idx]
  # choose point d furthest from the other two triangle edges
  ac = wp.cross(poly_norm, a - c)
  bc = wp.cross(poly_norm, b - c)
  #bp = b - poly
  max_val = float(-1e6)
  d_idx = int(0)
  for i in range(poly_count):
    val = wp.abs(wp.dot(b - poly[i], bc)) + wp.abs(wp.dot(a - poly[i], ac)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      d_idx = i
  return wp.vec4i(a_idx, b_idx, c_idx, d_idx)

@wp.kernel
def _manifold_points_kernel(
    poly: wp.array(dtype=wp.vec3),
    poly_mask: wp.array(dtype=wp.float32),
    poly_count: int,
    poly_norm: wp.vec3,
    result: wp.array(dtype=wp.vec4i)):
  result[0] = _manifold_points_wp(poly, poly_mask, poly_count, poly_norm)

def test_manifold_points():
    # Define a simple quadrilateral
    poly = jp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    poly_mask = jp.array([1, 1, 1, 1])  # All points are valid
    poly_norm = jp.array([0.0, 0.0, 1.0])  # Normal pointing in +Z direction

    indices = _manifold_points(poly, poly_mask, poly_norm)
    
    # Ensure we get four unique indices within the polygon range
    assert indices.shape == (4,)
    assert jp.all(indices >= 0)
    assert jp.all(indices < poly.shape[0])
    assert len(jp.unique(indices)) == 4  # Ensure all four points are unique
    print(indices)

def test_manifold_points_wp():
    # Define a simple quadrilateral
    poly = wp.array([
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(1.0, 1.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0)
    ], dtype = wp.vec3)
    poly_mask = wp.array([1.0, 1.0, 1.0, 1.0], dtype=wp.float32)  # All points are valid
    poly_count = 4
    poly_norm = wp.vec3(0.0, 0.0, 1.0)  # Normal pointing in +Z direction
    result = wp.zeros(1, dtype=wp.vec4i)

    # Launch the kernel
    wp.launch(
        kernel=_manifold_points_kernel,
        dim=1,
        inputs=[poly, poly_mask, poly_count, poly_norm],
        outputs=[result]
    )
    wp.synchronize()

    indices = result.numpy()[0]
    
    # Ensure we get four unique indices within the polygon range
    assert indices.shape == (4,)
    assert all(0 <= idx < poly_count for idx in indices)
    assert len(set(indices)) == 4  # Ensure all four points are unique
    print(indices)

test_manifold_points()
test_manifold_points_wp()
print("done")