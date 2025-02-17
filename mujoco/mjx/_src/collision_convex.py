import warp as wp

from . import types
from typing import Tuple, Iterator, Union

# Collision returned by collision functions:
#  - distance          distance between nearest points; neg: penetration
#  - position  (3,)    position of contact point: midpoint between geoms
#  - frame     (3, 3)  normal is in [0, :], points from geom[0] to geom[1]
@wp.struct
class Collision:
  """Collision data between two geoms."""
  distance: wp.array(dtype=wp.float32)  # distance between nearest points; neg: penetration
  position: wp.array(dtype=wp.vec3)     # position of contact point: midpoint between geoms 
  frame: wp.array(dtype=wp.mat33)       # normal is in [0, :], points from geom[0] to geom[1]

@wp.struct
class GeomInfo:
  """Geom properties for primitive shapes."""

  pos: wp.array(dtype=wp.vec3)
  mat: wp.array(dtype=wp.mat33)
  size: wp.array(dtype=wp.vec3)


@wp.struct
class ConvexInfo:
  """Geom properties for convex meshes."""

  pos: wp.array(dtype=wp.vec3)
  mat: wp.array(dtype=wp.mat33)
  size: wp.array(dtype=wp.vec3)
  vert: wp.array(dtype=wp.vec3)
  face: wp.array(dtype=wp.int32)
  face_normal: wp.array(dtype=wp.vec3)
  edge: wp.array(dtype=wp.int32)
  edge_face_normal: wp.array(dtype=wp.vec3)





@wp.func
def sel(condition: bool, onTrue: float, onFalse: float) -> float:
  """Returns onTrue if condition is true, otherwise returns onFalse."""
  if condition:
    return onTrue
  else:
    return onFalse

@wp.func
def _manifold_points(
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



@wp.func
def orthogonals(a: wp.vec3) -> tuple[wp.vec3, wp.vec3]:
  """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = sel(-0.5 < a[1] and a[1] < 0.5, y, z)
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
def plane_convex(index: int, plane: GeomInfo, convex: ConvexInfo, outBaseIndex : int, result : Collision):
  """Calculates contacts between a plane and a convex object."""
  vert = convex.vert

  # get points in the convex frame
  plane_pos = wp.transpose(convex.mat) @ (plane.pos[index] - convex.pos[index])
  n = wp.transpose(convex.mat) @ plane.mat[index, 2]
  support = (plane_pos - vert) @ n
  # search for manifold points within a 1mm skin depth
  idx = wp.vec4i(0)
  idx = _manifold_points(vert, support > wp.maximum(0.0, wp.max(support) - 1e-3), n)
  frame = make_frame(wp.vec3(plane.mat[0, 2], plane.mat[1, 2], plane.mat[2, 2]))

  # Initialize return value
  # ret = Collision4()

  for i in range(4):
    # Get vertex position and convert to world frame
    id = int(idx[i])
    pos_i = vert[id]
    pos_i = convex.pos + pos_i @ wp.transpose(convex.mat)

    # Compute uniqueness by comparing with previous indices
    count = 0
    for j in range(i+1):
      if idx[i] == idx[j]:
        count += 1
    unique = sel(count == 1, 1.0, 0.0)

    # Compute distance and final position
    dist_i = sel(unique > 0.0, -support[id], 1.0)
    pos_i = pos_i - 0.5 * dist_i * frame[2]

    # Store results
    result.distance[outBaseIndex + i] = dist_i
    result.position[outBaseIndex + i] = pos_i
    result.frame[outBaseIndex + i] = frame

  # return ret


@wp.kernel
def plane_convex_kernel(plane: wp.array(dtype=GeomInfo), convex: wp.array(dtype=ConvexInfo), result: Collision):
  id = wp.tid()
  plane_convex(id, plane, convex, 4*id, result)
  

def plane_convex_launch(m: types.Model, d: types.Data):
  
  num_geoms = m.ngeom
  infos = GeomInfo() # wp.array(dtype=GeomInfo, size=num_geoms)
  infos.pos = d.geom_xpos
  infos.mat = d.geom_xmat
  infos.size = m.geom_size

  # todo: only capture pairs that are actually plane and convex
  # for i in range(num_geoms):
  #   infos[i] = GeomInfo(d.geom_xpos[i], d.geom_xmat[i], m.geom_size[i])

  convex_infos = ConvexInfo() # wp.array(dtype=ConvexInfo, size=num_geoms)
  convex_infos.pos = d.geom_xpos
  convex_infos.mat = d.geom_xmat
  convex_infos.size = m.geom_size
  convex_infos.vert = d.geom_mesh_vert
  convex_infos.face = d.geom_mesh_face
  convex_infos.face_normal = d.geom_mesh_norm
  convex_infos.edge = d.geom_mesh_edge
  convex_infos.edge_face_normal = d.geom_mesh_edge_norm

  # for i in range(num_geoms):
  #   convex_infos[i] = ConvexInfo(d.geom_xpos[i], d.geom_xmat[i], m.geom_size[i], d.geom_mesh_vert[i], d.geom_mesh_face[i], d.geom_mesh_norm[i], d.geom_mesh_edge[i], d.geom_mesh_edge_norm[i])

  ret = Collision()
  ret.distance = wp.array(dtype=wp.float32, size=num_geoms)
  ret.position = wp.array(dtype=wp.vec3, size=num_geoms)
  ret.frame = wp.array(dtype=wp.mat33, size=num_geoms)

  wp.launch(kernel=plane_convex_kernel,
            grid=num_geoms,
            inputs=[infos, convex_infos],
            outputs=[ret])
  
  wp.synchronize()









print("end")