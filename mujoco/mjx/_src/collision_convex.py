import warp as wp

from . import types
from typing import Tuple, Iterator, Union

# Collision returned by collision functions:
#  - distance          distance between nearest points; neg: penetration
#  - position  (3,)    position of contact point: midpoint between geoms
#  - frame     (3, 3)  normal is in [0, :], points from geom[0] to geom[1]
@wp.struct
class Collision4:
  """Collision data between two geoms."""
  distance: wp.types.vector(4, dtype=wp.float32)  # distance between nearest points; neg: penetration
  position: wp.types.vector(4, dtype=wp.vec3)     # position of contact point: midpoint between geoms 
  frame: wp.types.vector(4, dtype=wp.mat33)       # normal is in [0, :], points from geom[0] to geom[1]

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
) -> wp.vec4:
  """Chooses four points on the polygon with approximately maximal area."""
  max_val = -1e6
  a_idx = 0
  for i in range(poly_count):
    val = sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      a_idx = i
  a = poly[a_idx]
  # choose point b furthest from a
  max_val = -1e6
  b_idx = 0
  for i in range(poly_count):
    val = wp.sum((a - poly[i]) * (a - poly[i])) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      b_idx = i
  b = poly[b_idx]
  # choose point c furthest along the axis orthogonal to (a-b)
  ab = wp.cross(poly_norm, a - b)
  ap = a - poly
  max_val = -1e6
  c_idx = 0
  for i in range(poly_count):
    val = wp.abs(wp.dot(ap[i], ab)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      c_idx = i
  c = poly[c_idx]
  # choose point d furthest from the other two triangle edges
  ac = wp.cross(poly_norm, a - c)
  bc = wp.cross(poly_norm, b - c)
  bp = b - poly
  max_val = -1e6
  d_idx = 0
  for i in range(poly_count):
    val = wp.abs(wp.dot(bp[i], bc)) + wp.abs(wp.dot(ap[i], ac)) + sel(poly_mask[i] > 0.0, 0.0, -1e6)
    if val > max_val:
      max_val = val
      d_idx = i
  return wp.vec4(a_idx, b_idx, c_idx, d_idx)



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
def plane_convex(index: int, plane: GeomInfo, convex: ConvexInfo) -> Collision4:
  """Calculates contacts between a plane and a convex object."""
  vert = convex.vert

  # get points in the convex frame
  plane_pos = wp.transpose(convex.mat) @ (plane.pos[index] - convex.pos[index])
  n = wp.transpose(convex.mat) @ plane.mat[index, 2]
  support = (plane_pos - vert) @ n
  # search for manifold points within a 1mm skin depth
  idx = wp.vec4(0.0)
  idx = _manifold_points(vert, support > wp.maximum(0.0, wp.max(support) - 1e-3), n)
  frame = make_frame(wp.vec3(plane.mat[0, 2], plane.mat[1, 2], plane.mat[2, 2]))

  # Initialize return value
  ret = Collision4()

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
    ret.distance[i] = dist_i
    ret.position[i] = pos_i
    ret.frame[i] = frame

  return ret








class GeomType:
  """Type of geometry.

  Members:
    PLANE: plane
    HFIELD: height field
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
    SDF: signed distance field
  """

  PLANE = 0
  HFIELD = 1
  SPHERE = 2
  CAPSULE = 3
  ELLIPSOID = 4
  CYLINDER = 5
  BOX = 6
  MESH = 7
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


# geoms for which we ignore broadphase
_GEOM_NO_BROADPHASE = {GeomType.HFIELD, GeomType.PLANE}



class FunctionKey:
  """Specifies how geom pairs group into collision_driver's function table.

  Attributes:
    types: geom type pair, which determines the collision function
    data_ids: geom data id pair: mesh id for mesh geoms, otherwise -1. Meshes
      have distinct face/vertex counts, so must occupy distinct entries in the
      collision function table.
    condim: grouping by condim of the colliision ensures that the size of the
      resulting constraint jacobian is determined at compile time.
    subgrid_size: the size determines the hfield subgrid to collide with
  """

  types: Tuple[int, int]
  data_ids: Tuple[int, int]
  condim: int
  subgrid_size: Tuple[int, int] = (-1, -1)



def geom_pairs(
    m: Union[types.Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int, int]]:
  """Yields geom pairs to check for collisions.

  Args:
    m: a MuJoCo or MJX model

  Yields:
    geom1, geom2, and pair index if defined in <pair> (else -1)
  """
  pairs = set()

  for i in range(m.npair):
    g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
    # order pairs by geom_type for correct function mapping
    if m.geom_type[g1] > m.geom_type[g2]:
      g1, g2 = g2, g1
    pairs.add((g1, g2))
    yield g1, g2, i

  exclude_signature = set(m.exclude_signature)
  geom_con = m.geom_contype | m.geom_conaffinity
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum

  for b1 in range(m.nbody):
    if not geom_con[b_start[b1] : b_end[b1]].any():
      continue
    w1 = m.body_weldid[b1]
    w1_p = m.body_weldid[m.body_parentid[w1]]

    for b2 in range(b1, m.nbody):
      if not geom_con[b_start[b2] : b_end[b2]].any():
        continue
      signature = (b1 << 16) + (b2)
      if signature in exclude_signature:
        continue
      w2 = m.body_weldid[b2]
      # ignore self-collisions
      if w1 == w2:
        continue
      w2_p = m.body_weldid[m.body_parentid[w2]]
      # ignore parent-child collisions
      if filterparent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
        continue
      g1_range = [g for g in range(b_start[b1], b_end[b1]) if geom_con[g]]
      g2_range = [g for g in range(b_start[b2], b_end[b2]) if geom_con[g]]

      for g1, g2 in itertools.product(g1_range, g2_range):
        t1, t2 = m.geom_type[g1], m.geom_type[g2]
        # order pairs by geom_type for correct function mapping
        if t1 > t2:
          g1, g2, t1, t2 = g2, g1, t2, t1
        # ignore plane<>plane and plane<>hfield
        if (t1, t2) == (GeomType.PLANE, GeomType.PLANE):
          continue
        if (t1, t2) == (GeomType.PLANE, GeomType.HFIELD):
          continue
        # geoms must match contype and conaffinity on some bit
        mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
        mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
        if not mask:
          continue

        if (g1, g2) not in pairs:
          pairs.add((g1, g2))
          yield g1, g2, -1

def _geom_groups(
    m: Union[Model, mujoco.MjModel],
) -> Dict[FunctionKey, List[Tuple[int, int, int]]]:
  """Returns geom pairs to check for collision grouped by collision function.

  The grouping consists of:
    - The collision function to run, which is determined by geom types
    - For mesh geoms, convex functions are run for each distinct mesh in the
      model, because the convex functions expect static mesh size. If a sphere
      collides with a cube and a tetrahedron, sphere_convex is called twice.
    - The condim of the collision. This ensures that the size of the resulting
      constraint jacobian is determined at compile time.

  Args:
    m: a MuJoCo or MJX model

  Returns:
    a dict with grouping key and values geom1, geom2, pair index
  """
  groups = {}

  for g1, g2, ip in geom_pairs(m):
    types = m.geom_type[g1], m.geom_type[g2]
    data_ids = m.geom_dataid[g1], m.geom_dataid[g2]
    if ip > -1:
      condim = m.pair_dim[ip]
    elif m.geom_priority[g1] > m.geom_priority[g2]:
      condim = m.geom_condim[g1]
    elif m.geom_priority[g1] < m.geom_priority[g2]:
      condim = m.geom_condim[g2]
    else:
      condim = max(m.geom_condim[g1], m.geom_condim[g2])

    key = FunctionKey(types, data_ids, condim)

    if types[0] == mujoco.mjtGeom.mjGEOM_HFIELD:
      # add static grid bounds to the grouping key for hfield collisions
      geom_rbound_hfield = (
          m.geom_rbound_hfield if isinstance(m, Model) else m.geom_rbound
      )
      nrow, ncol = m.hfield_nrow[data_ids[0]], m.hfield_ncol[data_ids[0]]
      xsize, ysize = m.hfield_size[data_ids[0]][:2]
      xtick, ytick = (2 * xsize) / (ncol - 1), (2 * ysize) / (nrow - 1)
      xbound = int(np.ceil(2 * geom_rbound_hfield[g2] / xtick)) + 1
      xbound = min(xbound, ncol)
      ybound = int(np.ceil(2 * geom_rbound_hfield[g2] / ytick)) + 1
      ybound = min(ybound, nrow)
      key = FunctionKey(types, data_ids, condim, (xbound, ybound))

    groups.setdefault(key, []).append((g1, g2, ip))

  return groups

def _contact_groups(m: Model, d: Data) -> Dict[FunctionKey, Contact]:
  """Returns contact groups to check for collisions.

  Contacts are grouped the same way as _geom_groups.  Only one contact is
  emitted per geom pair, even if the collision function emits multiple contacts.

  Args:
    m: MJX model
    d: MJX data

  Returns:
    a dict where the key is the grouping and value is a Contact
  """
  groups = {}
  eps = mujoco.mjMINVAL

  for key, geom_ids in _geom_groups(m).items():
    geom = np.array(geom_ids)
    geom1, geom2, ip = geom.T
    geom1, geom2, ip = geom1[ip == -1], geom2[ip == -1], ip[ip != -1]
    params = []

    if ip.size > 0:
      # pair contacts get their params from m.pair_* fields
      params.append((
          m.pair_margin[ip] - m.pair_gap[ip],
          jp.clip(m.pair_friction[ip], a_min=eps),
          m.pair_solref[ip],
          m.pair_solreffriction[ip],
          m.pair_solimp[ip],
      ))
    if geom1.size > 0 and geom2.size > 0:
      # other contacts get their params from geom fields
      margin = jp.maximum(m.geom_margin[geom1], m.geom_margin[geom2])
      gap = jp.maximum(m.geom_gap[geom1], m.geom_gap[geom2])
      solmix1, solmix2 = m.geom_solmix[geom1], m.geom_solmix[geom2]
      mix = solmix1 / (solmix1 + solmix2)
      mix = jp.where((solmix1 < eps) & (solmix2 < eps), 0.5, mix)
      mix = jp.where((solmix1 < eps) & (solmix2 >= eps), 0.0, mix)
      mix = jp.where((solmix1 >= eps) & (solmix2 < eps), 1.0, mix)
      mix = mix[:, None]  # for correct broadcasting
      # friction: max
      friction = jp.maximum(m.geom_friction[geom1], m.geom_friction[geom2])
      solref1, solref2 = m.geom_solref[geom1], m.geom_solref[geom2]
      # reference standard: mix
      solref_standard = mix * solref1 + (1 - mix) * solref2
      # reference direct: min
      solref_direct = jp.minimum(solref1, solref2)
      is_standard = (solref1[:, [0, 0]] > 0) & (solref2[:, [0, 0]] > 0)
      solref = jp.where(is_standard, solref_standard, solref_direct)
      solreffriction = jp.zeros(geom1.shape + (mujoco.mjNREF,))
      # impedance: mix
      solimp = mix * m.geom_solimp[geom1] + (1 - mix) * m.geom_solimp[geom2]

      pri = m.geom_priority[geom1] != m.geom_priority[geom2]
      if pri.any():
        # use priority geom when specified instead of mixing
        gp1, gp2 = m.geom_priority[geom1], m.geom_priority[geom2]
        gp = np.where(gp1 > gp2, geom1, geom2)[pri]
        friction = friction.at[pri].set(m.geom_friction[gp])
        solref = solref.at[pri].set(m.geom_solref[gp])
        solimp = solimp.at[pri].set(m.geom_solimp[gp])

      # unpack 5d friction:
      friction = friction[:, [0, 0, 1, 2, 2]]
      params.append((margin - gap, friction, solref, solreffriction, solimp))

    params = map(jp.concatenate, zip(*params))
    includemargin, friction, solref, solreffriction, solimp = params

    groups[key] = Contact(
        # dist, pos, frame get filled in by collision functions:
        dist=None,
        pos=None,
        frame=None,
        includemargin=includemargin,
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        dim=d.contact.dim,
        geom1=jp.array(geom[:, 0]),
        geom2=jp.array(geom[:, 1]),
        geom=jp.array(geom[:, :2]),
        efc_address=d.contact.efc_address,
    )

  return groups



def collision(m: types.Model, d: types.Data) -> types.Data:
  """Collides geometries."""
  if d.ncon == 0:
    return d

  max_geom_pairs = 100 # _numeric(m, 'max_geom_pairs')
  max_contact_points = 100 # _numeric(m, 'max_contact_points')

  # run collision functions on groups
  groups = _contact_groups(m, d)
  for key, contact in groups.items():
    # determine which contacts we'll use for collision testing by running a
    # broad phase cull if requested
    if (
        max_geom_pairs > -1
        and contact.geom.shape[0] > max_geom_pairs
        and not set(key.types) & _GEOM_NO_BROADPHASE
    ):
      pos1, pos2 = d.geom_xpos[contact.geom.T]
      size1, size2 = m.geom_rbound[contact.geom.T]
      dist = jax.vmap(jp.linalg.norm)(pos2 - pos1) - (size1 + size2)
      _, idx = jax.lax.top_k(-dist, k=max_geom_pairs)
      contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)

    # run the collision function specified by the grouping key
    func = _COLLISION_FUNC[key.types]
    ncon = func.ncon  # pytype: disable=attribute-error

    dist, pos, frame = func(m, d, key, contact.geom)
    if ncon > 1:
      # repeat contacts to match the number of collisions returned
      repeat_fn = lambda x, r=ncon: jp.repeat(x, r, axis=0)
      contact = jax.tree_util.tree_map(repeat_fn, contact)
    groups[key] = contact.replace(dist=dist, pos=pos, frame=frame)

  # collapse contacts together, ensuring they are grouped by condim
  condim_groups = {}
  for key, contact in groups.items():
    condim_groups.setdefault(key.condim, []).append(contact)

  # limit the number of contacts per condim group if requested
  if max_contact_points > -1:
    for key, contacts in condim_groups.items():
      contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)
      if contact.geom.shape[0] > max_contact_points:
        _, idx = jax.lax.top_k(-contact.dist, k=max_contact_points)
        contact = jax.tree_util.tree_map(lambda x, idx=idx: x[idx], contact)
      condim_groups[key] = [contact]

  contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
  contact = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *contacts)

  return d.replace(contact=contact)



print("end")