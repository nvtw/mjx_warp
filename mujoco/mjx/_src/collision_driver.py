import warp as wp

from . import types
from typing import Tuple, Iterator

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









from typing import Dict, List, Tuple
import numpy as np


class Contact:
    """Struct for storing contact information."""
    dist: np.ndarray
    pos: np.ndarray
    frame: np.ndarray
    includemargin: np.ndarray
    friction: np.ndarray
    solref: np.ndarray
    solreffriction: np.ndarray
    solimp: np.ndarray
    dim: np.ndarray
    geom1: np.ndarray
    geom2: np.ndarray
    geom: np.ndarray
    efc_address: np.ndarray


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


mjMINVAL = float(1E-15)

mjNREF = int(2) 
mjNIMP = int(5)

DisableBit_FILTERPARENT = int(512)

GeomType_PLANE = 0
GeomType_HFIELD = 1
GeomType_SPHERE = 2
GeomType_CAPSULE = 3
GeomType_ELLIPSOID = 4
GeomType_CYLINDER = 5
GeomType_BOX = 6
GeomType_CONVEX = 7
GeomType_size = 8


def geom_pairs(
    m: types.Model,
) -> Iterator[Tuple[int, int, int]]:
    """Yields geom pairs to check for collisions.

    Args:
        m: a MuJoCo or MJX model.

    Yields:
        geom1, geom2, and pair index if defined in <pair> (else -1).
    """
    pairs = set()

    # Iterate through predefined pairs in <pair> elements
    for i in range(m.npair):
        g1, g2 = m.pair_geom1[i], m.pair_geom2[i]
        if m.geom_type[g1] > m.geom_type[g2]:
            g1, g2 = g2, g1  # Ensure ordering for function mapping
        pairs.add((g1, g2))
        yield g1, g2, i  # Emit known pair

    # Handle dynamically computed geom pairs
    exclude_signature = set(m.exclude_signature)
    geom_con = m.geom_contype | m.geom_conaffinity
    filterparent = not (m.opt.disableflags & DisableBit_FILTERPARENT)

    b_start, b_end = m.body_geomadr, m.body_geomadr + m.body_geomnum

    for b1 in range(m.nbody):
        if not np.any(geom_con[b_start[b1]:b_end[b1]]):
            continue

        w1 = m.body_weldid[b1]
        w1_p = m.body_weldid[m.body_parentid[w1]]

        for b2 in range(b1, m.nbody):
            if not np.any(geom_con[b_start[b2]:b_end[b2]]):
                continue

            signature = (b1 << 16) + b2
            if signature in exclude_signature:
                continue

            w2 = m.body_weldid[b2]
            if w1 == w2:
                continue

            w2_p = m.body_weldid[m.body_parentid[w2]]
            if filterparent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
                continue

            g1_range = np.array([g for g in range(b_start[b1], b_end[b1]) if geom_con[g]])
            g2_range = np.array([g for g in range(b_start[b2], b_end[b2]) if geom_con[g]])

            if g1_range.size == 0 or g2_range.size == 0:
                continue

            g1_mesh, g2_mesh = np.meshgrid(g1_range, g2_range, indexing="ij")
            g1_list, g2_list = g1_mesh.flatten(), g2_mesh.flatten()

            for g1, g2 in zip(g1_list, g2_list):
                t1, t2 = m.geom_type[g1], m.geom_type[g2]
                if t1 > t2:
                    g1, g2, t1, t2 = g2, g1, t2, t1

                if (t1, t2) in [(GeomType_PLANE, GeomType_PLANE), (GeomType_PLANE, GeomType_HFIELD)]:
                    continue

                mask = (m.geom_contype[g1] & m.geom_conaffinity[g2]) | (m.geom_contype[g2] & m.geom_conaffinity[g1])
                if not mask:
                    continue

                if (g1, g2) not in pairs:
                    pairs.add((g1, g2))
                    yield g1, g2, -1



def _geom_groups(m: types.Model) -> Dict[FunctionKey, List[Tuple[int, int, int]]]:
    """Returns geom pairs to check for collision grouped by collision function.

    The grouping consists of:
      - The collision function to run, which is determined by geom types.
      - For mesh geoms, convex functions are run for each distinct mesh in the
        model, because the convex functions expect static mesh size.
      - The condim of the collision to ensure the constraint Jacobian size is
        determined at compile time.

    Args:
        m: A MuJoCo or MJX model.

    Returns:
        A dict with grouping key and values (geom1, geom2, pair index).
    """
    groups = {}

    for g1, g2, ip in geom_pairs(m):
        types = (m.geom_type[g1], m.geom_type[g2])
        data_ids = (m.geom_dataid[g1], m.geom_dataid[g2])

        if ip > -1:
            condim = m.pair_dim[ip]
        elif m.geom_priority[g1] > m.geom_priority[g2]:
            condim = m.geom_condim[g1]
        elif m.geom_priority[g1] < m.geom_priority[g2]:
            condim = m.geom_condim[g2]
        else:
            condim = max(m.geom_condim[g1], m.geom_condim[g2])

        key = FunctionKey(types, data_ids, condim)

        if types[0] == GeomType_HFIELD:
            # Add static grid bounds to the grouping key for hfield collisions
            geom_rbound_hfield = (
                m.geom_rbound_hfield if isinstance(m, types.Model) else m.geom_rbound
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


def _contact_groups(m: types.Model, d: types.Data) -> Dict[FunctionKey, Contact]:
    """Returns contact groups to check for collisions.

    Contacts are grouped the same way as _geom_groups. Only one contact is
    emitted per geom pair, even if the collision function emits multiple contacts.

    Args:
        m: MJX model
        d: MJX data

    Returns:
        A dict where the key is the grouping and value is a Contact.
    """
    groups = {}
    eps = mjMINVAL

    for key, geom_ids in _geom_groups(m).items():
        geom = np.array(geom_ids)
        geom1, geom2, ip = geom.T
        geom1, geom2, ip = geom1[ip == -1], geom2[ip == -1], ip[ip != -1]
        params = []

        if ip.size > 0:
            # Pair contacts get their params from m.pair_* fields
            params.append((
                m.pair_margin[ip] - m.pair_gap[ip],
                np.clip(m.pair_friction[ip], a_min=eps, a_max=None),
                m.pair_solref[ip],
                m.pair_solreffriction[ip],
                m.pair_solimp[ip],
            ))

        if geom1.size > 0 and geom2.size > 0:
            # Other contacts get their params from geom fields
            margin = np.maximum(m.geom_margin[geom1], m.geom_margin[geom2])
            gap = np.maximum(m.geom_gap[geom1], m.geom_gap[geom2])

            solmix1, solmix2 = m.geom_solmix[geom1], m.geom_solmix[geom2]
            mix = solmix1 / (solmix1 + solmix2)
            mix = np.where((solmix1 < eps) & (solmix2 < eps), 0.5, mix)
            mix = np.where((solmix1 < eps) & (solmix2 >= eps), 0.0, mix)
            mix = np.where((solmix1 >= eps) & (solmix2 < eps), 1.0, mix)
            mix = mix[:, None]  # Ensure correct broadcasting

            # Friction: max
            friction = np.maximum(m.geom_friction[geom1], m.geom_friction[geom2])
            solref1, solref2 = m.geom_solref[geom1], m.geom_solref[geom2]

            # Reference standard: mix
            solref_standard = mix * solref1 + (1 - mix) * solref2

            # Reference direct: min
            solref_direct = np.minimum(solref1, solref2)

            is_standard = (solref1[:, [0, 0]] > 0) & (solref2[:, [0, 0]] > 0)
            solref = np.where(is_standard, solref_standard, solref_direct)

            solreffriction = np.zeros(geom1.shape + (mjNREF,))

            # Impedance: mix
            solimp = mix * m.geom_solimp[geom1] + (1 - mix) * m.geom_solimp[geom2]

            pri = m.geom_priority[geom1] != m.geom_priority[geom2]
            if pri.any():
                # Use priority geom when specified instead of mixing
                gp1, gp2 = m.geom_priority[geom1], m.geom_priority[geom2]
                gp = np.where(gp1 > gp2, geom1, geom2)[pri]

                friction[pri] = m.geom_friction[gp]
                solref[pri] = m.geom_solref[gp]
                solimp[pri] = m.geom_solimp[gp]

            # Unpack 5D friction
            friction = friction[:, [0, 0, 1, 2, 2]]

            params.append((margin - gap, friction, solref, solreffriction, solimp))

        # Concatenate parameter lists
        params = [np.concatenate(p) for p in zip(*params)]
        includemargin, friction, solref, solreffriction, solimp = params

        groups[key] = Contact(
            dist=None,
            pos=None,
            frame=None,
            includemargin=includemargin,
            friction=friction,
            solref=solref,
            solreffriction=solreffriction,
            solimp=solimp,
            dim=d.contact.dim,
            geom1=np.array(geom[:, 0]),
            geom2=np.array(geom[:, 1]),
            geom=np.array(geom[:, :2]),
            efc_address=d.contact.efc_address,
        )

    return groups


def collision(m: types.Model, d: types.Data, max_geom_pairs: int, max_contact_points: int) -> types.Data:
    """Collides geometries."""
    if d.ncon == 0:
        return d

    # max_geom_pairs = _numeric(m, 'max_geom_pairs')
    # max_contact_points = _numeric(m, 'max_contact_points')

    # run collision functions on groups
    groups = _contact_groups(m, d)
    
    for key, contact in groups.items():
        # determine which contacts we'll use for collision testing by running a broad phase cull if requested
        if (
            max_geom_pairs > -1
            and contact.geom.shape[0] > max_geom_pairs
            #and not set(key.types) & _GEOM_NO_BROADPHASE
        ):
            pos1, pos2 = d.geom_xpos[contact.geom.T]
            size1, size2 = m.geom_rbound[contact.geom.T]
            dist = np.linalg.norm(pos2 - pos1, axis=1) - (size1 + size2)
            
            # Get indices of top-k elements
            idx = np.argsort(-dist)[:max_geom_pairs]
            
            # Apply indexing to contact
            contact = {k: v[idx] for k, v in contact._asdict().items()}
        
        # run the collision function specified by the grouping key
        func = plane_convex_launch #_COLLISION_FUNC[key.types]
        # ncon is the number of contacts returned by the collision function
        ncon = 4 # func.ncon  # pytype: disable=attribute-error

        dist, pos, frame = func(m, d, key, contact["geom"])
        
        if ncon > 1:
            # repeat contacts to match the number of collisions returned
            contact = {k: np.repeat(v, ncon, axis=0) for k, v in contact.items()}
        
        groups[key] = {**contact, "dist": dist, "pos": pos, "frame": frame}

    # collapse contacts together, ensuring they are grouped by condim
    condim_groups = {}
    for key, contact in groups.items():
        condim_groups.setdefault(key.condim, []).append(contact)

    # limit the number of contacts per condim group if requested
    if max_contact_points > -1:
        for key, contacts in condim_groups.items():
            # Concatenate contacts in each condim group
            contact = {k: np.concatenate([c[k] for c in contacts]) for k in contacts[0].keys()}
            
            if contact["geom"].shape[0] > max_contact_points:
                idx = np.argsort(-contact["dist"])[:max_contact_points]
                contact = {k: v[idx] for k, v in contact.items()}
            
            condim_groups[key] = [contact]

    contacts = sum([condim_groups[k] for k in sorted(condim_groups)], [])
    contact = {k: np.concatenate([c[k] for c in contacts]) for k in contacts[0].keys()}

    return d.replace(contact=contact)








print("end")