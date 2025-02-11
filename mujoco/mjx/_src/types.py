import warp as wp
import numpy as np


class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
    pass

vec10 = vec10f



@wp.struct
class Contact:
    """Result of collision detection functions.

    Attributes:
        dist: distance between nearest points; neg: penetration
        pos: position of contact point: midpoint between geoms (3,)
        frame: normal is in [0-2] (9,)
        includemargin: include if dist<includemargin=margin-gap (1,)
        friction: tangent1, 2, spin, roll1, 2 (5,)
        solref: constraint solver reference, normal direction (mjNREF,)
        solreffriction: constraint solver reference, friction directions (mjNREF,)
        solimp: constraint solver impedance (mjNIMP,)
        dim: contact space dimensionality: 1, 3, 4, or 6
        geom1: id of geom 1; deprecated, use geom[0]
        geom2: id of geom 2; deprecated, use geom[1]
        geom: geom ids (2,)
        efc_address: address in efc; -1: not included
    """

    dist: wp.array(dtype=wp.float32, ndim=1)
    pos: wp.array(dtype=wp.vec3, ndim=1)
    frame: wp.array(dtype=wp.mat33, ndim=1)
    includemargin: wp.array(dtype=wp.float32, ndim=1)
    friction: wp.array(dtype=wp.float32, ndim=2)  # (n_points, 5)
    solref: wp.array(dtype=wp.float32, ndim=2)  # (n_points, mjNREF)
    solreffriction: wp.array(dtype=wp.float32, ndim=2)  # (n_points, mjNREF)
    solimp: wp.array(dtype=wp.float32, ndim=2)  # (n_points, mjNIMP)
    dim: wp.array(dtype=wp.int32, ndim=1)
    geom1: wp.array(dtype=wp.int32, ndim=1)
    geom2: wp.array(dtype=wp.int32, ndim=1)
    efc_address: wp.array(dtype=wp.int32, ndim=1)


@wp.struct
class Model:
  nq: int
  nv: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  nmocap: int
  nM: int
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  body_leveladr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_levelsize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_tree: wp.array(dtype=wp.int32, ndim=1)   # warp only
  qLD_leveladr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_levelsize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_updates: wp.array(dtype=wp.vec3i, ndim=1)  # warp only
  body_jntadr: wp.array(dtype=wp.int32, ndim=1)
  body_jntnum: wp.array(dtype=wp.int32, ndim=1)
  body_parentid: wp.array(dtype=wp.int32, ndim=1)
  body_mocapid: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)

  #https://mujoco.readthedocs.io/en/3.1.3/APIreference/APItypes.html
  geom_size: wp.array(dtype=wp.vec3, ndim=1)
  geom_dataid: wp.array(dtype=wp.int32, ndim=1)
  npair : int
  geom_contype: wp.array(dtype=wp.int32, ndim=1)
  geom_conaffinity: wp.array(dtype=wp.int32, ndim=1)
  body_geomadr : wp.array(dtype=wp.int32, ndim=1)
  body_geomnum : wp.array(dtype=wp.int32, ndim=1)
  exclude_signature : wp.array(dtype=wp.int32, ndim=1)
  opt_disableflags : int
  body_weldid : wp.array(dtype=wp.int32, ndim=1)
  body_parentid : wp.array(dtype=wp.int32, ndim=1)
  mesh_convex : wp.array(dtype=wp.int32, ndim=1)

@wp.struct
class Data:
  nworld: int
  qpos: wp.array(dtype=wp.float32, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)
  cinert: wp.array(dtype=vec10, ndim=2)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  crb: wp.array(dtype=vec10, ndim=2)
  qM: wp.array(dtype=wp.float32, ndim=2)
  qLD: wp.array(dtype=wp.float32, ndim=2)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)

  #contact: wp.array(dtype=Contact, ndim=2)
  contact_pos: wp.array(dtype=wp.vec3, ndim=2)

