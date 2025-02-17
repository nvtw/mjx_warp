import warp as wp

class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
    pass

vec10 = vec10f

@wp.struct
class Option:
  gravity: wp.vec3
  is_sparse: bool # warp only

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
  qpos_spring: wp.array(dtype=wp.float32, ndim=1)
  body_leveladr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_levelsize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_tree: wp.array(dtype=wp.int32, ndim=1)   # warp only
  qLD_leveladr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_levelsize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_sparse_updates: wp.array(dtype=wp.vec3i, ndim=1)  # warp only
  qLD_dense_tilesize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_dense_tileid: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_dofadr: wp.array(dtype=wp.int32, ndim=1)
  body_dofnum: wp.array(dtype=wp.int32, ndim=1)
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
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  opt: Option
  npair: int
  pair_geom1: wp.array(dtype=wp.int32, ndim=1)
  pair_geom2: wp.array(dtype=wp.int32, ndim=1)
  geom_type: wp.array(dtype=wp.int32, ndim=1)
  geom_dataid: wp.array(dtype=wp.int32, ndim=1)
  geom_priority: wp.array(dtype=wp.int32, ndim=1)
  pair_dim: wp.array(dtype=wp.int32, ndim=1)
  geom_condim: wp.array(dtype=wp.int32, ndim=1)
  exclude_signature : wp.array(dtype=wp.int32, ndim=1)
  geom_contype : wp.array(dtype=wp.int32, ndim=1)
  geom_conaffinity : wp.array(dtype=wp.int32, ndim=1)
  body_weldid : wp.array(dtype=wp.int32, ndim=1)
  opt_disableflags : int
  body_geomadr : wp.array(dtype=wp.int32, ndim=1)
  body_geomnum : wp.array(dtype=wp.int32, ndim=1)

@wp.struct
class Data:
  nworld: int
  qpos: wp.array(dtype=wp.float32, ndim=2)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  qacc: wp.array(dtype=wp.float32, ndim=2)
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
  qM: wp.array(dtype=wp.float32, ndim=3)
  qLD: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)
  cvel: wp.array(dtype=wp.spatial_vector, ndim=2)
  cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)
  qacc_smooth: wp.array(dtype=wp.float32, ndim=2)

  geom_size: wp.array(dtype=wp.vec3, ndim=2)
  geom_mesh_vert: wp.array(dtype=wp.vec3, ndim=2)
  geom_mesh_face: wp.array(dtype=wp.vec3i, ndim=2)
  geom_mesh_norm: wp.array(dtype=wp.vec3, ndim=2)
  geom_mesh_edge: wp.array(dtype=wp.vec3i, ndim=2)
  geom_mesh_edge_norm: wp.array(dtype=wp.vec3, ndim=2)


