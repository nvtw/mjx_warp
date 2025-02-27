import warp as wp
import mujoco
from . import smooth
from . import support
from . import types

MAX_LS_ITER = 64


class veclsf(wp.types.vector(length=MAX_LS_ITER, dtype=wp.float32)):
  pass


vecls = veclsf


@wp.struct
class Context:
  Jaref: wp.array(dtype=wp.float32, ndim=1)
  Ma: wp.array(dtype=wp.float32, ndim=2)
  grad: wp.array(dtype=wp.float32, ndim=2)
  grad_dot: wp.array(dtype=wp.float32, ndim=1)
  Mgrad: wp.array(dtype=wp.float32, ndim=2)
  search: wp.array(dtype=wp.float32, ndim=2)
  search_dot: wp.array(dtype=wp.float32, ndim=1)
  gauss: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  prev_cost: wp.array(dtype=wp.float32, ndim=1)
  solver_niter: wp.array(dtype=wp.int32, ndim=1)
  active: wp.array(dtype=wp.int32, ndim=1)
  mv: wp.array(dtype=wp.float32, ndim=2)
  jv: wp.array(dtype=wp.float32, ndim=1)
  quad: wp.array(dtype=wp.vec3f, ndim=1)
  quad_gauss: wp.array(dtype=wp.vec3f, ndim=1)
  quad_total: wp.array(dtype=wp.vec3f, ndim=1)
  h: wp.array(dtype=wp.float32, ndim=3)
  alpha: wp.array(dtype=wp.float32, ndim=1)
  prev_grad: wp.array(dtype=wp.float32, ndim=2)
  prev_Mgrad: wp.array(dtype=wp.float32, ndim=2)
  beta: wp.array(dtype=wp.float32, ndim=1)
  beta_num: wp.array(dtype=wp.float32, ndim=1)
  beta_den: wp.array(dtype=wp.float32, ndim=1)
  done: wp.array(dtype=wp.int32, ndim=1)
  alpha_candidate: wp.array(dtype=wp.float32, ndim=1)
  cost_candidate: wp.array(dtype=veclsf, ndim=1)
  quad_total_candidate: wp.array(dtype=wp.vec3f, ndim=2)


def _context(m: types.Model, d: types.Data) -> Context:
  ctx = Context()
  ctx.Jaref = wp.empty(shape=(d.njmax,), dtype=wp.float32)
  ctx.Ma = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.grad_dot = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.search = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.search_dot = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.gauss = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.cost = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.prev_cost = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.solver_niter = wp.empty(shape=(d.nworld,), dtype=wp.int32)
  ctx.active = wp.empty(shape=(d.njmax,), dtype=wp.int32)
  ctx.mv = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.jv = wp.empty(shape=(d.njmax,), dtype=wp.float32)
  ctx.quad = wp.empty(shape=(d.njmax,), dtype=wp.vec3f)
  ctx.quad_gauss = wp.empty(shape=(d.nworld,), dtype=wp.vec3f)
  ctx.quad_total = wp.empty(shape=(d.nworld,), dtype=wp.vec3f)
  ctx.h = wp.empty(shape=(d.nworld, m.nv, m.nv), dtype=wp.float32)
  ctx.alpha = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.prev_grad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.prev_Mgrad = wp.empty(shape=(d.nworld, m.nv), dtype=wp.float32)
  ctx.beta = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.beta_num = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.beta_den = wp.empty(shape=(d.nworld,), dtype=wp.float32)
  ctx.done = wp.empty(shape=(d.nworld,), dtype=wp.int32)
  ctx.alpha_candidate = wp.empty(shape=(MAX_LS_ITER,), dtype=wp.float32)
  ctx.cost_candidate = wp.empty(shape=(d.nworld,), dtype=veclsf)
  ctx.quad_total_candidate = wp.empty(shape=(d.nworld, MAX_LS_ITER), dtype=wp.vec3f)

  return ctx


def _create_context(ctx: Context, m: types.Model, d: types.Data, grad: bool = True):
  # jaref = d.efc_J @ d.qacc - d.efc_aref
  ctx.Jaref.zero_()

  @wp.kernel
  def _jaref(ctx: Context, m: types.Model, d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      ctx.Jaref,
      efcid,
      d.efc_J[efcid, dofid] * d.qacc[worldid, dofid] - d.efc_aref[efcid] / float(m.nv),
    )

  wp.launch(_jaref, dim=(d.njmax, m.nv), inputs=[ctx, m, d])

  # Ma = qM @ qacc
  support.mul_m(m, d, ctx.Ma, d.qacc)

  ctx.cost.fill_(wp.inf)
  ctx.solver_niter.zero_()
  ctx.done.zero_()

  _update_constraint(m, d, ctx)
  if grad:
    _update_gradient(m, d, ctx)

    # search = -Mgrad
    ctx.search_dot.zero_()

    @wp.kernel
    def _search(ctx: Context):
      worldid, dofid = wp.tid()
      search = -1.0 * ctx.Mgrad[worldid, dofid]
      ctx.search[worldid, dofid] = search
      wp.atomic_add(ctx.search_dot, worldid, search * search)

    wp.launch(_search, dim=(d.nworld, m.nv), inputs=[ctx])


def _update_constraint(m: types.Model, d: types.Data, ctx: Context):
  wp.copy(ctx.prev_cost, ctx.cost)
  ctx.cost.zero_()

  @wp.kernel
  def _efc_kernel(ctx: Context, d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    Jaref = ctx.Jaref[efcid]
    efc_D = d.efc_D[efcid]

    # TODO(team): active and conditionally active constraints
    active = int(Jaref < 0.0)
    ctx.active[efcid] = active

    # efc_force = -efc_D * Jaref * active
    d.efc_force[efcid] = -1.0 * efc_D * Jaref * float(active)

    # cost = 0.5 * sum(efc_D * Jaref * Jaref * active))
    wp.atomic_add(ctx.cost, worldid, 0.5 * efc_D * Jaref * Jaref * float(active))

  wp.launch(_efc_kernel, dim=(d.njmax,), inputs=[ctx, d])

  # qfrc_constraint = efc_J.T @ efc_force
  d.qfrc_constraint.zero_()

  @wp.kernel
  def _qfrc_constraint(d: types.Data):
    dofid, efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      d.qfrc_constraint[worldid],
      dofid,
      d.efc_J[efcid, dofid] * d.efc_force[efcid],
    )

  wp.launch(_qfrc_constraint, dim=(m.nv, d.njmax), inputs=[d])

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)
  ctx.gauss.zero_()

  @wp.kernel
  def _gauss(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    gauss_cost = (
      0.5
      * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid])
      * (d.qacc[worldid, dofid] - d.qacc_smooth[worldid, dofid])
    )
    wp.atomic_add(ctx.gauss, worldid, gauss_cost)
    wp.atomic_add(ctx.cost, worldid, gauss_cost)

  wp.launch(_gauss, dim=(d.nworld, m.nv), inputs=[ctx, d])


def _update_gradient(m: types.Model, d: types.Data, ctx: Context):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  ctx.grad_dot.zero_()

  @wp.kernel
  def _grad(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    grad = (
      ctx.Ma[worldid, dofid]
      - d.qfrc_smooth[worldid, dofid]
      - d.qfrc_constraint[worldid, dofid]
    )
    ctx.grad[worldid, dofid] = grad
    wp.atomic_add(ctx.grad_dot, worldid, grad * grad)

  wp.launch(_grad, dim=(d.nworld, m.nv), inputs=[ctx, d])

  if m.opt.solver == 1:  # CG
    smooth.solve_m(m, d, ctx.grad, ctx.Mgrad)
  elif m.opt.solver == 2:  # Newton
    # TODO(team): sparse version
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    @wp.kernel
    def _copy_lower_triangle(m: types.Model, d: types.Data, ctx: Context):
      worldid, elementid = wp.tid()
      rowid = m.dof_tri_row[elementid]
      colid = m.dof_tri_col[elementid]
      ctx.h[worldid, rowid, colid] = d.qM[worldid, rowid, colid]

    wp.launch(
      _copy_lower_triangle, dim=(d.nworld, m.dof_tri_row.size), inputs=[m, d, ctx]
    )

    @wp.kernel
    def _JTDAJ(ctx: Context, m: types.Model, d: types.Data):
      efcid, elementid = wp.tid()
      dofi = m.dof_tri_row[elementid]
      dofj = m.dof_tri_col[elementid]

      if efcid >= d.nefc_total[0]:
        return

      efc_D = d.efc_D[efcid]
      active = ctx.active[efcid]
      if efc_D == 0.0 or active == 0:
        return

      worldid = d.efc_worldid[efcid]
      wp.atomic_add(
        ctx.h[worldid, dofi],
        dofj,
        d.efc_J[efcid, dofi] * d.efc_J[efcid, dofj] * efc_D * float(active),
      )

    wp.launch(_JTDAJ, dim=(d.njmax, m.dof_tri_row.size), inputs=[ctx, m, d])

    TILE = m.nv

    @wp.kernel
    def _cholesky(ctx: Context):
      worldid = wp.tid()
      mat_tile = wp.tile_load(ctx.h[worldid], shape=(TILE, TILE))
      fact_tile = wp.tile_cholesky(mat_tile)
      input_tile = wp.tile_load(ctx.grad[worldid], shape=TILE)
      output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
      wp.tile_store(ctx.Mgrad[worldid], output_tile)

    wp.launch_tiled(_cholesky, dim=(d.nworld,), inputs=[ctx], block_dim=256)


@wp.func
def _rescale(m: types.Model, value: float) -> float:
  return value / (m.stat.meaninertia * float(wp.max(1, m.nv)))


def _linesearch(m: types.Model, d: types.Data, ctx: Context):
  # mv = qM @ search
  support.mul_m(m, d, ctx.mv, ctx.search)

  # jv = efc_J @ search
  ctx.jv.zero_()

  @wp.kernel
  def _jv(ctx: Context, d: types.Data):
    efcid, dofid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    wp.atomic_add(
      ctx.jv,
      efcid,
      d.efc_J[efcid, dofid] * ctx.search[worldid, dofid],
    )

  wp.launch(_jv, dim=(d.njmax, m.nv), inputs=[ctx, d])

  # prepare quadratics
  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  ctx.quad_gauss.zero_()

  @wp.kernel
  def _quad_gauss(ctx: Context, m: types.Model, d: types.Data):
    worldid, dofid = wp.tid()
    search = ctx.search[worldid, dofid]
    quad_gauss = wp.vec3(
      ctx.gauss[worldid] / float(m.nv),
      search * (ctx.Ma[worldid, dofid] - d.qfrc_smooth[worldid, dofid]),
      0.5 * search * ctx.mv[worldid, dofid],
    )
    wp.atomic_add(ctx.quad_gauss, worldid, quad_gauss)

  wp.launch(_quad_gauss, dim=(d.nworld, m.nv), inputs=[ctx, m, d])

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]
  @wp.kernel
  def _quad(ctx: Context, d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    Jaref = ctx.Jaref[efcid]
    jv = ctx.jv[efcid]
    efc_D = d.efc_D[efcid]
    ctx.quad[efcid][0] = 0.5 * Jaref * Jaref * efc_D
    ctx.quad[efcid][1] = jv * Jaref * efc_D
    ctx.quad[efcid][2] = 0.5 * jv * jv * efc_D

  wp.launch(_quad, dim=(d.njmax), inputs=[ctx, d])

  @wp.kernel
  def _quad_total(ctx: Context, m: types.Model):
    worldid, alphaid = wp.tid()

    if alphaid >= m.opt.ls_iterations:
      return

    ctx.quad_total_candidate[worldid, alphaid] = ctx.quad_gauss[worldid]

  wp.launch(_quad_total, dim=(d.nworld, MAX_LS_ITER), inputs=[ctx, m])

  @wp.kernel
  def _quad_total_candidate(ctx: Context, m: types.Model, d: types.Data):
    efcid, alphaid = wp.tid()

    if alphaid >= m.opt.ls_iterations:
      return

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    x = ctx.Jaref[efcid] + ctx.alpha_candidate[alphaid] * ctx.jv[efcid]
    # TODO(team): active and conditionally active constraints
    if x < 0.0:
      wp.atomic_add(ctx.quad_total_candidate[worldid], alphaid, ctx.quad[efcid])

  wp.launch(_quad_total_candidate, dim=(d.njmax, MAX_LS_ITER), inputs=[ctx, m, d])

  @wp.kernel
  def _cost_alpha(ctx: Context, m: types.Model):
    worldid, alphaid = wp.tid()

    if alphaid >= m.opt.ls_iterations:
      ctx.cost_candidate[worldid][alphaid] = wp.inf
      return

    alpha = ctx.alpha_candidate[alphaid]
    alpha_sq = alpha * alpha
    quad_total0 = ctx.quad_total_candidate[worldid, alphaid][0]
    quad_total1 = ctx.quad_total_candidate[worldid, alphaid][1]
    quad_total2 = ctx.quad_total_candidate[worldid, alphaid][2]

    ctx.cost_candidate[worldid][alphaid] = (
      alpha_sq * quad_total2 + alpha * quad_total1 + quad_total0
    )

  wp.launch(_cost_alpha, dim=(d.nworld, MAX_LS_ITER), inputs=[ctx, m])

  @wp.kernel
  def _best_alpha(ctx: Context):
    worldid = wp.tid()
    bestid = wp.argmin(ctx.cost_candidate[worldid])
    ctx.alpha[worldid] = ctx.alpha_candidate[bestid]

  wp.launch(_best_alpha, dim=(d.nworld), inputs=[ctx])

  @wp.kernel
  def _qacc_ma(ctx: Context, d: types.Data):
    worldid, dofid = wp.tid()
    alpha = ctx.alpha[worldid]
    d.qacc[worldid, dofid] += alpha * ctx.search[worldid, dofid]
    ctx.Ma[worldid, dofid] += alpha * ctx.mv[worldid, dofid]

  wp.launch(_qacc_ma, dim=(d.nworld, m.nv), inputs=[ctx, d])

  @wp.kernel
  def _jaref(ctx: Context, d: types.Data):
    efcid = wp.tid()

    if efcid >= d.nefc_total[0]:
      return

    worldid = d.efc_worldid[efcid]
    ctx.Jaref[efcid] += ctx.alpha[worldid] * ctx.jv[efcid]

  wp.launch(_jaref, dim=(d.njmax,), inputs=[ctx, d])


def solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""

  # warmstart
  wp.copy(d.qacc, d.qacc_warmstart)

  ctx = _context(m, d)
  _create_context(ctx, m, d, grad=True)

  # alpha candidates
  @wp.kernel
  def _alpha_candidate(ctx: Context, m: types.Model):
    tid = wp.tid()

    if tid >= m.opt.ls_iterations:
      return

    ctx.alpha_candidate[tid] = float(tid) / float(wp.max(m.opt.ls_iterations - 1, 1))

  wp.launch(_alpha_candidate, dim=(MAX_LS_ITER), inputs=[ctx, m])

  for i in range(m.opt.iterations):
    _linesearch(m, d, ctx)
    wp.copy(ctx.prev_grad, ctx.grad)
    wp.copy(ctx.prev_Mgrad, ctx.Mgrad)
    _update_constraint(m, d, ctx)
    _update_gradient(m, d, ctx)

    if m.opt.solver == 2:  # Newton
      ctx.search_dot.zero_()

      @wp.kernel
      def _search_newton(ctx: Context):
        worldid, dofid = wp.tid()
        search = -1.0 * ctx.Mgrad[worldid, dofid]
        ctx.search[worldid, dofid] = search
        wp.atomic_add(ctx.search_dot, worldid, search * search)

      wp.launch(_search_newton, dim=(d.nworld, m.nv), inputs=[ctx])
    else:  # polak-ribiere
      ctx.beta_num.zero_()
      ctx.beta_den.zero_()

      @wp.kernel
      def _beta_num_den(ctx: Context):
        worldid, dofid = wp.tid()
        prev_Mgrad = ctx.prev_Mgrad[worldid][dofid]
        wp.atomic_add(
          ctx.beta_num,
          worldid,
          ctx.grad[worldid, dofid] * (ctx.Mgrad[worldid, dofid] - prev_Mgrad),
        )
        wp.atomic_add(ctx.beta_den, worldid, ctx.prev_grad[worldid, dofid] * prev_Mgrad)

      wp.launch(_beta_num_den, dim=(d.nworld, m.nv), inputs=[ctx])

      @wp.kernel
      def _beta(ctx: Context):
        worldid = wp.tid()
        ctx.beta[worldid] = wp.max(
          0.0, ctx.beta_num[worldid] / wp.max(mujoco.mjMINVAL, ctx.beta_den[worldid])
        )

      wp.launch(_beta, dim=(d.nworld,), inputs=[ctx])

      ctx.search_dot.zero_()

      @wp.kernel
      def _search_cg(ctx: Context):
        worldid, dofid = wp.tid()
        search = (
          -1.0 * ctx.Mgrad[worldid, dofid]
          + ctx.beta[worldid] * ctx.search[worldid, dofid]
        )
        ctx.search[worldid, dofid] = search
        wp.atomic_add(ctx.search_dot, worldid, search * search)

      wp.launch(_search_cg, dim=(d.nworld, m.nv), inputs=[ctx])

    @wp.kernel
    def _done(ctx: Context, m: types.Model, solver_niter: int):
      worldid = wp.tid()
      improvement = _rescale(m, ctx.prev_cost[worldid] - ctx.cost[worldid])
      gradient = _rescale(m, wp.math.sqrt(ctx.grad_dot[worldid]))
      done = solver_niter >= m.opt.iterations
      done = done or (improvement < m.opt.tolerance)
      done = done or (gradient < m.opt.tolerance)
      ctx.done[worldid] = int(done)

    wp.launch(_done, dim=(d.nworld,), inputs=[ctx, m, i])
    # TODO(team): return if all done

  wp.copy(d.qacc_warmstart, d.qacc)
