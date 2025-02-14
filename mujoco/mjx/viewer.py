import logging
import time
from typing import Sequence, Union
import warp as wp

from absl import app
from absl import flags
import mujoco
import mujoco.viewer

from ._src import io
from ._src import smooth
from ._src.types import Data
from ._src.types import Model
# pylint: enable=g-importing-member



_JIT = flags.DEFINE_bool("jit", True, "To jit or not to jit.")
_MODEL_PATH = flags.DEFINE_string(
  "mjcf", None, "Path to a MuJoCo MJCF file.", required=True
)


_VIEWER_GLOBAL_STATE = {
  "running": True,
  "single_step": False,
}


def forward(m: Model, d: Data) -> Data:
  """Forward dynamics."""
  fwd_position(m, d)
  # d = fwd_velocity(m, d)
  # d = fwd_actuation(m, d)
  # d = fwd_acceleration(m, d)

  # if d.efc_J.size == 0:
  #   d = d.replace(qacc=d.qacc_smooth)
  #   return d

  # d = named_scope(solver.solve)(m, d)

  return d


def fwd_position(m: Model, d: Data) -> Data:
  """Position-dependent computations."""
  # TODO(robotics-simulation): tendon
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  # smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.factor_m(m, d)
  # collision_driver.collision(m, d)
  # constraint.make_constraint(m, d)
  # smooth.transmission(m, d)
  return d


def fwd_velocity(m: Model, d: Data) -> Data:
  """Velocity-dependent computations."""
  d = d.replace(
      actuator_velocity=d.actuator_moment @ d.qvel,
      ten_velocity=d.ten_J @ d.qvel,
  )
  d = smooth.com_vel(m, d)
  d = passive.passive(m, d)
  d = smooth.rne(m, d)
  return d


def fwd_acceleration(m: Model, d: Data) -> Data:
  """Add up all non-constraint forces, compute qacc_smooth."""
  qfrc_applied = d.qfrc_applied + support.xfrc_accumulate(m, d)
  qfrc_smooth = d.qfrc_passive - d.qfrc_bias + d.qfrc_actuator + qfrc_applied
  qacc_smooth = smooth.solve_m(m, d, qfrc_smooth)
  d = d.replace(qfrc_smooth=qfrc_smooth, qacc_smooth=qacc_smooth)
  return d


def key_callback(key: int) -> None:
  if key == 32:  # Space bar
    _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
    logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])
  if key == 115:  # Step
    _VIEWER_GLOBAL_STATE["single_step"] = True
    logging.info("single_step = %s", _VIEWER_GLOBAL_STATE["single_step"])


def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by mjx_warp."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading model from: {_MODEL_PATH.value}.")
  if _MODEL_PATH.value.endswith(".mjb"):
    m = mujoco.MjModel.from_binary_path(_MODEL_PATH.value)
  else:
    m = mujoco.MjModel.from_xml_path(_MODEL_PATH.value)
  d = mujoco.MjData(m)
  # mx = mjx.put_model(m)
  # dx = mjx.put_data(m, d)
  mw = io.put_model(m)
  # dw = io.make_data(m, nworld=1)
  dw = io.put_data(m, d, nworld=1)

  # print(f'Default backend: {jax.default_backend()}')
  def step_fn(mw, dw):
    dw = forward(mw, dw)
    return dw

  if _JIT.value:
    start = time.perf_counter()
    wp.clear_kernel_cache()
    step_fn(mw, dw)
    wp.synchronize()
    # step_fn = jax.jit(step_fn).lower(mx, dx).compile()
    elapsed = time.perf_counter() - start
    print(f"Compilation took {elapsed}s.")

  viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)
  with viewer:
    while True:
      start = time.time()

      # TODO(robotics-simulation): recompile when changing disable flags, etc.
      # dx = dx.replace(
      #     ctrl=jp.array(d.ctrl),
      #     act=jp.array(d.act),
      #     xfrc_applied=jp.array(d.xfrc_applied),
      # )
      dw = io.put_data(m, d, nworld=1)

      step = False

      if _VIEWER_GLOBAL_STATE['running']:
          step = True
      if _VIEWER_GLOBAL_STATE['single_step']:
        step = True
        _VIEWER_GLOBAL_STATE['single_step'] = False
    
      if step:
        dw = step_fn(mw, dw)
        get_data_into(d, m, dw)
      # mjx.get_data_into(d, m, dw)

      viewer.sync()

      elapsed = time.time() - start
      if elapsed < m.opt.timestep:
        time.sleep(m.opt.timestep - elapsed)


def get_data_into(
    result: Union[mujoco.MjData, list[mujoco.MjData]],
    m: mujoco.MjModel,
    d: Data,
    ):
      # batch size 1
      d_i = d
      result_i = result
      for field_name in (
              "qpos",
              "mocap_pos",
              "mocap_quat",
              "xanchor",
              "xaxis",
              "xmat",
              "xpos",
              "xquat",
              "xipos",
              "ximat",
              "subtree_com",
              "geom_xpos",
              "geom_xmat",
              "site_xpos",
              "site_xmat",
              "cinert",
              "cdof",
              "crb",
              "qM",
              "qLD",
              "qLDiagInv",
              ):
          value = getattr(d_i, field_name)

          if field_name == "qpos":
            result_field = getattr(result_i, field_name)
          if isinstance(value, wp.array) and value.shape:
          # if restricted_to in ('mujoco', 'mjx'):
            if field_name in [
                "nworld",
                ]:
              continue  # don't copy fields that are warp-only
            else:
              value = value.numpy().squeeze()
              result_field = getattr(result_i, field_name)
              if result_field.shape != value.shape:
                # reshape (might require transpose in some dim)
                value = value.reshape(result_field.shape)
                #print(
                #    f'Input field {field_name} has shape {value.shape}, but output'
                #    f' has shape {result_field.shape}'
                #    )
                #continue
                #raise ValueError(
                #    f'Input field {field_name} has shape {value.shape}, but output'
                #    f' has shape {result_field.shape}'
                #)
              result_field[:] = value
          else:
            setattr(result_i, field_name, value)

def main():
  app.run(_main)


if __name__ == "__main__":
  main()
