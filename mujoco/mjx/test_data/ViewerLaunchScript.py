import mujoco
from mujoco.viewer import launch

model = mujoco.MjModel.from_xml_path("convex/convex_plane.xml")  # Replace with your file path
data = mujoco.MjData(model)

with launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()