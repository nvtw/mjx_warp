from absl.testing import absltest
from etils import epath
import mujoco
from mujoco import mjx
import numpy as np

from . import broadphase


def _compare_contacts(test_cls, d, c):
  """Compares JAX and Mujoco contacts."""
  c_geom1 = c.geom1.numpy()
  c_geom2 = c.geom2.numpy()
  c_geom = np.stack((c_geom1, c_geom2), axis=1)
  c_dist = c.dist.numpy()
  c_frame = c.frame.numpy()
  c_pos = c.pos.numpy()
  for env_id, (g1, g2) in enumerate(zip(d.contact.geom, c_geom)):
    for g1_key in np.unique(g1, axis=0):
      idx1 = np.where((g1 == g1_key).all(axis=1))
      idx2 = np.where((g2 == g1_key).all(axis=1))
      dist1 = d.contact.dist[env_id][idx1]
      dist2 = c_dist[env_id][idx2]
      # contacts may appear in JAX with dist>0, but not in CUDA.
      if (dist1 > 0).any():
        if dist2.shape[0]:
          test_cls.assertTrue((dist2 >= 0).any())
        continue
      test_cls.assertTrue((dist1 < 0).all())
      # contact distance in JAX are dynamically calculated, so we only
      # check that CUDA distances are equal to the first JAX distance.
      np.testing.assert_array_almost_equal(dist1[0], dist2, decimal=3)
      # normals should be equal.
      normal1 = d.contact.frame[env_id, :, 0][idx1]
      normal2 = c_frame[env_id, :, 0][idx2]
      test_cls.assertLess(np.abs(normal1[0] - normal2).max(), 1e-5)
      # contact points are not as accurate in CUDA, the test is rather loose.
      found_point = 0
      pos1 = d.contact.pos[env_id][idx1]
      pos2 = c_pos[env_id][idx2]
      for pos1_idx in range(pos1.shape[0]):
        pos2_idx = np.abs(pos1[pos1_idx] - pos2).sum(axis=1).argmin()
        found_point += np.abs(pos1[pos1_idx] - pos2[pos2_idx]).max() < 0.11
      test_cls.assertGreater(found_point, 0)


class EngineCollisionDriverTest(absltest.TestCase):

  _CONVEX_CONVEX = """
    <mujoco>
      <asset>
        <mesh name="meshbox"
              vertex="-1 -1 -1
                      1 -1 -1
                      1  1 -1
                      1  1  1
                      1 -1  1
                      -1  1 -1
                      -1  1  1
                      -1 -1  1"/>
        <mesh name="poly" scale="0.5 0.5 0.5"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
        <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
          vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
          face="0 1 2  0 3 1  0 2 3  1 3 2"/>
      </asset>
      <custom>
        <numeric data="12" name="max_contact_points"/>
      </custom>
      <worldbody>
        <light pos="-.5 .7 1.5" cutoff="55"/>
        <body pos="0.0 2.0 0.35" euler="0 0 90">
          <freejoint/>
          <geom type="mesh" mesh="meshbox"/>
        </body>
        <body pos="0.0 2.0 1.781" euler="180 0 0">
          <freejoint/>
          <geom type="mesh" mesh="poly"/>
          <geom pos="0.5 0 -0.2" type="sphere" size="0.3"/>
        </body>
        <body pos="0.0 2.0 2.081">
          <freejoint/>
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
        <body pos="0.0 0.0 -2.0">
          <geom type="plane" size="40 40 40"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_shapes(self):
    """Tests collision driver return shapes."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    batch_size = 12

    # loop right now, can batch later
    for val in np.arange(-1, 1, 2 / batch_size):
      d = mujoco.MjData(m)
      d.qpos[2] = val

      mujoco.mj_forward(m, d)

      mx = mjx.put_model(m)
      dx = mjx.put_data(m, d)

      c = broadphase.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

      # from CPU forward
      npts = d.contact.pos.shape[1]
      self.assertTupleEqual(c.dist.shape, (npts,))
      self.assertTupleEqual(c.pos.shape, (npts, 3))
      self.assertTupleEqual(c.frame.shape, (npts, 3, 3))
      self.assertTupleEqual(c.friction.shape, (npts, 5))
      self.assertTupleEqual(c.solimp.shape, (npts, mujoco.mjNIMP))
      self.assertTupleEqual(c.solref.shape, (npts, mujoco.mjNREF))
      self.assertTupleEqual(
          c.solreffriction.shape, (npts, mujoco.mjNREF)
      )
      # self.assertTupleEqual(c.geom.shape, (npts, 2)) not implemented right now
      self.assertTupleEqual(c.geom1.shape, (npts))
      self.assertTupleEqual(c.geom2.shape, (npts))

  def test_contacts_batched_model_data(self):
    """Tests collision driver results."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    batch_size = 3

    # looping, can batch later
    for val in np.arange(-1, 1, 2 / batch_size):
      d = mujoco.MjData(m)
      d.qpos[2] = val

      mujoco.mj_forward(m, d)

      mx = mjx.put_model(m)
      dx = mjx.put_data(m, d)

      c = broadphase.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

      # test contact normals and penetration depths.
      _compare_contacts(self, d, c)

  def test_contacts_batched_data(self):
    """Tests collision driver results."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    batch_size = 3

    # looping, can batch later
    for val in np.arange(-1, 1, 2 / batch_size):
      d = mujoco.MjData(m)
      d.qpos[2] = val

      mujoco.mj_forward(m, d)

      mx = mjx.put_model(m)
      dx = mjx.put_data(m, d)

      c = broadphase.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

      # test contact normals and penetration depths.
      _compare_contacts(self, d, c)

  _CONVEX_CONVEX_2 = """
    <mujoco>
      <asset>
        <mesh name="poly" scale="0.5 0.5 0.5"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
        <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
          vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
          face="0 1 2  0 3 1  0 2 3  1 3 2"/>
      </asset>
      <custom>
        <numeric data="2" name="max_contact_points"/>
      </custom>
      <worldbody>
        <light pos="-.5 .7 1.5" cutoff="55"/>
        <body pos="0.0 2.0 1.781" euler="180 0 0">
          <freejoint/>
          <geom type="mesh" mesh="poly"/>
        </body>
        <body pos="0.0 2.0 2.081">
          <freejoint/>
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_solparams(self):
    """Tests collision driver solparams."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    batch_size = 3

    # looping, can batch later
    for val in np.arange(0.1, 1.1, 1 / batch_size):
      m.geom_solref[0] = val
      m.geom_solimp[0] = val
      m.geom_friction[0] = val
      d = mujoco.MjData(m)

      mujoco.mj_forward(m, d)

      mx = mjx.put_model(m)
      dx = mjx.put_data(m, d)

      c = broadphase.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

      np.testing.assert_array_almost_equal(c.solref, d.contact.solref)
      np.testing.assert_array_almost_equal(c.solimp, d.contact.solimp)
      np.testing.assert_array_almost_equal(c.friction, d.contact.friction)


if __name__ == "__main__":
  absltest.main()