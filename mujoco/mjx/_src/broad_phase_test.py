# Copyright 2025 The Physics-Next Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for broad phase functions."""

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from mujoco import mjx
import numpy as np
import warp as wp

from . import test_util


class BroadPhaseTest(parameterized.TestCase):
  def test_broad_phase(self):
    """Tests broad phase."""
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml")

    mjx.broadphase(m, d)


if __name__ == "__main__":
  wp.init()
  absltest.main()
