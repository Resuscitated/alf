# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test cases adpated from tf_agents' py_environment_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import unittest

from alf.tensor_specs import TensorSpec, BoundedTensorSpec
from alf.environments.random_torch_environment import RandomTorchEnvironment
from alf.nest import nest


class TorchEnvironmentTest(unittest.TestCase):
    def testResetSavesCurrentTimeStep(self):
        obs_spec = BoundedTensorSpec((1, ), torch.int32)
        action_spec = BoundedTensorSpec((1, ), torch.int64)

        random_env = RandomTorchEnvironment(
            observation_spec=obs_spec, action_spec=action_spec)

        time_step = random_env.reset()
        current_time_step = random_env.current_time_step()
        nest.map_structure(self.assertEqual, time_step, current_time_step)

    def testStepSavesCurrentTimeStep(self):
        obs_spec = BoundedTensorSpec((1, ), torch.int32)
        action_spec = BoundedTensorSpec((1, ), torch.int64)

        random_env = RandomTorchEnvironment(
            observation_spec=obs_spec, action_spec=action_spec)

        random_env.reset()
        time_step = random_env.step(action=torch.ones((1, )))
        current_time_step = random_env.current_time_step()
        nest.map_structure(self.assertEqual, time_step, current_time_step)


if __name__ == '__main__':
    unittest.main()