# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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

import torch
import torch.nn as nn
import alf
from alf.networks import ActorDistributionNetwork
from alf.tensor_specs import BoundedTensorSpec
import functools
import time

if __name__ == '__main__':
    alf.set_default_device('cuda')

    CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

    actor_network_cls = functools.partial(
        ActorDistributionNetwork,
        fc_layer_params=(512, ),
        conv_layer_params=CONV_LAYER_PARAMS)

    actor_network = nn.DataParallel(actor_network_cls(
        input_tensor_spec=BoundedTensorSpec(
            shape=(4, 150, 150), dtype=torch.float32, minimum=0., maximum=1.),
        action_spec=BoundedTensorSpec(
            shape=(), dtype=torch.int64, minimum=0, maximum=3)))

    start_time = time.time()
    for i in range(1000):
        observation = torch.rand(640, 4, 150, 150)
        action_distribution, actor_state = actor_network(observation, state=())
    print(f'{time.time() - start_time} seconds elapsed')
