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

import os
import functools
import time
import multiprocessing as mp

import alf
from alf.networks import ActorDistributionNetwork
from alf.tensor_specs import BoundedTensorSpec

# DDP
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank: int):
    print(f'available devices: {torch.cuda.device_count()}')
    dist.init_process_group('gloo', rank=rank, world_size=2)
    print(f'Initialize worker on device {rank}')

    alf.set_default_device('cuda')
    CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

    print('ok')

    actor_network_cls = functools.partial(
        ActorDistributionNetwork,
        fc_layer_params=(512, ),
        conv_layer_params=CONV_LAYER_PARAMS)
    
    actor_network = DDP(actor_network_cls(
        input_tensor_spec=BoundedTensorSpec(
            shape=(4, 84, 84), dtype=torch.float32, minimum=0., maximum=1.),
        action_spec=BoundedTensorSpec(
            shape=(), dtype=torch.int64, minimum=0, maximum=3)).to(rank), device_ids=[rank])

    start_time = time.time()
    for i in range(1000):
        observation = torch.rand(64, 4, 84, 84)
        action_distribution, actor_state = actor_network(observation, state=())
    print(f'{time.time() - start_time} seconds elapsed on device {rank}')


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    processes = []
    for i in range(2):
        processes.append(mp.Process(target=worker, args=(i,)))
        processes[i].start()

    for proc in processes:
        proc.join()
