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

import torch.multiprocessing as mp

import alf
from alf.networks import ActorDistributionNetwork
from alf.tensor_specs import BoundedTensorSpec

# DDP
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank: int, start_time, device_count: int):
    if device_count > 1:
        dist.init_process_group('nccl', rank=rank, world_size=2)
    print(f'Initialize worker on device {rank}')

    batch_size = 256 // device_count

    alf.set_default_device('cuda')
    CONV_LAYER_PARAMS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

    actor_network_cls = functools.partial(
        ActorDistributionNetwork,
        fc_layer_params=(512, ),
        conv_layer_params=CONV_LAYER_PARAMS)

    if device_count > 1:
        actor_network = DDP(actor_network_cls(
            input_tensor_spec=BoundedTensorSpec(
                shape=(4, 84, 84), dtype=torch.float32, minimum=0., maximum=1.),
            action_spec=BoundedTensorSpec(
                shape=(), dtype=torch.int64, minimum=0, maximum=3)).to(rank), device_ids=[rank])
    else:
        actor_network = actor_network_cls(
            input_tensor_spec=BoundedTensorSpec(
                shape=(4, 84, 84), dtype=torch.float32, minimum=0., maximum=1.),
            action_spec=BoundedTensorSpec(
                shape=(), dtype=torch.int64, minimum=0, maximum=3)).to(rank)

    for i in range(5000):
        observation = torch.rand(batch_size, 4, 84, 84, device=rank)
        action_distribution, actor_state = actor_network(observation, state=())
        action = action_distribution.sample()
        reward = torch.rand(batch_size, device=rank)
        loss = - torch.mean(action_distribution.log_prob(action) * reward)
        loss.backward()
        if i % 100 == 0:
            print(f'iteration {i} - {time.time() - start_time} seconds elapsed on device {rank}')
    print(f'{time.time() - start_time} seconds elapsed on device {rank}')


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    processes = []
    start_time = time.time()

    if os.getenv('DEVID') is None:
        print(f'available devices: {torch.cuda.device_count()}')

        n = torch.cuda.device_count()

        # Run the distributed version
        for i in range(n):
            processes.append(mp.Process(target=worker, args=(i, start_time, n)))
            processes[i].start()

        for proc in processes:
            proc.join()
    else:
        worker(int(os.getenv('DEVID')), start_time, 1)

