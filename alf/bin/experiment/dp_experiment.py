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

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 384)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(384, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.fc1(x)
        h = self.relu1(h)
        h = self.fc2(h)
        h = self.relu2(h)
        h = self.fc3(h)
        h = self.relu3(h)
        h = self.fc4(h)
        h = self.sigmoid(h)
        return h


if __name__ == '__main__':
    input_size = 128
    output_size = 16
    model = Network(input_size, output_size).cuda()
    # model = nn.DataParallel(model).cuda()

    optimizer = optim.SGD(params=model.parameters(), lr=1e-3)
    cls_criterion = nn.BCELoss()

    # Prepare the data
    data = torch.randn(256000, input_size)
    targets = torch.empty(256000, output_size).random_(2).cuda().view(-1, output_size)

    start_time = time.time()
    output = model(data.cuda())
    optimizer.zero_grad()
    loss = cls_criterion(output, targets)
    loss.backward()
    optimizer.step()
    print(f'{time.time() - start_time} seconds')
