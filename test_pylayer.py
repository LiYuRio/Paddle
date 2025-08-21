# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.autograd import PyLayer


class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return {"y": y}

    @staticmethod
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad


paddle.seed(2023)
data = paddle.randn([2, 3], dtype="float64")
data.stop_gradient = False
z = cus_tanh.apply(data)
print(z)
z.mean().backward()

print(data.grad)
