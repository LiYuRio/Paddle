// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/distributed/auto_parallel/copy_reshard_function.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard_utils.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace distributed {

bool CopyReshardFunction::IsSuitable(const DistTensor& in,
                                     const TensorDistAttr& out_dist_attr) {
  bool flag = true;
  flag &= (in.dist_attr() == out_dist_attr);
  return flag;
}

void CopyReshardFunction::Eval(phi::DeviceContext* dev_ctx,
                               const DistTensor& in,
                               const TensorDistAttr& out_dist_attr,
                               DistTensor* out) {
  VLOG(3) << "Call CopyReshardFunction Eval";

  RESHARD_FUNCTOR_WITHOUT_DTYPE(dev_ctx,
                                Copy,
                                in.value(),
                                in.place(),
                                /*blocking*/ true,
                                GetMutableTensor(out));
  SetDistProps(out, in.dims(), out_dist_attr);
}

REGISTER_RESHARD_FUNC(CopyReshardFunction);

}  // namespace distributed
}  // namespace phi
