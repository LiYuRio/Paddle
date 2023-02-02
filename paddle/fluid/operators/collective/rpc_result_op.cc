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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/phi/common/pstring.h"

namespace paddle {
namespace operators {

class RpcResultOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class RpcResultOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("X", "(Tensor) tensor for output.");
    AddAttr<int>("request_id", "(int default 0) Unique id for request.")
        .SetDefault(0);
    AddAttr<std::string>("service_name",
                         "(string default service_name) Service name.")
        .SetDefault("service_name");
    AddComment(R"DOC(
RpcResultOpMaker Operator

)DOC");
  }
};

template <typename T>
class RpcResultOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "Run in Rpc result op";
    int request_id = ctx.Attr<int>("request_id");
    const auto& result =
        platform::RequestIdMap::Instance().GetRequestResult(request_id);
    VLOG(3) << result;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(rpc_result,
                             ops::RpcResultOp,
                             ops::RpcResultOpMaker);

REGISTER_OP_CPU_KERNEL(rpc_result,
                       ops::RpcResultOpKernel<float>,
                       ops::RpcResultOpKernel<phi::dtype::pstring>);
