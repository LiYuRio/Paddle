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

#include "brpc/channel.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/collective_helper.h"

namespace paddle {
namespace operators {

class RpcCallOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class RpcCallOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<int>("request_id", "(int default 0) Unique id for request.")
        .SetDefault(0);
    AddAttr<std::string>("url", "(string default url) Service url.")
        .SetDefault("url");
    AddAttr<std::string>("service_name",
                         "(string default service_name) Service name.")
        .SetDefault("service_name");
    AddAttr<std::string>("request",
                         "(string default request) Request to service.")
        .SetDefault("request");
    AddComment(R"DOC(
RpcCallOpMaker Operator

)DOC");
  }
};

template <typename T>
class RpcCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "Run in Rpc call op";
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.connect_timeout_ms = 1000;
    options.timeout_ms = 1000;
    options.max_retry = 5;
    const std::string& url = ctx.Attr<std::string>("url");
    int request_id = ctx.Attr<int>("request_id");
    VLOG(3) << "Plan to send request to remote server " << url;
    PADDLE_ENFORCE_EQ(channel.Init(url.c_str(), &options),
                      0,
                      platform::errors::Unavailable(
                          "Rpc call op failed: init brpc channel error."));
    platform::RequestIdMap::Instance().Insert(request_id, "");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(rpc_call, ops::RpcCallOp, ops::RpcCallOpMaker);

REGISTER_OP_CPU_KERNEL(rpc_call, ops::RpcCallOpKernel<float>);
