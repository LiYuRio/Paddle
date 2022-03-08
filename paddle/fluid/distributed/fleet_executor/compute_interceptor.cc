// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/compute_interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/carrier.h"

#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

ComputeInterceptor::ComputeInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

void ComputeInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();

  for (auto up : upstream) {
    in_readys_.emplace(up.first, std::make_pair(up.second, 0));
    in_stops_.emplace(up.first, false);
  }
  for (auto down : downstream) {
    out_buffs_.emplace(down.first, std::make_pair(down.second, 0));
  }

  // source compute node, should we add a new SourceInterceptor?
  if (upstream.empty()) {
    is_source_ = true;
    PADDLE_ENFORCE_GT(node_->max_run_times(), 0,
                      platform::errors::InvalidArgument(
                          "Source ComputeInterceptor must run at least one "
                          "times, but now max_run_times=%ld",
                          node_->max_run_times()));
    in_readys_.emplace(-1,
                       std::make_pair(std::numeric_limits<int64_t>::max(), 0));
  }

  // If there is no downstream or every downstream is in different rank,
  // then this interceptor is the last one for current rank.
  // This can be get during init, can be cached for later use.
  is_last_ = downstream.empty();
}

void ComputeInterceptor::RunOps() {
  VLOG(3) << "ComputeInterceptor " << interceptor_id_ << " running ops for the "
          << step_ + 1 << " time.";
  for (auto op : node_->ops()) {
    op->Run(*microbatch_scopes_[step_ % node_->max_run_times()], place_);
    if (gc_) {
      framework::DeleteUnusedTensors(
          *microbatch_scopes_[step_ % node_->max_run_times()], op,
          node_->unused_vars(), gc_.get());
    }
  }
}

void ComputeInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    VLOG(3) << "id=" << GetInterceptorId() << " ComputeInterceptor running";

    RunOps();
    ++step_;

    // send to downstream and increase buff used
    SendDataReadyToDownStream();
    // reply to upstream and decrease ready data
    ReplyCompletedToUpStream();
    // Try to stop Carrier
    if (is_last_ && (step_ % node_->max_run_times() == 0)) {
      VLOG(3) << "Interceptor " << GetInterceptorId()
              << " is stopping carrier.";
      // FIXME(wangxi): with multi sink interceptor
      StopCarrier();
    }
  }
}

void ComputeInterceptor::ReceivedStop(int64_t up_id) {
  received_stop_ = true;

  // source node has no upstream, stop is send by carrier or others
  if (is_source_ && up_id == -1) return;

  auto it = in_stops_.find(up_id);
  PADDLE_ENFORCE_NE(it, in_stops_.end(),
                    platform::errors::NotFound(
                        "Cannot find upstream=%lld in in_stops.", up_id));
  PADDLE_ENFORCE_EQ(
      it->second, false,
      platform::errors::AlreadyExists("Already received stop from %lld, stop "
                                      "cannot be send more than once."));
  it->second = true;
}

void ComputeInterceptor::TryStop() {
  if (!received_stop_) return;

  // can stop only when all upstream is stop and
  // downstream complete
  for (auto& in_stop : in_stops_) {
    if (!in_stop.second) return;
  }
  for (auto& out_buff : out_buffs_) {
    auto used_size = out_buff.second.second;
    if (used_size != 0) return;
  }

  // send stop to downstream
  for (auto& out : out_buffs_) {
    auto down_id = out.first;
    InterceptorMessage stop;
    stop.set_message_type(STOP);
    Send(down_id, stop);
  }
  stop_ = true;
}

void ComputeInterceptor::Compute(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    IncreaseReady(msg.src_id());
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    DecreaseBuff(msg.src_id());
    Run();
  } else if (msg.message_type() == STOP) {
    ReceivedStop(msg.src_id());
  }

  TryStop();
}

REGISTER_INTERCEPTOR(Compute, ComputeInterceptor);

}  // namespace distributed
}  // namespace paddle
