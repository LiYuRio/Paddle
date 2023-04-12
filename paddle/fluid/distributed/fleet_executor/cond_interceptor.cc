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

#include "paddle/fluid/distributed/fleet_executor/cond_interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace distributed {

CondInterceptor::CondInterceptor(int64_t interceptor_id, TaskNode* node)
    : Interceptor(interceptor_id, node) {
  total_num_of_scopes_ = node->max_run_times();
  PrepareDeps();
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Run(msg); });
}

void CondInterceptor::PrepareDeps() {
  auto& upstream = node_->upstream();
  auto& downstream = node_->downstream();
  auto& id_to_dep_type = node_->id_to_dep_type();

  for (const auto& up : upstream) {
    if (id_to_dep_type.at(up.first) == DependType::NORMAL) {
      normal_in_id_.insert(up.first);
      generation_step_ = up.second;
    } else if (id_to_dep_type.at(up.first) == DependType::LOOP) {
      loop_id_ = up.first;
    }
  }

  for (const auto& down : downstream) {
    if (id_to_dep_type.at(down.first) == DependType::NORMAL) {
      normal_out_id_.insert(down.first);
    } else if (id_to_dep_type.at(down.first) == DependType::STOP_LOOP) {
      stop_loop_id_ = down.first;
    }
  }
}

bool CondInterceptor::GetCondResult(int64_t scope_id) {
  PADDLE_ENFORCE_LT(scope_id,
                    microbatch_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "Step out of range. There are %ld "
                        "microbatch_scopes, but recevice scope index %ld",
                        microbatch_scopes_.size(),
                        scope_id));
  auto* cond_var = microbatch_scopes_[scope_id]->FindVar(node_->cond_var());
  PADDLE_ENFORCE(cond_var,
                 platform::errors::NotFound(
                     "Condition variable %s not exists in scope %ld",
                     node_->cond_var(),
                     scope_id));
  const auto& cond_tensor = cond_var->Get<phi::DenseTensor>();
  bool res = false;
  if (platform::is_gpu_place(cond_tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::DenseTensor cpu_tensor;
    framework::TensorCopy(cond_tensor, platform::CPUPlace(), &cpu_tensor);
    platform::DeviceContextPool::Instance().Get(cond_tensor.place())->Wait();
    res = cpu_tensor.data<bool>()[0];
#endif
  } else if (platform::is_cpu_place(cond_tensor.place())) {
    res = cond_tensor.data<bool>()[0];
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupport device for cond interceptor."));
  }
  return res;
}

void CondInterceptor::SendDataReady(int64_t down_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_READY);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(down_id, ready_msg);
}

void CondInterceptor::SendStartLoop(int64_t down_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(START_LOOP);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(down_id, ready_msg);
}

void CondInterceptor::ReplyDataIsUseless(int64_t up_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_USELESS);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(up_id, ready_msg);
}

void CondInterceptor::Compute() {
  VLOG(3) << "Loop again in scope " << cur_scope_id_;
  for (auto& down_id : normal_out_id_) {
    SendStartLoop(down_id);
  }
}

void CondInterceptor::ComputeAfterGen() {
  VLOG(3) << "Finish loop in scope " << cur_scope_id_ << " with "
          << scope_id_to_gen_step_.at(cur_scope_id_) << " generation steps.";
  // Clear the finish scope from map
  scope_id_to_gen_step_.erase(cur_scope_id_);
  SendDataReady(stop_loop_id_);
  for (auto& up_id : normal_in_id_) {
    ReplyDataIsUseless(up_id);
  }
  // Gc the variable in while block
  if (gc_) {
    VLOG(3) << "Release vars in while block in scope " << cur_scope_id_;
    framework::DeleteUnusedTensors(*microbatch_scopes_[cur_scope_id_],
                                   node_->while_block_vars(),
                                   gc_.get());
  }
}

void CondInterceptor::Run(const InterceptorMessage& msg) {
  if (msg.message_type() == DATA_IS_READY) {
    VLOG(3) << "Receving data ready message from " << msg.src_id() << " scope "
            << msg.scope_idx();
    scope_counter_++;
    auto scope_id = msg.scope_idx();
    bool cond = GetCondResult(scope_id);
    if (cond) {
      ++num_of_generation_;
      ready_scope_id_.emplace_back(scope_id);
    } else {
      cur_scope_id_ = scope_id;
      ComputeAfterGen();
    }

    if (num_of_generation_ == generation_step_ ||
        scope_counter_ % total_num_of_scopes_ == 0) {
      std::sort(ready_scope_id_.begin(), ready_scope_id_.end());
      for (auto& scope_id : ready_scope_id_) {
        cur_scope_id_ = scope_id;
        if (scope_id_to_gen_step_.find(scope_id) !=
            scope_id_to_gen_step_.end()) {
          scope_id_to_gen_step_.at(scope_id) =
              scope_id_to_gen_step_.at(scope_id) + 1;
        } else {
          scope_id_to_gen_step_.emplace(scope_id, 0);
        }
        Compute();
      }
      ready_scope_id_.clear();
      finish_scope_id_.clear();
    }
  } else if (msg.message_type() == DATA_WITH_VARS) {
    int64_t scope_id = msg.scope_idx();
    VLOG(3) << "Receving loop again message from " << msg.src_id()
            << " in scope " << scope_id;
    ++counter_;
    still_running_scope_id_.emplace_back(scope_id);
    if (counter_ == num_of_generation_) {
      std::sort(still_running_scope_id_.begin(), still_running_scope_id_.end());
      for (auto& scope_id : still_running_scope_id_) {
        bool cond = GetCondResult(scope_id);
        // bool not_early_stop = !(scope_id % 4 == 2 &&
        // scope_id_to_gen_step_[scope_id] == 5);
        if (cond) {
          ready_scope_id_.emplace_back(scope_id);
        } else {
          VLOG(3) << "Start test early stop generation.";
          finish_scope_id_.emplace_back(scope_id);
        }
      }
      still_running_scope_id_.clear();
      counter_ = 0;

      for (auto& scope_id : finish_scope_id_) {
        cur_scope_id_ = scope_id;
        ComputeAfterGen();
        --num_of_generation_;
      }

      if (finish_scope_id_.empty() ||
          scope_counter_ % total_num_of_scopes_ == 0) {
        for (auto& scope_id : ready_scope_id_) {
          cur_scope_id_ = scope_id;
          scope_id_to_gen_step_.at(scope_id) =
              scope_id_to_gen_step_.at(scope_id) + 1;
          Compute();
        }
        ready_scope_id_.clear();
        finish_scope_id_.clear();
      }
    }
  }
}

REGISTER_INTERCEPTOR(Cond, CondInterceptor);

}  // namespace distributed
}  // namespace paddle
