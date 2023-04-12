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
#include <algorithm>
#include <ostream>
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

void CondInterceptor::SendStartLoop(int64_t down_id, int64_t gen_step) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(START_LOOP);
  ready_msg.set_scope_idx(cur_scope_id_);
  ready_msg.set_gen_step(gen_step);
  Send(down_id, ready_msg);
}

void CondInterceptor::ReplyDataIsUseless(int64_t up_id) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_USELESS);
  ready_msg.set_scope_idx(cur_scope_id_);
  Send(up_id, ready_msg);
}

void CondInterceptor::Compute(int64_t gen_step) {
  VLOG(3) << "Loop again in scope " << cur_scope_id_ << " with gen_step "
          << gen_step;
  for (auto& down_id : normal_out_id_) {
    SendStartLoop(down_id, gen_step);
  }
}

void CondInterceptor::ComputeAfterGen() {
  VLOG(3) << "Finish loop in scope " << cur_scope_id_ << " with "
          << scope_id_to_gen_step_.at(cur_scope_id_) << " generation steps.";
  // Clear the finish scope from map
  scope_id_to_gen_step_.erase(cur_scope_id_);
  scope_id_to_compute_gen_step_.erase(cur_scope_id_);
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
        scope_id_to_compute_gen_step_[scope_id] = 0;
        Compute(0);
      }
      ready_scope_id_.clear();
      finish_scope_id_.clear();
      start_to_record_ = false;
    }
  } else if (msg.message_type() == DATA_WITH_VARS) {
    int64_t scope_id = msg.scope_idx();
    VLOG(3) << "Receving loop again message from " << msg.src_id()
            << " in scope " << scope_id;
    bool cond = GetCondResult(scope_id);
    // bool early_stop = scope_id % 4 == 2 && scope_id_to_gen_step_[scope_id] ==
    // 5;
    if (!cond) {
      VLOG(3) << "Start to record finish scope";
      start_to_record_ = true;
    }
    if (start_to_record_) {
      if (cond) {
        ready_scope_id_.emplace_back(scope_id);
      } else {
        finish_scope_id_.emplace_back(scope_id);
      }
      bool finish_capture =
          static_cast<int>(ready_scope_id_.size() + finish_scope_id_.size()) ==
          num_of_generation_;
      if (finish_capture || scope_counter_ % total_num_of_scopes_ == 0) {
        for (auto& scope_id : finish_scope_id_) {
          cur_scope_id_ = scope_id;
          ComputeAfterGen();
          --num_of_generation_;
        }
        if (scope_counter_ % total_num_of_scopes_ == 0) {
          std::sort(ready_scope_id_.begin(), ready_scope_id_.end());
          for (auto& scope_id : ready_scope_id_) {
            cur_scope_id_ = scope_id;
            Compute(scope_id_to_compute_gen_step_.at(scope_id));
          }
          ready_scope_id_.clear();
          finish_scope_id_.clear();
          start_to_record_ = false;
        }
      }
    } else {
      int64_t scope_id = msg.scope_idx();
      PADDLE_ENFORCE_NE(
          scope_id_to_compute_gen_step_.find(scope_id),
          scope_id_to_compute_gen_step_.end(),
          platform::errors::InvalidArgument(
              "Can not find scope id %ld in scope_id_to_tmp_gen_step",
              scope_id));
      // Keep the message in order with scope_id
      // message with scope 3 never send before scope 1.
      int64_t gen_step = scope_id_to_compute_gen_step_.at(scope_id) + 1;
      bool wait_prev_scope = false;
      // If the previous scope gen_step less than cur scope
      // means: the previous scope doesn't finish last step generation, should
      // wait.
      auto iter = scope_id_to_compute_gen_step_.begin();
      while (iter != scope_id_to_compute_gen_step_.end()) {
        if (iter->first == scope_id) {
          break;
        }
        if (iter->second < gen_step) {
          wait_prev_scope = true;
          break;
        }
        ++iter;
      }
      scope_id_to_compute_gen_step_.at(scope_id) = gen_step;
      scope_id_to_gen_step_.at(scope_id) =
          scope_id_to_gen_step_.at(scope_id) + 1;
      if (!wait_prev_scope) {
        // Start send message to all scopes gen_step equal to cur_scope
        std::vector<int64_t> ready_scope_ids;
        while (iter != scope_id_to_compute_gen_step_.end()) {
          if (iter->second == gen_step) {
            ready_scope_ids.emplace_back(iter->first);
          } else if (iter->second > gen_step) {
            PADDLE_THROW(platform::errors::Fatal(
                "Some error may occur. Scope %ld's "
                "gen_step is much larger than previous with %ld.",
                iter->first,
                iter->second));
          } else {
            break;
          }
          ++iter;
        }
        for (auto& scope_id : ready_scope_ids) {
          cur_scope_id_ = scope_id;
          Compute(gen_step);
        }
      }
    }
  }
}

REGISTER_INTERCEPTOR(Cond, CondInterceptor);

}  // namespace distributed
}  // namespace paddle
