#include "paddle/fluid/distributed/fleet_executor/while_interceptor.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace distributed {

WhileInterceptor::WhileInterceptor(int64_t interceptor_id, TaskNode* node) 
  : Interceptor(interceptor_id, node) {
  RegisterMsgHandle([this](const InterceptorMessage& msg) { Compute(msg); });
}

bool WhileInterceptor::CheckCondition(int32_t scope_idx) {
  PADDLE_ENFORCE_EQ(node_->ops().size(), 1, "Task for while interceptor should only include while op");
  auto* while_op = node_->ops()[0];
  framework::Scope* scope = microbatch_scopes_[scope_idx];
  auto &cond = scope->FindVar(while_op->Input(operators::kCondition))->Get<framework::LoDTensor>();
  return operators::GetCondData(cond);
}

void WhileInterceptor::SendDataReadyToDownStream(int32_t scope_idx) {
  InterceptorMessage ready_msg;
  ready_msg.set_message_type(DATA_IS_READY);
  ready_msg.set_scope_idx(scope_idx);
  PADDLE_ENFORCE_NE(sub_block_src_interceptor_id_, -1, "Source interceptor id has not been set.");
  Send(sub_block_src_interceptor_id_, ready_msg);
}

void WhileInterceptor::RunInScope(int32_t scope_idx) {
  bool cond_flag = CheckCondition(scope_idx);
  if (cond_flag) {
    SendDataReadyToDownStream(scope_idx);
  } else {
    Interceptor::SendDataReadyToDownStream();
    ReplyCompletedToUpStream();
  }
}

void WhileInterceptor::Run() {
  while (IsInputReady() && CanWriteOutput()) {
    ++step_;
    RunInScope(step_ % node_->max_run_times());
  }
}

void WhileInterceptor::Compute(const InterceptorMessage& msg) {
  int64_t src_id = msg.src_id();
  if (msg.message_type() == DATA_IS_READY) {
    IncreaseReady(src_id);
    Run();
  } else if (msg.message_type() == DATA_IS_USELESS) {
    if (src_id == sub_block_sink_interceptor_id_) {
      // Message is from the last interceptor of sub-block
      int32_t scope_idx = msg.scope_idx();
      RunInScope(scope_idx);
    } else {
      // Message is from the op after while
      DecreaseBuff(src_id);
      Run();
    }
  } else if (msg.message_type() == STOP) {
  }
  
  // TryStop();
}

REGISTER_INTERCEPTOR(While, WhileInterceptor);

} // namespace distributed
} // namespace paddle
