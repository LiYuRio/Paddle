#pragma once

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"

namespace paddle {
namespace distributed {

class WhileInterceptor : public Interceptor {
 public:
  WhileInterceptor(int64_t interceptor_id, TaskNode* node);
   
 private:
  void Compute(const InterceptorMessage& msg);
  bool CheckCondition(int32_t scope_idx);
  void SendDataReadyToDownStream(int32_t scope_idx);
  void RunInScope(int32_t scope_idx);
  void Run();

  int64_t sub_block_src_interceptor_id_{-1};
  int64_t sub_block_sink_interceptor_id_{-1};
};

} // namespace distributed
} // namespace paddle
