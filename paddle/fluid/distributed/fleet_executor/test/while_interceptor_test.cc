#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
 
USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(fill_constant);
USE_OP_ITSELF(less_than);
USE_OP_ITSELF(while);

namespace paddle {
namespace distributed {

framework::Scope* PrepareScope() {
  framework::Scope* scope = new framework::Scope();
  scope->Var("cond")->GetMutable<framework::LoDTensor>();
  scope->Var("i")->GetMutable<framework::LoDTensor>();
  scope->Var("n")->GetMutable<framework::LoDTensor>();
  scope->Var("step")->GetMutable<framework::LoDTensor>();
  return scope;
}

std::vector<framework::OperatorBase*> PreparePrevOps() {
  framework::AttributeMap attrs;
  attrs["dtype"] = framework::proto::VarType::INT32;
  attrs["shape"] = phi::vectorize<int>({1});
  attrs["value"] = 0;
  auto fill_i = framework::OpRegistry::CreateOp("fill_constant", {},
                                                 {{"Out", {"i"}}}, attrs);
  attrs["value"] = 5;
  auto fill_n = framework::OpRegistry::CreateOp("fill_constant", {},
                                                 {{"Out", {"n"}}}, attrs);
  attrs["value"] = 1;
  auto fill_step = framework::OpRegistry::CreateOp("fill_constant", {},
                                                 {{"Out", {"step"}}}, attrs);
  auto less_than = framework::OpRegistry::CreateOp("less_than", {{"X", {"i"}}, {"Y", {"n"}}}, 
      {{"Out", {"cond"}}}, framework::AttributeMap());
  return {fill_i.release(), fill_n.release(), fill_step.release(), less_than.release()};
}

std::vector<framework::OperatorBase*> PrepareWhileCondOps() {
  auto while_op = framework::OpRegistry::CreateOp("while", {{"Condition", {"cond"}}}, 
      {}, framework::AttributeMap());
  return {while_op.release()};
}

std::vector<framework::OperatorBase*> PrepareUpdateOps() {
  auto increase = framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"i"}}, {"Y", {"step"}}}, {{"Out", {"i"}}},
      framework::AttributeMap());
  auto less_than = framework::OpRegistry::CreateOp("less_than", {{"X", {"i"}}, {"Y", {"n"}}}, 
      {{"Out", {"cond"}}}, framework::AttributeMap());
  return {increase.release()};
}

TEST(WhileInterceptor, While) {
  // Prepare scopes
  framework::Scope* scope = PrepareScope();
  std::vector<framework::Scope*> scopes = {scope, scope};
  platform::Place place = platform::CPUPlace();
  // Prepare ops
  auto prev_ops = PreparePrevOps(); 
  auto cond_ops = PrepareWhileCondOps(); 
  auto update_ops = PrepareUpdateOps(); 

  // Prepare task nodes, "max_run_times" is 2 represent 2 micro-batch
  TaskNode* node_a = new TaskNode(0, prev_ops, 0, 0, 2, 0);
  TaskNode* node_b = new TaskNode(0, cond_ops, 0, 1, 2, 0);
  TaskNode* node_c = new TaskNode(0, update_ops, 0, 2, 2, 0);
  TaskNode* node_d = new TaskNode(0, 0, 2, 2, 0);

  // Add dependence link
  // a -> b -> c
  //      |
  //      d
  node_a->AddDownstreamTask(1);
  node_b->AddUpstreamTask(0);
  node_b->AddDownstreamTask(2);
  node_b->AddDownstreamTask(3);
  node_c->AddUpstreamTask(1);
  node_d->AddUpstreamTask(1);

  // Init fleet executor environment
  std::string carrier_id = "0";
  Carrier* carrier = GlobalMap<std::string, Carrier>::Create(carrier_id, carrier_id);
  carrier->Init(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}});
  MessageBus* msg_bus = GlobalVal<MessageBus>::Create();
  msg_bus->Init(0, {{0, "127.0.0.0:0"}}, "");

  // Create interceptors
  auto* a = carrier->SetInterceptor(0, InterceptorFactory::Create("Compute", 0, node_a));
  carrier->SetInterceptor(1, InterceptorFactory::Create("While", 1, node_b));
  carrier->SetInterceptor(2, InterceptorFactory::Create("Compute", 2, node_c));
  carrier->SetInterceptor(3, InterceptorFactory::Create("Compute", 3, node_d));
  a->SetPlace(place);
  a->SetMicroBatchScope(scopes);

  // Start execution
  InterceptorMessage msg;
  msg.set_message_type(DATA_IS_READY);
  msg.set_src_id(-1);
  msg.set_dst_id(0);
  carrier->EnqueueInterceptorMessage(msg);

  // Environment tears down
  carrier->Wait();
  carrier->Release();
}  
} // namespace distributed
} // namespace paddle
