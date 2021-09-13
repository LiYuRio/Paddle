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

#include "paddle/fluid/framework/event_based_executor.h"
#include <string>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/runtime_graph.h"

namespace paddle {
namespace framework {

EventBasedExecutor::~EventBasedExecutor() {
  std::cout << "In EventBased Deconstructor" << std::endl;
}

void EventBasedExecutor::Compile(const ProgramDesc& program,
                                 const std::string& grain) {
  if (grain == "coarse") {
    CompileCoarseGrainGraph(program);
  } else {
    CompileFineGrainGraph(program);
  }
}

void EventBasedExecutor::CompileCoarseGrainGraph(const ProgramDesc& program) {
  runtime_graph_.reset(new RuntimeGraph(program));
  runtime_graph_->PrintGraph();
}

void EventBasedExecutor::CompileFineGrainGraph(const ProgramDesc& program) {
  std::cout << "Compile Fine Grain Graph" << std::endl;
}

void EventBasedExecutor::Run() {
  std::cout << "In Event Based Executor Run" << std::endl;
}
}  // namespace framework
}  // namespace paddle
