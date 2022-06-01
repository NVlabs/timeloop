/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "model/engine.hpp"

namespace model
{

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.
Engine::Specs Engine::ParseSpecs(config::CompoundConfigNode setting, bool is_sparse_topology)
{
  Specs specs;
  std::string version;

  if (!setting.exists("version") || (setting.lookupValue("version", version) && (version != "0.2" && version != "0.3"))) {
    // format used in the ISPASS paper
    // std::cout << "ParseSpecs" << std::endl;
    auto arithmetic = setting.lookup("arithmetic");
    auto topology = setting.lookup("storage");
    specs.topology = Topology::ParseSpecs(topology, arithmetic, is_sparse_topology);
  } else {
    // format used in Accelergy v0.2/v0.3
    // std::cout << "ParseTreeSpecs" << std::endl;
    specs.topology = Topology::ParseTreeSpecs(setting, is_sparse_topology);
  }

  return specs;
}

void Engine::Spec(Engine::Specs specs)
{
  specs_ = specs;
  topology_.Spec(specs.topology);
  is_specced_ = true;
}

const Topology& Engine::GetTopology() const
{
  return topology_;
}

std::vector<EvalStatus> Engine::PreEvaluationCheck(const Mapping& mapping, problem::Workload& workload, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure)
{
  nest_analysis_.Init(&workload, &mapping.loop_nest, mapping.fanoutX_map, mapping.fanoutY_map);
  return topology_.PreEvaluationCheck(mapping, &nest_analysis_, sparse_optimizations, break_on_failure);
}

std::vector<EvalStatus> Engine::Evaluate(Mapping& mapping, problem::Workload& workload, sparse::SparseOptimizationInfo* sparse_optimizations, bool break_on_failure)
{
  nest_analysis_.Init(&workload, &mapping.loop_nest, mapping.fanoutX_map, mapping.fanoutY_map);
    
  auto eval_status = topology_.Evaluate(mapping, &nest_analysis_, sparse_optimizations, break_on_failure);

  is_evaluated_ = std::accumulate(eval_status.begin(), eval_status.end(), true,
                                  [](bool cur, const EvalStatus& status)
                                  { return cur && status.success; });

  return eval_status;
}
  
double Engine::Energy() const
{
  return topology_.Energy();
}

double Engine::Area() const
{
  return topology_.Area();
}

std::uint64_t Engine::Cycles() const
{
  return topology_.Cycles();
}

double Engine::Utilization() const
{
  return topology_.Utilization();
}

std::ostream& operator << (std::ostream& out, Engine& engine)
{
  out << engine.topology_;
  return out;
}

} // namespace model
