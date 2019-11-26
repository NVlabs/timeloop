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

#include <cassert>
#include <string>
#include <stdexcept>

#include "model/topology.hpp"
#include "model/network-mesh.hpp"

namespace model
{

std::ostream& operator<<(std::ostream& out, const Topology& topology)
{
  // Save ios format state.
  std::ios state(NULL);
  state.copyfmt(out);
  out << std::fixed << std::setprecision(2);

  //
  // Detailed specs and stats.
  //

  out << "Buffer and Arithmetic Levels" << std::endl;
  out << "----------------------------" << std::endl;

  int level_id = 0;
  for (auto & level : topology.levels_)
  {
    out << "Level " << level_id << std::endl;
    out << "-------" << std::endl;
    out << *level;
    level_id++;
  }

  out << "Networks" << std::endl;
  out << "--------" << std::endl;

  int network_id = 0;
  for (auto & network : topology.networks_)
  {
    out << "Network " << network_id << std::endl;
    out << "---------" << std::endl;
    out << *network;
    network_id++;
  }  

  if (topology.is_evaluated_)
  {
    out << "Total topology energy: " << topology.Energy() << " pJ" << std::endl;
    out << "Total topology area: " << topology.Area() << " um^2" << std::endl;
    out << "Max topology cycles: " << topology.Cycles() << std::endl;
  }

  out << std::endl;

  //
  // Summary stats.
  //
  out << "Summary Stats" << std::endl;
  out << "-------------" << std::endl;
    
  if (topology.is_evaluated_)
  {
    out << "Utilization: " << topology.Utilization() << std::endl;
    out << "Cycles: " << topology.Cycles() << std::endl;
    out << "Energy: " << topology.Energy() / 1000000 << " uJ" << std::endl;
  }
  out << "Area: " << topology.Area() / 1000000 << " mm^2" << std::endl;
  out << std::endl;

  if (topology.is_evaluated_)
  {
    auto num_maccs = topology.MACCs();
    out << "MACCs = " << num_maccs << std::endl;
    out << "pJ/MACC" << std::endl;

    std::size_t max_name_length = 0;
    for (unsigned i = 0; i < topology.NumLevels(); i++)
    {
      max_name_length = std::max(max_name_length, topology.GetLevel(i)->Name().length());
    }
    for (unsigned i = 0; i < topology.NumNetworks(); i++)
    {
      max_name_length = std::max(max_name_length, topology.GetNetwork(i)->Name().length());
    }

    std::string indent = "    ";
    int align = max_name_length + 1;

    for (unsigned i = 0; i < topology.NumLevels(); i++)
    {
      auto level = topology.GetLevel(i);
      out << indent << std::setw(align) << std::left << level->Name() << "= "
          << level->Energy() / num_maccs << std::endl;
    }

    for (unsigned i = 0; i < topology.NumNetworks(); i++)
    {
      auto network = topology.GetNetwork(i);
      out << indent << std::setw(align) << std::left << network->Name() << "= "
          << network->Energy() / num_maccs << std::endl;
    }

    out << indent << std::setw(align) << std::left << "Total" << "= "
        << topology.Energy() / num_maccs << std::endl;
  }

  // Restore ios format state.
  out.copyfmt(state);

  return out;
}

void Topology::Spec(const Topology::Specs& specs)
{
  specs_ = specs;

  for (auto& level : levels_)
  {
    level.reset();
  }
  levels_.clear();

  for (auto& network : networks_)
  {
    network.reset();
  }
  networks_.clear();

  // Construct and spec Buffer and Arithmeic levels.
  for (unsigned i = 0; i < specs.NumLevels(); i++)
  {
    auto level_specs = specs.GetLevel(i);
      
    // What type of level is this?
    if (level_specs->Type() == "BufferLevel")
    {
      BufferLevel::Specs& specs = *std::static_pointer_cast<BufferLevel::Specs>(level_specs);
      std::shared_ptr<BufferLevel> buffer_level = std::make_shared<BufferLevel>(specs);
      std::shared_ptr<Level> level = std::static_pointer_cast<Level>(buffer_level);
      levels_.push_back(level);
    }
    else if (level_specs->Type() == "ArithmeticUnits")
    {
      ArithmeticUnits::Specs& specs = *std::static_pointer_cast<ArithmeticUnits::Specs>(level_specs);
      std::shared_ptr<ArithmeticUnits> arithmetic_level = std::make_shared<ArithmeticUnits>(specs);
      std::shared_ptr<Level> level = std::static_pointer_cast<Level>(arithmetic_level);
      levels_.push_back(level);
    }
    else
    {
      std::cerr << "ERROR: illegal level specs type: " << level_specs->Type() << std::endl;
      exit(1);
    }
  }

  // Construct and spec networks.
  for (unsigned i = 0; i < specs.NumNetworks(); i++)
  {
    auto network_specs = specs.GetNetwork(i);

    MeshNetwork::Specs& mesh_specs = *std::static_pointer_cast<MeshNetwork::Specs>(network_specs);
    std::shared_ptr<MeshNetwork> mesh_network = std::make_shared<MeshNetwork>(mesh_specs);
    std::shared_ptr<Network> network = std::static_pointer_cast<Network>(mesh_network);
    networks_.push_back(network);
  }

  // Connect levels to network.
  assert(specs.NumLevels() == specs.NumNetworks() + 1);
  for (unsigned i = 0; i < specs.NumNetworks(); i++)
  {
    // Note! We are linking levels[i+1] as the outer level for networks[i].
    auto inner = levels_.at(i);
    auto outer = levels_.at(i+1);
    auto network = networks_.at(i);

    if (i > 0)
    {
      auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
      inner_buffer->ConnectFill(network);
      inner_buffer->ConnectDrain(network);
    }
    else
    {
      // auto arithmetic = std::static_pointer_cast<ArithmeticLevel>(inner);
      // arithmetic->ConnectOperand(network);
      // arithmetic->ConnectResult(network);
    }

    auto outer_buffer = std::static_pointer_cast<BufferLevel>(outer);

    //
    // FIXME: network source and sink need to be *bound* per-dataspace at eval (mapping) time.
    //

    outer_buffer->ConnectRead(network);
    network->ConnectSink(inner);
    network->ConnectSource(outer);

    outer_buffer->ConnectUpdate(network);
    // network->ConnectSource(inner);
    // network->ConnectSink(outer);

    std::string network_name = outer->Name() + " <==> " + inner->Name();
    network->SetName(network_name);
  }

  is_specced_ = true;
}

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.

// This function implements the "classic" hierarchical topology
// with arithmetic units at level 0 and storage units at level 1+.
Topology::Specs Topology::ParseSpecs(config::CompoundConfigNode storage,
                                     config::CompoundConfigNode arithmetic)
{
  Specs specs;
  
  assert(storage.isList());

  // Level 0: arithmetic.
  // Use multiplication factor == 0 to ensure .instances attribute is set
  auto level_specs_p = std::make_shared<ArithmeticUnits::Specs>(ArithmeticUnits::ParseSpecs(arithmetic, 0));
  specs.AddLevel(0, std::static_pointer_cast<LevelSpecs>(level_specs_p));

  // Storage levels.
  int num_storage_levels = storage.getLength();
  for (int i = 0; i < num_storage_levels; i++)
  {
    auto level_specs_p = std::make_shared<BufferLevel::Specs>(BufferLevel::ParseSpecs(storage[i], 0));
    specs.AddLevel(i, std::static_pointer_cast<LevelSpecs>(level_specs_p));

    // Networks specs are parsed and extracted from the storage config.
    auto network_specs_p = std::make_shared<MeshNetwork::Specs>(MeshNetwork::ParseSpecs(storage[i]));
    specs.AddNetwork(std::static_pointer_cast<NetworkSpecs>(network_specs_p));
  }

  return specs;
}

// This function implements the "tree-like" hierarchical architecture description
// used in Accelergy v0.2. The lowest level is level 0 and should have
// arithmetic units, while other level are level 1+ with some buffer/storage units
Topology::Specs Topology::ParseTreeSpecs(config::CompoundConfigNode designRoot)
{
  Specs specs;
  auto curNode = designRoot;

  std::vector<std::shared_ptr<LevelSpecs>> storages; // serialize all storages
  std::vector<std::shared_ptr<NetworkSpecs>> networks;

  uint32_t multiplication = 1;

  // Walk the tree to find each buffer and arithmetic units
  // and add them to the specs.
  while (curNode.exists("subtree"))
  {
    auto subTrees = curNode.lookup("subtree");
    // Timeloop currently supports one subtree per level.
    assert(subTrees.isList() && subTrees.getLength() == 1);
    curNode = subTrees[0];

    std::string curNodeName;
    curNode.lookupValue("name", curNodeName);

    uint32_t subTreeSize = config::parseElementSize(curNodeName);
    multiplication *= subTreeSize;

    if (curNode.exists("local"))
    {
      auto curLocal = curNode.lookup("local");
      assert(curLocal.isList());

      std::vector<std::shared_ptr<LevelSpecs>> localStorages;
      std::vector<std::shared_ptr<NetworkSpecs>> localNetworks;

      for (int c = 0; c < curLocal.getLength() ; c++)
      {
        std::string cName, cClass;
        curLocal[c].lookupValue("name", cName);
        curLocal[c].lookupValue("class", cClass);
        uint32_t localElementSize = config::parseElementSize(cName);
        uint32_t nElements = multiplication * localElementSize;

        if (isBufferClass(cClass))
        {
          // Create a buffer.
          auto level_specs_p = std::make_shared<BufferLevel::Specs>(BufferLevel::ParseSpecs(curLocal[c], nElements));
          localStorages.push_back(level_specs_p);

          // Create a network.
          auto network_specs_p = std::make_shared<MeshNetwork::Specs>(MeshNetwork::ParseSpecs(curLocal[c]));
          localNetworks.push_back(network_specs_p);
        }
        else if (isComputeClass(cClass))
        {
          // Create arithmetic.
          auto level_specs_p = std::make_shared<ArithmeticUnits::Specs>(ArithmeticUnits::ParseSpecs(curLocal[c], nElements));
          specs.AddLevel(0, std::static_pointer_cast<LevelSpecs>(level_specs_p));
        }
        else
        {
          // std::cout << "  Neglect component: " << cName << " due to unknown class: " << cClass << std::endl;
        }
      }
      // The deeper the tree, the closer the buffer to be with ArithmeticUnits.
      // Reverse the order so that top in the local list is at the bottem, matching the tree seq
      storages.insert(storages.begin(), localStorages.rbegin(), localStorages.rend());
      networks.insert(networks.begin(), localNetworks.rbegin(), localNetworks.rend());
    }
  } // end while

  // Add storages to specs. We can do this only after walking the whole tree.
  for (uint32_t i = 0; i < storages.size(); i++)
  {
    auto storage = storages[i];
    specs.AddLevel(i, storage);

    auto network = networks[i];
    specs.AddNetwork(network);
  }

  return specs;
};

void Topology::Specs::ParseAccelergyERT(config::CompoundConfigNode ert)
{
  // std::cout << "Replacing energy numbers..." << std::endl;
  std::vector<std::string> keys;
  assert(ert.exists("tables"));
  auto table = ert.lookup("tables");
  table.getMapKeys(keys);

  for (auto key : keys) {
    auto componentERT = table.lookup(key);
    auto pos = key.rfind(".");
    auto componentName = key.substr(pos + 1, key.size() - pos - 1);
    // std::cout << componentName << std::endl;

    // update levels by name and the type of it
    if (componentName == "wire" || componentName == "Wire") { // special case, update interal wire model
      float transferEnergy;
      auto actionERT = componentERT.lookup("transfer_random");
      if (actionERT.lookupValue("energy", transferEnergy)) {
        for (unsigned i = 0; i < NumStorageLevels(); i++) { // update wire energy for all storage levels
          auto bufferSpec = GetStorageLevel(i);
          auto networkSpec = GetNetwork(i); // FIXME.
          std::static_pointer_cast<MeshNetwork::Specs>(networkSpec)->wire_energy = transferEnergy; // FIXME.
        }
      }
    } else {
      // Find the level that matches this name and see what type it is
      bool isArithmeticUnit = false;
      bool isBuffer = false;
      std::shared_ptr<LevelSpecs> specToUpdate;
      for (auto level : levels) {
        //std::cout << "  level: " << level->level_name << std::endl;
        if (level->level_name == componentName) {
          specToUpdate = level;
          if (level->Type() == "BufferLevel") isBuffer = true;
          if (level->Type() == "ArithmeticUnits") isArithmeticUnit = true;
        }
      }
      // Find the most expensive action as the unit cost
      std::vector<std::string> actions;
      componentERT.getMapKeys(actions);
      double opEnergy = 0.0;
      double argEnergy = 0.0;
      for (auto action : actions) {
        auto actionERT = componentERT.lookup(action);
        if (actionERT.isList()) { // action support argument
          for (int i = 0; i < actionERT.getLength(); i ++) {
            if (actionERT[i].lookupValue("energy", argEnergy)) {
              opEnergy = std::max(argEnergy, opEnergy);
            }
          }
        } else { // no argument action
          if (actionERT.lookupValue("energy", argEnergy)) {
            opEnergy = std::max(argEnergy, opEnergy);
          }
        }
      }
      // Replace the energy per action
      if (isArithmeticUnit) {
        // std::cout << "  Replace " << componentName << " energy with energy " << opEnergy << std::endl;
        auto arithmeticSpec = GetArithmeticLevel();
        arithmeticSpec->energy_per_op = opEnergy;
      } else if (isBuffer) {
        auto bufferSpec = std::static_pointer_cast<BufferLevel::Specs>(specToUpdate);
        // std::cout << "  Replace " << componentName << " VectorAccess energy with energy " << opEnergy << std::endl;
        bufferSpec->vector_access_energy = opEnergy / bufferSpec->cluster_size.Get();
      } else {
        // std::cout << "  Unused component ERT: "  << key << std::endl;
      }
    }
  }

  return;
}

std::vector<std::string> Topology::Specs::LevelNames() const
{
  std::vector<std::string> level_names;
  for (unsigned level_id = 0; level_id < NumLevels(); level_id++)
  {
    level_names.push_back(GetLevel(level_id)->level_name);
  }
  return level_names;
}

std::vector<std::string> Topology::Specs::StorageLevelNames() const
{
  std::vector<std::string> storage_level_names;
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    storage_level_names.push_back(GetStorageLevel(storage_level_id)->level_name);
  }
  return storage_level_names;
}

//
// Level accessors.
//

void Topology::Specs::AddLevel(unsigned typed_id, std::shared_ptr<LevelSpecs> level_specs)
{
  if (level_specs->Type() == "BufferLevel")
  {
    storage_map[typed_id] = levels.size();
  }
  else if (level_specs->Type() == "ArithmeticUnits")
  {
    assert(typed_id == 0);
    arithmetic_map = levels.size();
  }
  else
  {
    std::cerr << "ERROR: illegal level specs type: " << level_specs->Type() << std::endl;
    exit(1);
  }
  levels.push_back(level_specs);
}

void Topology::Specs::AddNetwork(std::shared_ptr<NetworkSpecs> specs)
{
  networks.push_back(specs);
}

unsigned Topology::Specs::NumLevels() const
{
  return levels.size();
}

unsigned Topology::Specs::NumStorageLevels() const
{
  return storage_map.size();
}

unsigned Topology::Specs::NumNetworks() const
{
  return networks.size();
}

std::shared_ptr<LevelSpecs> Topology::Specs::GetLevel(unsigned level_id) const
{
  return levels.at(level_id);
}

std::shared_ptr<BufferLevel::Specs> Topology::Specs::GetStorageLevel(unsigned storage_level_id) const
{
  auto level_id = storage_map.at(storage_level_id);
  return std::static_pointer_cast<BufferLevel::Specs>(levels.at(level_id));
}

std::shared_ptr<ArithmeticUnits::Specs> Topology::Specs::GetArithmeticLevel() const
{
  auto level_id = arithmetic_map;
  return std::static_pointer_cast<ArithmeticUnits::Specs>(levels.at(level_id));
}

std::shared_ptr<NetworkSpecs> Topology::Specs::GetNetwork(unsigned network_id) const
{
  return networks.at(network_id);
}

//
// Topology class.
//
unsigned Topology::NumLevels() const
{
  assert(is_specced_);
  return levels_.size();
}

unsigned Topology::NumStorageLevels() const
{
  assert(is_specced_);
  return specs_.NumStorageLevels();
}

unsigned Topology::NumNetworks() const
{
  assert(is_specced_);
  return specs_.NumNetworks();
}

std::shared_ptr<Level> Topology::GetLevel(unsigned level_id) const
{
  return levels_.at(level_id);
}

std::shared_ptr<BufferLevel> Topology::GetStorageLevel(unsigned storage_level_id) const
{
  auto level_id = specs_.StorageMap(storage_level_id);
  return std::static_pointer_cast<BufferLevel>(levels_.at(level_id));
}

std::shared_ptr<ArithmeticUnits> Topology::GetArithmeticLevel() const
{
  auto level_id = specs_.ArithmeticMap();
  return std::static_pointer_cast<ArithmeticUnits>(levels_.at(level_id));
}

std::shared_ptr<Network> Topology::GetNetwork(unsigned id) const
{
  return networks_.at(id);
}

// PreEvaluationCheck(): allows for a very fast capacity-check
// based on given working-set sizes that can be trivially derived
// by the caller. The more powerful Evaluate() function also
// performs these checks, but computes both tile sizes and access counts
// and requires full tiling data that is generated by a very slow
// Nest::ComputeWorkingSets() algorithm. The PreEvaluationCheck()
// function is an optional call that extensive design-space searches
// can use to fail early.
// FIXME: integrate with Evaluate() and re-factor.
// FIXME: what about instances and fanout checks?
std::vector<EvalStatus> Topology::PreEvaluationCheck(const Mapping& mapping,
                                                     analysis::NestAnalysis* analysis,
                                                     bool break_on_failure)
{
  auto masks = tiling::TransposeMasks(mapping.datatype_bypass_nest);
  auto working_set_sizes = analysis->GetWorkingSetSizes_LTW();

  std::vector<EvalStatus> eval_status(NumLevels(), { .success = true, .fail_reason = "" });
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    auto level_id = specs_.StorageMap(storage_level_id);
    auto s = GetStorageLevel(storage_level_id)->PreEvaluationCheck(
      working_set_sizes.at(storage_level_id), masks.at(storage_level_id),
      break_on_failure);
    eval_status.at(level_id) = s;
    if (break_on_failure && !s.success)
      break;
  }

  return eval_status;
}

std::vector<EvalStatus> Topology::Evaluate(Mapping& mapping,
                                           analysis::NestAnalysis* analysis,
                                           const problem::Workload& workload,
                                           bool break_on_failure)
{
  assert(is_specced_);

  // ==================================================================
  // TODO: connect buffers to networks based on bypass mask in mapping.
  // ==================================================================
  // for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  // {
  //   auto storage_level = GetStorageLevel(storage_level_id);
  //   auto network = GetNetwork(storage_level_id);

  //   storage_level->ConnectNetwork(network);
  //   network->ConnectBuffer(storage_level);
  // }  

  std::vector<EvalStatus> eval_status(NumLevels(), { .success = true, .fail_reason = "" });
  bool success_accum = true;
  
  // Compute working-set tile hierarchy for the nest.
  problem::PerDataSpace<std::vector<tiling::TileInfo>> ws_tiles;
  try
  {
    ws_tiles = analysis->GetWorkingSets();
  }
  catch (std::runtime_error& e)
  {
    std::fill(eval_status.begin(), eval_status.end(),
              EvalStatus({ .success = false, .fail_reason = "" }));
    return eval_status;
  }

  // Ugh... FIXME.
  auto compute_cycles = analysis->GetBodyInfo().accesses;

  // Create a mask indicating which levels support distributed multicast.
  tiling::CompoundMaskNest distribution_supported;
  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
    distribution_supported[pv].reset();
    for (unsigned storage_level = 0; storage_level < NumStorageLevels(); storage_level++)
    {
      if (GetStorageLevel(storage_level)->GetReadNetwork()->DistributedMulticastSupported())
      {
        distribution_supported[pv].set(storage_level);
      }
    }
  }
  
  // Collapse tiles into a specified number of tiling levels. The solutions are
  // received in a set of per-problem::Shape::DataSpaceID arrays.
  auto collapsed_tiles = tiling::CollapseTiles(ws_tiles, specs_.NumStorageLevels(),
                                               mapping.datatype_bypass_nest,
                                               distribution_supported);

  // Transpose the tiles into level->datatype structure.
  auto tiles = tiling::TransposeTiles(collapsed_tiles);
  assert(tiles.size() == NumStorageLevels());

  // Transpose the datatype bypass nest into level->datatype structure.
  auto keep_masks = tiling::TransposeMasks(mapping.datatype_bypass_nest);
  assert(keep_masks.size() >= NumStorageLevels());
  
  // Area of all the compute + buffer elements in inner levels
  // (needed for wire energy calculation).
  // FIXME: Breaks abstraction by making assumptions about arithmetic
  // (multiplier) organization and querying multiplier area.
  double inner_tile_area = GetArithmeticLevel()->AreaPerInstance();

  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    auto storage_level = GetStorageLevel(storage_level_id);
    
    // Evaluate Loop Nest on hardware structures: calculate
    // primary statistics.
    auto level_id = specs_.StorageMap(storage_level_id);
    auto s = storage_level->Evaluate(tiles[storage_level_id], keep_masks[storage_level_id],
                                     compute_cycles, break_on_failure);
    eval_status.at(level_id) = s;
    success_accum &= s.success;

    if (break_on_failure && !s.success)
      break;

    // Evaluate network.
    auto network = GetNetwork(storage_level_id); // FIXME.
    s = network->Evaluate(tiles[storage_level_id], inner_tile_area, break_on_failure);
    eval_status.at(level_id) = s;
    success_accum &= s.success;

    if (break_on_failure && !s.success)
      break;
    
    // The inner tile area is the area of the local sub-level that I will
    // send data to. Note that it isn't the area of the entire sub-level
    // because I may only have reach into a part of the level, which will
    // reduce my wire energy costs. To determine this, we use the fanout
    // from this level inwards.
    // FIXME: We need a better floorplanner.
    double cur_level_area = storage_level->AreaPerInstance();
    inner_tile_area = cur_level_area + (inner_tile_area * network->MaxFanout());
  }

  if (!break_on_failure || success_accum)
  {
    auto level_id = specs_.ArithmeticMap();
    auto s = GetArithmeticLevel()->HackEvaluate(analysis, workload);
    eval_status.at(level_id) = s;
    success_accum &= s.success;
  }

  if (success_accum)
    is_evaluated_ = true;

  return eval_status;
}

double Topology::Energy() const
{
  double energy = 0;
  for (auto level : levels_)
  {
    assert(level->Energy() >= 0);
    energy += level->Energy();
  }

  // for (unsigned i = 1 /*note*/; i < NumLevels(); i++)
  // {
  //   energy += std::static_pointer_cast<BufferLevel>(GetLevel(i))->network_.Energy();    
  // }
  for (unsigned i = 0; i < NumNetworks(); i++)
  {
    auto e = GetNetwork(i)->Energy();
    assert(e >= 0);
    energy += e;
  }

  return energy;
}

double Topology::Area() const
{
  double area = 0;
  for (auto level : levels_)
  {
    assert(level->Area() >= 0);
    area += level->Area();
  }
  return area;
}

std::uint64_t Topology::Cycles() const
{
  std::uint64_t cycles = 0;
  for (auto level : levels_)
  {
    cycles = std::max(cycles, level->Cycles());
  }
  return cycles;
}

double Topology::Utilization() const
{
  // FIXME.
  return (GetArithmeticLevel()->IdealCycles() / Cycles());
}

std::vector<problem::PerDataSpace<std::uint64_t>> Topology::TileSizes() const
{
  std::vector<problem::PerDataSpace<std::uint64_t>> tile_sizes;
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    problem::PerDataSpace<std::uint64_t> uc;
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      uc[pv] = GetStorageLevel(storage_level_id)->UtilizedCapacity(pv);
    }
    tile_sizes.push_back(uc);
  }
  return tile_sizes;
}

std::vector<problem::PerDataSpace<std::uint64_t>> Topology::UtilizedInstances() const
{
  std::vector<problem::PerDataSpace<std::uint64_t>> utilized_instances;
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    problem::PerDataSpace<std::uint64_t> uc;
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      uc[pv] = GetStorageLevel(storage_level_id)->UtilizedInstances(pv);
    }
    utilized_instances.push_back(uc);
  }
  return utilized_instances;
}

std::uint64_t Topology::MACCs() const
{
  return GetArithmeticLevel()->MACCs();
}

std::uint64_t Topology::LastLevelAccesses() const
{
  return GetStorageLevel(NumStorageLevels()-1)->Accesses();
}

bool isBufferClass(std::string className) {
  for (auto s : bufferClasses) {
    if (className.find(s) != std::string::npos) return true;
  }
  return false;
}

bool isComputeClass(std::string className) {
  for (auto s : computeClasses) {
    if (className.find(s) != std::string::npos) return true;
  }
  return false;
}

}  // namespace model
