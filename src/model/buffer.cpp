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
#include <numeric>
#include <string>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/buffer.hpp"
//BOOST_CLASS_EXPORT(model::BufferLevel::Specs)
BOOST_CLASS_EXPORT(model::BufferLevel)

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "pat/pat.hpp"

namespace model
{

// ==================================== //
//             Buffer Level             //
// ==================================== //

BufferLevel::BufferLevel()
{ }

BufferLevel::BufferLevel(const Specs& specs) :
    specs_(specs)
{
  is_evaluated_ = false;
}

BufferLevel::~BufferLevel()
{ }

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.
// FIXME: re-factor level-specific code to Buffer class.
BufferLevel::Specs BufferLevel::ParseSpecs(config::CompoundConfigNode level, uint32_t n_elements)
{
  auto& buffer = level;

  Specs specs;

  // Name. This has to go first. Since the rest can be attributes
  std::string name;
  if (buffer.lookupValue("name", name))
  {
    specs.name = config::parseName(name);
  }

  std::string className = "";
  if (buffer.exists("attributes"))
  {
    buffer.lookupValue("class", className);
    buffer = buffer.lookup("attributes");
  }

  // Word Bits.
  std::uint32_t word_bits;
  if (buffer.lookupValue("word-bits", word_bits) ||
      buffer.lookupValue("word_width", word_bits) ||
      buffer.lookupValue("datawidth", word_bits) )
  {
    specs.word_bits = word_bits;
  }
  else
  {
    specs.word_bits = Specs::kDefaultWordBits;
  }

  // Block size.
  std::uint32_t block_size;
  specs.block_size = 1;
  if (buffer.lookupValue("block-size", block_size) ||
      buffer.lookupValue("n_words", block_size) )
  {
    specs.block_size = block_size;
  }

  // Cluster size.
  std::uint32_t cluster_size;
  specs.cluster_size = 1;
  std::uint32_t width;
  if (buffer.lookupValue("cluster-size", cluster_size))
  {
    specs.cluster_size = cluster_size;
  }
  else if (buffer.lookupValue("width", width))
  {
    word_bits = specs.word_bits.Get();
    block_size = specs.block_size.Get();
    assert(width % (word_bits * block_size)  == 0);
    specs.cluster_size = width / (word_bits * block_size);
  }

  // Size.
  // It has dependency on BlockSize and thus is initialized after BlockSize.
  std::uint32_t size;
  if (buffer.lookupValue("entries", size) )
  {
    assert(buffer.exists("sizeKB") == false);
    specs.size = size;
  }
  else if (buffer.lookupValue("depth", size) ||
           buffer.lookupValue("memory_depth", size))
  {
    assert(buffer.exists("sizeKB") == false);
    assert(buffer.exists("entries") == false);
    specs.size = size * specs.block_size.Get();
  }
  else if (buffer.lookupValue("sizeKB", size))
  {
    specs.size = size * 1024 * 8 / specs.word_bits.Get();
  }


  // Technology.
  // Unfortunately ".technology" means different things between ISPASS format
  // and Accelergy v0.2 format. So we use the class name to find out what to
  // assume.
  std::string technology;
  specs.technology = Technology::SRAM;
  if (className == "DRAM") specs.technology = Technology::DRAM;

  if (buffer.lookupValue("technology", technology) && technology == "DRAM")
  {
    specs.technology = Technology::DRAM;
  }

  // SRAM Type.
  std::uint32_t num_ports = 2;
  specs.num_ports = num_ports;
  if (buffer.lookupValue("num-ports", num_ports))
  {
    if (num_ports == 1)
    {
      specs.num_ports = num_ports;
    }
    else
    {
      assert(num_ports == 2);
    }
  }

  // Number of Banks.
  std::uint32_t num_banks = 2;
  specs.num_banks = num_banks;
  if (buffer.lookupValue("num-banks", num_banks))
  {
    specs.num_banks = num_banks;
  }

  // Bandwidth.
  double bandwidth;
  if (buffer.lookupValue("bandwidth", bandwidth))
  {
    std::cerr << "WARNING: bandwidth is deprecated. Assuming read_bandwidth = write_bandwidth = bandwidth/2" << std::endl;
    specs.read_bandwidth  = bandwidth / 2;
    specs.write_bandwidth = bandwidth / 2;
  }

  double read_bandwidth;
  if (buffer.lookupValue("read_bandwidth", read_bandwidth))
  {
    specs.read_bandwidth = read_bandwidth;
  }

  double write_bandwidth;
  if (buffer.lookupValue("write_bandwidth", write_bandwidth))
  {
    specs.write_bandwidth = write_bandwidth;
  }

  // Multiple-buffering factor (e.g., 2.0 means double buffering)
  double multiple_buffering;
  if (buffer.lookupValue("multiple-buffering", multiple_buffering))
  {
    specs.multiple_buffering = multiple_buffering;
  }
  else
  {
    specs.multiple_buffering = 1.0;
  }
  
  if (specs.size.IsSpecified())
  {
    specs.effective_size = static_cast<uint64_t>(std::floor(
            specs.size.Get() / specs.multiple_buffering.Get()));
  }

  // Minimum utilization factor (e.g., 1.0 requires full utilization of effective capacity)
  double min_utilizaiton;
  if (buffer.lookupValue("min-utilization", min_utilizaiton))
  {
    specs.min_utilization = min_utilizaiton;
  }
  else
  {
    specs.min_utilization = 0.0;
  }
  if (specs.min_utilization.Get() != 0.0)
  {
    assert(specs.effective_size.IsSpecified());
  }

  // Instances.
  std::uint32_t instances;
  if (buffer.lookupValue("instances", instances))
  {
    specs.instances = instances;
  } else {
    specs.instances = n_elements;
  }

  // MeshX.
  std::uint32_t meshX;
  if (buffer.lookupValue("meshX", meshX))
  {
    specs.meshX = meshX;
  }

  // MeshY.
  std::uint32_t meshY;
  if (buffer.lookupValue("meshY", meshY))
  {
    specs.meshY = meshY;
  }

  // Vector Access Energy
  double tmp_access_energy = 0;
  double tmp_storage_area = 0;

  if (specs.technology.Get() == Technology::DRAM)
  {
    assert(specs.cluster_size.Get() == 1);
    tmp_access_energy = pat::DRAMEnergy(specs.word_bits.Get() * specs.block_size.Get());
    tmp_storage_area = 0;
  }
  else if (specs.size.Get() == 0)
  {
    //SRAM
    tmp_access_energy = 0;
    tmp_storage_area = 0;
  }
  else
  {
    std::uint64_t tmp_entries = specs.size.Get();
    std::uint64_t tmp_word_bits = specs.word_bits.Get();
    std::uint64_t tmp_block_size = specs.block_size.Get();
    std::uint64_t tmp_cluster_size = specs.cluster_size.Get();
    std::uint64_t width = tmp_word_bits * tmp_block_size * tmp_cluster_size;
    std::uint64_t height =
      (tmp_entries % tmp_block_size == 0) ?
      (tmp_entries / tmp_block_size)      :
      (tmp_entries / tmp_block_size) + 1;  
    tmp_access_energy = pat::SRAMEnergy(height, width, specs.num_banks.Get(), specs.num_ports.Get()) / tmp_cluster_size;
    tmp_storage_area = pat::SRAMArea(height, width, specs.num_banks.Get(), specs.num_ports.Get()) / tmp_cluster_size;
    // std::cout << "Entries = " << tmp_entries
    //           << ", word_size = " << tmp_word_bits
    //           << ", block_size = " << tmp_block_size
    //           << ", cluster_size = " << tmp_cluster_size
    //           << ", num_banks = " << specs.num_banks.Get()
    //           << ", num_ports = " << specs.num_ports.Get()
    //           << ", energy = " << tmp_access_energy
    //           << ", area = " << tmp_storage_area << std::endl;
  }

  // Allow user to override the access energy.
  buffer.lookupValue("vector-access-energy", tmp_access_energy);

  // Allow user to override the cluster area.
  double tmp_cluster_area = 0;
  buffer.lookupValue("cluster-area", tmp_cluster_area);
  if (tmp_cluster_area > 0)
    tmp_storage_area = tmp_cluster_area / specs.cluster_size.Get();

  // Set final area and energy.
  specs.vector_access_energy = tmp_access_energy;
  specs.storage_area = tmp_storage_area; //FIXME: check with Angshu

  // std::cout << "BUFFER " << specs.name << " vector access energy = "
  //           << specs.vector_access_energy << " pJ, cluster area = "
  //           << specs.storage_area.Get() * specs.cluster_size.Get()
  //           << " um^2" << std::endl;

  specs.level_name = specs.name.Get();

  ValidateTopology(specs);
    
  return specs;
}

// Make sure the topology is consistent,
// and update unspecified parameters if they can
// be inferred from other specified parameters.
void BufferLevel::ValidateTopology(BufferLevel::Specs& specs)
{
  bool error = false;
  if (specs.instances.IsSpecified())
  {
    if (specs.meshX.IsSpecified())
    {
      if (specs.meshY.IsSpecified())
      {
        // All 3 are specified.
        assert(specs.meshX.Get() * specs.meshY.Get() == specs.instances.Get());
      }
      else
      {
        // Instances and MeshX are specified.
        assert(specs.instances.Get() % specs.meshX.Get() == 0);
        specs.meshY = specs.instances.Get() / specs.meshX.Get();
      }
    }
    else if (specs.meshY.IsSpecified())
    {
      // Instances and MeshY are specified.
      assert(specs.instances.Get() % specs.meshY.Get() == 0);
      specs.meshX = specs.instances.Get() / specs.meshY.Get();
    }
    else
    {
      // Only Instances is specified.
      specs.meshX = specs.instances.Get();
      specs.meshY = 1;
    }
  }
  else if (specs.meshX.IsSpecified())
  {
    if (specs.meshY.IsSpecified())
    {
      // MeshX and MeshY are specified.
      specs.instances = specs.meshX.Get() * specs.meshY.Get();
    }
    else
    {
      // Only MeshX is specified. We can make assumptions but it's too dangerous.
      error = true;
    }
  }
  else if (specs.meshY.IsSpecified())
  {
    // Only MeshY is specified. We can make assumptions but it's too dangerous.
    error = true;
  }
  else
  {
    // Nothing is specified.
    error = true;
  }

  if (error)
  {
    std::cerr << "ERROR: " << specs.name.Get()
              << ": instances and/or meshX * meshY must be specified."
              << std::endl;
    exit(1);        
  }
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
EvalStatus BufferLevel::PreEvaluationCheck(
  const problem::PerDataSpace<std::size_t> working_set_sizes,
  const tiling::CompoundMask mask,
  const bool break_on_failure)
{
  (void) break_on_failure;

  bool success = true;
  std::ostringstream fail_reason;
  
  if (specs_.size.IsSpecified())
  {
    // Ugh. If we can do a distributed multicast from this level,
    // then the required size may be smaller. However, that depends
    // on the multicast factor etc. that we don't know at this point.
    // Use a very loose filter and fail this check only if there's
    // no chance that this mapping can fit.
    auto available_capacity = specs_.effective_size.Get();
    if (network_->DistributedMulticastSupported())
    {
      available_capacity *= specs_.instances.Get();
    }

    // Find the total capacity required by all un-masked data types.
    std::size_t required_capacity = 0;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      if (mask[pvi])
      {
        required_capacity += working_set_sizes.at(problem::Shape::DataSpaceID(pvi));
      }
    }

    if (required_capacity > available_capacity)
    {
      success = false;
      fail_reason << "mapped tile size " << required_capacity << " exceeds buffer capacity "
                  << available_capacity;
    }
    else if (required_capacity < specs_.effective_size.Get()
             * specs_.min_utilization.Get())
    {
      success = false;
      fail_reason << "mapped tile size " << required_capacity << " is less than constrained "
                  << "minimum utilization " << specs_.effective_size.Get() * specs_.min_utilization.Get();
    }
  }

  EvalStatus eval_status;
  eval_status.success = success;
  eval_status.fail_reason = fail_reason.str();

  return eval_status;  
}

//
// Heavyweight Evaluate() function.
// FIXME: Derive FanoutX, FanoutY, MeshX, MeshY from mapping if unspecified.
//
EvalStatus BufferLevel::Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                                 const std::uint64_t compute_cycles,
                                 const bool break_on_failure)
{
  auto eval_status = ComputeAccesses(tile, mask, break_on_failure);
  if (!break_on_failure || eval_status.success)
  {
    ComputeArea();
    ComputeBufferEnergy();
    ComputeReductionEnergy();
    ComputeAddrGenEnergy();
    ComputePerformance(compute_cycles);
  }
  return eval_status;
}

bool BufferLevel::HardwareReductionSupported()
{
  // FIXME: take this information from an explicit arch spec.
  return !(specs_.technology.IsSpecified() &&
           specs_.technology.Get() == Technology::DRAM);
}

void BufferLevel::Connect(std::shared_ptr<Network> network)
{
  network_ = network;
}

EvalStatus BufferLevel::ComputeAccesses(const tiling::CompoundTile& tile,
                                        const tiling::CompoundMask& mask,
                                        const bool break_on_failure)
{
  (void) break_on_failure;

  bool success = true;
  std::ostringstream fail_reason;
  
  // Subnest FSM should be same for each problem::Shape::DataSpaceID in the list,
  // so just copy it from datatype #0.
  subnest_ = tile[0].subnest;

  //
  // 1. Collect stats (stats are always collected per-DataSpaceID).
  //
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    stats_.keep[pv] = mask[pv];
    
    stats_.partition_size[pv] = tile[pvi].partition_size;
    stats_.utilized_capacity[pv] = tile[pvi].size;
    stats_.utilized_instances[pv] = tile[pvi].replication_factor;

    assert((tile[pvi].size == 0) == (tile[pvi].content_accesses == 0));

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // First epoch is an Update, all subsequent epochs are Read-Modify-Update.
      assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

      stats_.reads[pv] = tile[pvi].content_accesses - tile[pvi].partition_size;
      stats_.updates[pv] = tile[pvi].content_accesses;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar

      // FIXME: temporal reduction and network costs if hardware reduction isn't
      // supported appears to be wonky - network costs may need to trickle down
      // all the way to the level that has the reduction hardware.
      stats_.temporal_reductions[pv] = tile[pvi].content_accesses - tile[pvi].partition_size;
    }
    else // Read-only data type.
    {
      stats_.reads[pv] = tile[pvi].content_accesses;
      stats_.updates[pv] = 0;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.reads[pv] + stats_.fills[pv]; // scalar
      stats_.temporal_reductions[pv] = 0;
    }
  }

  //
  // 2. Derive/validate architecture specs based on stats.
  //
  if (!specs_.technology.IsSpecified())
    specs_.technology = Technology::SRAM;
      
  if (!specs_.num_ports.IsSpecified()) {
    specs_.num_ports = 2;
  }
      
  if (!specs_.num_banks.IsSpecified()) {
    specs_.num_banks = 2; //FIXME: default 2 banks
  }
      
  auto total_utilized_capacity = std::accumulate(stats_.utilized_capacity.begin(),
                                                 stats_.utilized_capacity.end(),
                                                 0ULL);
  if (!specs_.size.IsSpecified())
    specs_.size = std::ceil(total_utilized_capacity
                            * specs_.multiple_buffering.Get());
  else if (total_utilized_capacity > specs_.effective_size.Get())
  {
    success = false;
    fail_reason << "mapped tile size " << total_utilized_capacity << " exceeds buffer capacity "
                << specs_.effective_size.Get();
  }
  else if (total_utilized_capacity < specs_.effective_size.Get()
           * specs_.min_utilization.Get())
  {
    success = false;
    fail_reason << "mapped tile size " << total_utilized_capacity << " is less than constrained "
                << "minimum utilization " << specs_.effective_size.Get() * specs_.min_utilization.Get();
  }

  assert (specs_.block_size.IsSpecified());
    
  assert (specs_.cluster_size.IsSpecified());
      
  specs_.addr_gen_bits = static_cast<unsigned long>(
    std::ceil(std::log2(
                static_cast<double>(specs_.size.Get()) / specs_.block_size.Get()
                )));
      
  if (!specs_.instances.IsSpecified())
    specs_.instances = stats_.utilized_instances.Max();
  else if (stats_.utilized_instances.Max() > specs_.instances.Get())
  {
    success = false;
    fail_reason << "mapped instances " << stats_.utilized_instances.Max() << " exceeds available hardware instances "
                << specs_.instances.Get();
  }

  // Bandwidth constraints cannot be checked/inherited at this point
  // because the calculation is a little more involved. We will do
  // this later in the ComputePerformance() function.      

  // Compute utilized clusters.
  // FIXME: should derive this from precise spatial mapping.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // The following equation assumes fully condensed mapping. Do a ceil-div.
    // stats_.utilized_clusters[pv] = 1 + (stats_.utilized_instances[pv] - 1) /
    //    specs_.cluster_size.Get();
    // Assume utilized instances are sprinkled uniformly across all clusters.
    auto num_clusters = specs_.instances.Get() / specs_.cluster_size.Get();
    stats_.utilized_clusters[pv] = std::min(stats_.utilized_instances[pv],
                                            num_clusters);
  }

  is_evaluated_ = success;

  EvalStatus eval_status;
  eval_status.success = success;
  eval_status.fail_reason = fail_reason.str();
    
  return eval_status;
}

void BufferLevel::ComputeArea()
{
  // YUCK. FIXME. The area is now already stored in a specs_ attribute.
  // The stats_ here are just a copy of the specs_. Do we really need both?

  // Store the total area in the Num/All slot. We used to split it up between the
  // various data-types depending on their contribution towards utilization, but
  // this gives incorrect area if the structure is underutilized, especially when
  // the level is completely bypassed.
  stats_.area = specs_.storage_area.Get();
}

// Compute buffer energy.
void BufferLevel::ComputeBufferEnergy()
{
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto instance_accesses = stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv);

    auto block_size = specs_.block_size.Get();
    double vector_accesses =
      (instance_accesses % block_size == 0) ?
      (instance_accesses / block_size)      :
      (instance_accesses / block_size) + 1;
    
    double cluster_access_energy = vector_accesses *
      specs_.vector_access_energy.Get();

    // Spread out the cost between the utilized instances in each cluster.
    // This is because all the later stat-processing is per-instance.
    if (stats_.utilized_instances.at(pv) > 0)
    {
      double cluster_utilization = double(stats_.utilized_instances.at(pv)) /
        double(stats_.utilized_clusters.at(pv));
      stats_.energy[pv] = cluster_access_energy / cluster_utilization;
      stats_.energy_per_access[pv] = stats_.energy.at(pv) / instance_accesses;
    }
    else
    {
      stats_.energy[pv] = 0;
      stats_.energy_per_access[pv] = 0;
    }
  }
}

//
// Compute reduction energy.
//
void BufferLevel::ComputeReductionEnergy()
{
  // Temporal reduction: add a value coming in on the network to a value stored locally.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.temporal_reduction_energy[pv] = stats_.temporal_reductions[pv] * 
        pat::AdderEnergy(specs_.word_bits.Get(), network_->GetSpecs().word_bits.Get());
    }
    else
    {
      stats_.temporal_reduction_energy[pv] = 0;
    }
  }
}

//
// Compute address generation energy.
//
void BufferLevel::ComputeAddrGenEnergy()
{
  // Note! Address-generation is amortized across the cluster width.
  // We compute the per-cluster energy here. When we sum across instances,
  // we need to be careful to only count each cluster once.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    // We'll use an addr-gen-bits + addr-gen-bits adder, though
    // it's probably cheaper than that. However, we can't assume
    // a 1-bit increment.
    auto pv = problem::Shape::DataSpaceID(pvi);
    stats_.addr_gen_energy[pv] = stats_.address_generations[pv] *
      pat::AdderEnergy(specs_.addr_gen_bits.Get(), specs_.addr_gen_bits.Get());
  }
}

//
// Compute performance.
//
void BufferLevel::ComputePerformance(const std::uint64_t compute_cycles)
{
  // Ugh... have to fix per-datatype word size.
  double word_size = 0;
  word_size = std::max(word_size, double(specs_.word_bits.Get()) / 8);
  
  //
  // Step 1: Compute unconstrained bandwidth demand.
  //
  problem::PerDataSpace<double> unconstrained_read_bandwidth;
  problem::PerDataSpace<double> unconstrained_write_bandwidth;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto total_read_accesses    =   stats_.reads.at(pv);
    auto total_write_accesses   =   stats_.updates.at(pv) + stats_.fills.at(pv);
    unconstrained_read_bandwidth[pv]  = (double(total_read_accesses)  / compute_cycles) * word_size;
    unconstrained_write_bandwidth[pv] = (double(total_write_accesses) / compute_cycles) * word_size;
  }

  //
  // Step 2: Compare vs. specified bandwidth and calculate slowdown.
  //
  stats_.slowdown = 1.0;

  // Find slowdown.
  auto total_unconstrained_read_bandwidth  = std::accumulate(unconstrained_read_bandwidth.begin(),  unconstrained_read_bandwidth.end(),  0.0);
  auto total_unconstrained_write_bandwidth = std::accumulate(unconstrained_write_bandwidth.begin(), unconstrained_write_bandwidth.end(), 0.0);

  if (specs_.read_bandwidth.IsSpecified() &&
      specs_.read_bandwidth.Get() < total_unconstrained_read_bandwidth)
  {
    stats_.slowdown =
      std::min(stats_.slowdown,
               specs_.read_bandwidth.Get() / total_unconstrained_read_bandwidth);
  }
  if (specs_.write_bandwidth.IsSpecified() &&
      specs_.write_bandwidth.Get() < total_unconstrained_write_bandwidth)
  {
    stats_.slowdown =
      std::min(stats_.slowdown,
               specs_.write_bandwidth.Get() / total_unconstrained_write_bandwidth);
  }

  //
  // Step 3:
  // Calculate real bandwidths based on worst slowdown. For shared buffers this
  // ends up effectively slowing down each datatype's bandwidth by the slowdown
  // amount, which is slightly weird but appears to be harmless.
  //
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    stats_.read_bandwidth[pv]  = stats_.slowdown * unconstrained_read_bandwidth.at(pv);
    stats_.write_bandwidth[pv] = stats_.slowdown * unconstrained_write_bandwidth.at(pv);
  }

  //
  // Step 4: Calculate execution cycles.
  //
  stats_.cycles = std::uint64_t(ceil(compute_cycles / stats_.slowdown));

  //
  // Step 5: Update arch specs.
  //
  if (!specs_.read_bandwidth.IsSpecified())
    specs_.read_bandwidth = std::accumulate(stats_.read_bandwidth.begin(), stats_.read_bandwidth.end(), 0.0);
  if (!specs_.write_bandwidth.IsSpecified())
    specs_.write_bandwidth = std::accumulate(stats_.write_bandwidth.begin(), stats_.write_bandwidth.end(), 0.0);
}

//
// Accessors.
//

STAT_ACCESSOR(double, BufferLevel, StorageEnergy, stats_.energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, TemporalReductionEnergy, stats_.temporal_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, AddrGenEnergy, stats_.addr_gen_energy.at(pv) * stats_.utilized_clusters.at(pv)) // Note!!! clusters, not instances.
STAT_ACCESSOR(double, BufferLevel, Energy,
              StorageEnergy(pv) +
              TemporalReductionEnergy(pv) +
              AddrGenEnergy(pv))

STAT_ACCESSOR(std::uint64_t, BufferLevel, Accesses, stats_.utilized_instances.at(pv) * (stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv)))
STAT_ACCESSOR(std::uint64_t, BufferLevel, UtilizedCapacity, stats_.utilized_capacity.at(pv))
STAT_ACCESSOR(std::uint64_t, BufferLevel, UtilizedInstances, stats_.utilized_instances.at(pv))

std::string BufferLevel::Name() const
{
  return specs_.name.Get();
}

double BufferLevel::Area() const
{
  double area = 0;
  area += specs_.storage_area.Get() * specs_.instances.Get();
  return area;
}

double BufferLevel::AreaPerInstance() const
{
  double area = 0;
  area += stats_.area;
  return area;
}

double BufferLevel::Size() const
{
  // FIXME: this is per-instance. This is inconsistent with the naming
  // convention of some of the other methods, which are summed across instances.
  double size = 0;
  size += specs_.size.Get();
  return size;
}

double BufferLevel::CapacityUtilization() const
{
  double utilized_capacity = 0;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    utilized_capacity += stats_.utilized_capacity.at(pv) *
      stats_.utilized_instances.at(pv);
  }

  double total_capacity = Size() * specs_.instances.Get();

  return utilized_capacity / total_capacity;
}

std::uint64_t BufferLevel::Cycles() const
{
  return stats_.cycles;
}

// ---------------
//    Printers
// ---------------

std::ostream& operator<<(std::ostream& out, const BufferLevel::Technology& tech)
{
  switch (tech)
  {
    case BufferLevel::Technology::SRAM: out << "SRAM"; break;
    case BufferLevel::Technology::DRAM: out << "DRAM"; break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const BufferLevel& buffer_level)
{
  buffer_level.Print(out);
  return out;
}

void BufferLevel::Print(std::ostream& out) const
{
  std::string indent = "    ";

  auto& specs = specs_;
  auto& stats = stats_;

  // Print level name.
  out << "=== " << specs.level_name << " ===" << std::endl;  
  out << std::endl;

  // Print specs.
  out << indent << "SPECS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << indent << "Technology           : " << specs.technology << std::endl;
  out << indent << indent << "Size                 : " << specs.size << std::endl;
  out << indent << indent << "Word bits            : " << specs.word_bits << std::endl;    
  out << indent << indent << "Block size           : " << specs.block_size << std::endl;
  out << indent << indent << "Cluster size         : " << specs.cluster_size << std::endl;
  out << indent << indent << "Instances            : " << specs.instances << " ("
      << specs.meshX << "*" << specs.meshY << ")" << std::endl;
  out << indent << indent << "Read bandwidth       : " << specs.read_bandwidth << std::endl;    
  out << indent << indent << "Write bandwidth      : " << specs.write_bandwidth << std::endl;    
  out << indent << indent << "Multiple buffering   : " << specs.multiple_buffering << std::endl;
  out << indent << indent << "Effective size       : " << specs.effective_size << std::endl;
  out << indent << indent << "Min utilization      : " << specs.min_utilization << std::endl;
  out << indent << indent << "Vector access energy : " << specs.vector_access_energy << " pJ" << std::endl;
  out << indent << indent << "Area                 : " << specs.storage_area << " um^2" << std::endl;

  out << std::endl;

  // If the buffer hasn't been evaluated on a specific mapping yet, return.
  if (!IsEvaluated())
  {
    return;
  }

  // Print mapping.
  out << indent << "MAPPING" << std::endl;
  out << indent << "-------" << std::endl;
  out << indent << "Loop nest:" << std::endl;
  std::string loopindent = "  ";
  for (auto loop = subnest_.rbegin(); loop != subnest_.rend(); loop++)
  {
    // Do not print loop if it's a trivial factor.
    if ((loop->start + loop->stride) < loop->end)
    {
      out << indent << loopindent << *loop << std::endl;
      loopindent += "  ";
    }
  }
  out << std::endl;

  // Print stats.
  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << "Cycles               : " << stats.cycles << std::endl;
  out << indent << "Bandwidth throttling : " << stats.slowdown << std::endl;
  
  // Print per-DataSpaceID stats.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    if (stats.keep.at(pv))
    {
      out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

      out << indent + indent << "Partition size                           : " << stats.partition_size.at(pv) << std::endl;
      out << indent + indent << "Utilized capacity                        : " << stats.utilized_capacity.at(pv) << std::endl;
      out << indent + indent << "Utilized instances (max)                 : " << stats.utilized_instances.at(pv) << std::endl;
      out << indent + indent << "Utilized clusters (max)                  : " << stats.utilized_clusters.at(pv) << std::endl;
      out << indent + indent << "Scalar reads (per-instance)              : " << stats.reads.at(pv) << std::endl;
      out << indent + indent << "Scalar updates (per-instance)            : " << stats.updates.at(pv) << std::endl;
      out << indent + indent << "Scalar fills (per-instance)              : " << stats.fills.at(pv) << std::endl;
      out << indent + indent << "Temporal reductions (per-instance)       : " << stats.temporal_reductions.at(pv) << std::endl;
      out << indent + indent << "Address generations (per-cluster)        : " << stats.address_generations.at(pv) << std::endl;
      
      out << indent + indent << "Energy (per-scalar-access)               : " << stats.energy_per_access.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Energy (per-instance)                    : " << stats.energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Energy (total)                           : " << stats.energy.at(pv) * stats.utilized_instances.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Temporal Reduction Energy (per-instance) : "
          << stats.temporal_reduction_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Temporal Reduction Energy (total)        : "
          << stats.temporal_reduction_energy.at(pv) * stats.utilized_instances.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Address Generation Energy (per-cluster)  : "
          << stats.addr_gen_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Address Generation Energy (total)        : "
          << stats.addr_gen_energy.at(pv) * stats.utilized_clusters.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Read Bandwidth (per-instance)            : " << stats.read_bandwidth.at(pv) << " bytes/cycle" << std::endl;
      out << indent + indent << "Read Bandwidth (total)                   : " << stats.read_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " bytes/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (per-instance)           : " << stats.write_bandwidth.at(pv) << " bytes/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (total)                  : " << stats.write_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " bytes/cycle" << std::endl;
    }
  }

  // Print area.
  out << indent + indent << "Area (per-instance)                      : " << stats.area << " um2" << std::endl;
  
  out << std::endl;
}

}  // namespace model
