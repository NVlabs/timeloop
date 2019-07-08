/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

BufferLevel::BufferLevel(const Specs& specs) :
    specs_(specs)
{
  is_evaluated_ = false;
}

void BufferLevel::ParseBufferSpecs(libconfig::Setting& buffer, problem::Shape::DataSpaceID pv, Specs& specs)
{
  // Word Bits.
  std::uint32_t word_bits;
  if (buffer.lookupValue("word-bits", word_bits))
  {
    specs.WordBits(pv) = word_bits;
  }
  else
  {
    specs.WordBits(pv) = Specs::kDefaultWordBits;
  }

  // Size.
  std::uint32_t size;
  if (buffer.lookupValue("entries", size))
  {
    assert(buffer.exists("sizeKB") == false);
    specs.Size(pv) = size;
  }
  else if (buffer.lookupValue("sizeKB", size))
  {
    specs.Size(pv) = size * 1024 * 8 / specs.WordBits(pv).Get();
  }

  // Block size.
  std::uint32_t block_size;
  specs.BlockSize(pv) = 1; // FIXME: derive block size from tile.
  if (buffer.lookupValue("block-size", block_size))
  {
    specs.BlockSize(pv) = block_size;
  }

  // Cluster size.
  std::uint32_t cluster_size;
  specs.ClusterSize(pv) = 1;
  if (buffer.lookupValue("cluster-size", cluster_size))
  {
    specs.ClusterSize(pv) = cluster_size;
  }

  // Name.
  std::string name;
  if (buffer.lookupValue("name", name))
  {
    specs.Name(pv) = name;
  }
      
  // Technology.
  std::string technology;
  specs.Tech(pv) = Technology::SRAM;
  if (buffer.lookupValue("technology", technology))
  {
    if (technology == "DRAM")
    {
      specs.Tech(pv) = Technology::DRAM;
    } else {
      assert(technology == "SRAM");
    }
  }

  // SRAM Type.
  std::uint32_t num_ports = 2;
  specs.NumPorts(pv) = num_ports;
  if (buffer.lookupValue("num-ports", num_ports))
  {
    if (num_ports == 1)
    {
      specs.NumPorts(pv) = num_ports;
    } else {
      assert(num_ports == 2);
    }
  }

  // Number of Banks.
  std::uint32_t num_banks = 2;
  specs.NumBanks(pv) = num_banks;
  if (buffer.lookupValue("num-banks", num_banks))
  {
    specs.NumBanks(pv) = num_banks;
  }

  // Bandwidth.
  // Have to parse both int and double because $#@%$# libconfig
  // won't parse an int as a double.
  double d_bandwidth;
  unsigned i_bandwidth;
  if (buffer.lookupValue("bandwidth", d_bandwidth))
  {
    std::cerr << "WARNING: bandwidth is deprecated. Assuming read_bandwidth = write_bandwidth = bandwidth/2" << std::endl;
    specs.ReadBandwidth(pv)  = d_bandwidth / 2;
    specs.WriteBandwidth(pv) = d_bandwidth / 2;
  }
  else if (buffer.lookupValue("bandwidth", i_bandwidth))
  {
    std::cerr << "WARNING: bandwidth is deprecated. Assuming read_bandwidth = write_bandwidth = bandwidth/2" << std::endl;
    specs.ReadBandwidth(pv) = static_cast<double>(i_bandwidth) / 2;
    specs.WriteBandwidth(pv) = static_cast<double>(i_bandwidth) / 2;
  }

  double d_read_bandwidth;
  unsigned i_read_bandwidth;
  if (buffer.lookupValue("read_bandwidth", d_read_bandwidth))
  {
    specs.ReadBandwidth(pv) = d_read_bandwidth;
  }
  else if (buffer.lookupValue("read_bandwidth", i_read_bandwidth))
  {
    specs.ReadBandwidth(pv) = static_cast<double>(i_read_bandwidth);
  }

  double d_write_bandwidth;
  unsigned i_write_bandwidth;
  if (buffer.lookupValue("write_bandwidth", d_write_bandwidth))
  {
    specs.WriteBandwidth(pv) = d_write_bandwidth;
  }
  else if (buffer.lookupValue("write_bandwidth", i_write_bandwidth))
  {
    specs.WriteBandwidth(pv) = static_cast<double>(i_write_bandwidth);
  }

  // Multiple-buffering factor (e.g., 2.0 means double buffering)
  // Have to parse both int and double because libconfig
  // won't parse an int as a double.
  double d_multiple_buffering;
  unsigned i_multiple_buffering;
  if (buffer.lookupValue("multiple-buffering", d_multiple_buffering))
  {
    specs.MultipleBuffering(pv) = d_multiple_buffering;
  }
  else if (buffer.lookupValue("multiple-buffering", i_multiple_buffering))
  {
    specs.MultipleBuffering(pv) = static_cast<double>(i_multiple_buffering);
  }
  else
  {
    specs.MultipleBuffering(pv) = 1.0;
  }
  
  if (specs.Size(pv).IsSpecified())
  {
    specs.EffectiveSize(pv) = static_cast<uint64_t>(std::floor(
            specs.Size(pv).Get() / specs.MultipleBuffering(pv).Get()));
  }

  // Minimum utilization factor (e.g., 1.0 requires full utilization of effective capacity)
  // Have to parse both int and double because libconfig
  // won't parse an int as a double.
  double d_min_utilizaiton;
  unsigned i_min_utilizaiton;
  if (buffer.lookupValue("min-utilization", d_min_utilizaiton))
  {
    specs.MinUtilization(pv) = d_min_utilizaiton;
  }
  else if (buffer.lookupValue("min-utilization", i_min_utilizaiton))
  {
    specs.MinUtilization(pv) = static_cast<double>(i_min_utilizaiton);
  }
  else
  {
    specs.MinUtilization(pv) = 0.0;
  }
  if (specs.MinUtilization(pv).Get() != 0.0)
  {
    assert(specs.EffectiveSize(pv).IsSpecified());
  }

  // Instances.
  std::uint32_t instances;
  if (buffer.lookupValue("instances", instances))
  {
    specs.Instances(pv) = instances;
  }

  // MeshX.
  std::uint32_t meshX;
  if (buffer.lookupValue("meshX", meshX))
  {
    specs.MeshX(pv) = meshX;
  }

  // MeshY.
  std::uint32_t meshY;
  if (buffer.lookupValue("meshY", meshY))
  {
    specs.MeshY(pv) = meshY;
  }

  // Network Type.
  std::string network_type;
  if (buffer.lookupValue("network-type", network_type))
  {
    if (network_type.compare("1:1") == 0)
      specs.NetworkType(pv) = Specs::Network::Type::OneToOne;
    else if (network_type.compare("1:N") == 0)
      specs.NetworkType(pv) = Specs::Network::Type::OneToMany;
    else if (network_type.compare("M:N") == 0)
      specs.NetworkType(pv) = Specs::Network::Type::ManyToMany;
    else
    {
      std::cerr << "ERROR: Unrecognized network type: " << network_type << std::endl;
      exit(1);
    }
  }
    
  // Network Word Bits.
  std::uint32_t network_word_bits;
  if (buffer.lookupValue("network-word-bits", network_word_bits))
  {
    specs.NetworkWordBits(pv) = network_word_bits;
  }
  else
  {
    specs.NetworkWordBits(pv) = specs.WordBits(pv).Get();
  }

  // Fanout.
  std::uint32_t fanout;
  if (buffer.lookupValue("fanout", fanout))
  {
    std::cerr << "ERROR: Fanout cannot be specified, it must be derived." << std::endl;
    exit(1);
  }

  // Router energy.
  double router_energy = 0;
  buffer.lookupValue("router-energy", router_energy);
  specs.RouterEnergy(pv) = router_energy;

  // Vector Access Energy
  double tmp_access_energy = 0;
  double tmp_storage_area = 0;

  if (specs.Tech(pv).Get() == Technology::DRAM) {
    assert(specs.ClusterSize(pv).Get() == 1);
    tmp_access_energy = pat::DRAMEnergy(specs.NetworkWordBits(pv).Get() * specs.BlockSize(pv).Get());
    tmp_storage_area = 0;
  } else if (specs.Size(pv).Get() == 0) {
    //SRAM
    tmp_access_energy = 0;
    tmp_storage_area = 0;
  } else {
    std::uint64_t tmp_entries = specs.Size(pv).Get();
    std::uint64_t tmp_word_bits = specs.WordBits(pv).Get();
    std::uint64_t tmp_block_size = specs.BlockSize(pv).Get();
    std::uint64_t tmp_cluster_size = specs.ClusterSize(pv).Get();
    std::uint64_t width = tmp_word_bits * tmp_block_size * tmp_cluster_size;
    std::uint64_t height =
      (tmp_entries % tmp_block_size == 0) ?
      (tmp_entries / tmp_block_size)      :
      (tmp_entries / tmp_block_size) + 1;  
    tmp_access_energy = pat::SRAMEnergy(height, width, specs.NumBanks(pv).Get(), specs.NumPorts(pv).Get()) / tmp_cluster_size;
    tmp_storage_area = pat::SRAMArea(height, width, specs.NumBanks(pv).Get(), specs.NumPorts(pv).Get()) / tmp_cluster_size;
    // std::cout << "Entries = " << tmp_entries
    //           << ", word_size = " << tmp_word_bits
    //           << ", block_size = " << tmp_block_size
    //           << ", cluster_size = " << tmp_cluster_size
    //           << ", num_banks = " << specs.NumBanks(pv).Get()
    //           << ", num_ports = " << specs.NumPorts(pv).Get()
    //           << ", energy = " << tmp_access_energy
    //           << ", area = " << tmp_storage_area << std::endl;
  }

  // Allow user to override the access energy. FIXME: clean up this code.
  buffer.lookupValue("vector-access-energy", tmp_access_energy);

  specs.VectorAccessEnergy(pv) = tmp_access_energy;
  specs.StorageArea(pv) = tmp_storage_area; //FIXME: check with Angshu
}

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.
// FIXME: re-factor level-specific code to Buffer class.
BufferLevel::Specs BufferLevel::ParseSpecs(libconfig::Setting& level)
{
  // Legacy code treats partitioned and shared in completely different code paths.
  // Much of that code still exists across this buffer implementation. However,
  // that style of partitioning in hindsight is not general enough - it makes the
  // number of *architectural* partitions be the number of workload datatypes -
  // which is unnatural. Longer term, we want to decouple the concept of architectural
  // buffers that form a logical level from the workload/problem datatypes that are
  // mapped onto it.

  bool partitioned = false;
  if (level.exists("buffers"))
  {
    partitioned = true;
  }
  
  DataSpaceIDSharing sharing = partitioned ? DataSpaceIDSharing::Partitioned : DataSpaceIDSharing::Shared;
 
  // We now know if the buffer is partitioned or shared.
  // Start constructing the specs.
  Specs specs(sharing);
  
  // BufferLevel-name vs. name (pre-parse).
  if (partitioned)
  {
    assert(level.lookupValue("name", specs.level_name));

    auto& buffers = level.lookup("buffers");
    assert(buffers.isList());
    assert(buffers.getLength() == int(problem::GetShape()->NumDataSpaces));

    // Ugly, FIXME: buffer specs are serially assigned to pvis.
    unsigned pvi = 0;
    for (auto& buffer: buffers)
    {
      // Sophia
      ParseBufferSpecs(buffer, problem::Shape::DataSpaceID(pvi), specs);
      pvi++;
    }
  }
  else
  {
    ParseBufferSpecs(level, problem::GetShape()->NumDataSpaces, specs);
    specs.level_name = specs.Name().Get();
  }

  ValidateTopology(specs);
    
  return specs;
}

// Make sure the topology is consistent,
// and update unspecified parameters if they can
// be inferred from other specified parameters.
void BufferLevel::ValidateTopology(BufferLevel::Specs& specs)
{
  for (unsigned pvi = 0; pvi < specs.NumPartitions(); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    
    bool error = false;
    if (specs.Instances(pv).IsSpecified())
    {
      if (specs.MeshX(pv).IsSpecified())
      {
        if (specs.MeshY(pv).IsSpecified())
        {
          // All 3 are specified.
          assert(specs.MeshX(pv).Get() * specs.MeshY(pv).Get() == specs.Instances(pv).Get());
        }
        else
        {
          // Instances and MeshX are specified.
          assert(specs.Instances(pv).Get() % specs.MeshX(pv).Get() == 0);
          specs.MeshY(pv) = specs.Instances(pv).Get() / specs.MeshX(pv).Get();
        }
      }
      else if (specs.MeshY(pv).IsSpecified())
      {
        // Instances and MeshY are specified.
        assert(specs.Instances(pv).Get() % specs.MeshY(pv).Get() == 0);
        specs.MeshX(pv) = specs.Instances(pv).Get() / specs.MeshY(pv).Get();
      }
      else
      {
        // Only Instances is specified.
        specs.MeshX(pv) = specs.Instances(pv).Get();
        specs.MeshY(pv) = 1;
      }
    }
    else if (specs.MeshX(pv).IsSpecified())
    {
      if (specs.MeshY(pv).IsSpecified())
      {
        // MeshX and MeshY are specified.
        specs.Instances(pv) = specs.MeshX(pv).Get() * specs.MeshY(pv).Get();
      }
      else
      {
        // Only MeshX is specified. We can make assumptions but it's too dangerous.
        error = true;
      }
    }
    else if (specs.MeshY(pv).IsSpecified())
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
      std::cerr << "ERROR: " << specs.Name(pv).Get()
                << ": instances and/or meshX * meshY must be specified."
                << std::endl;
      exit(1);        
    }
  }
}

bool BufferLevel::DistributedMulticastSupported()
{
  bool retval = true;

  for (unsigned pvi = 0; pvi < specs_.NumPartitions(); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    retval &= (specs_.NetworkType(pv).IsSpecified() &&
               specs_.NetworkType(pv).Get() == Specs::Network::Type::ManyToMany);
  }

  return retval;
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
bool BufferLevel::PreEvaluationCheck(
    const problem::PerDataSpace<std::size_t> working_set_sizes,
    const tiling::CompoundMask mask,
    const bool break_on_failure)
{
  bool success = true;
  
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);      
      if (specs_.Size(pv).IsSpecified() && mask[pvi])
      {
        // Ugh. If we can do a distributed multicast from this level,
        // then the required size may be smaller. However, that depends
        // on the multicast factor etc. that we don't know at this point.
        // Use a very loose filter and fail this check only if there's
        // no chance that this mapping can fit.
        auto available_capacity = specs_.EffectiveSize(pv).Get();
        if (DistributedMulticastSupported())
        {
          available_capacity *= specs_.Instances(pv).Get();
        }

        if (working_set_sizes.at(pv) > available_capacity)
        {
          success = false;
          if (break_on_failure)
            break;
        }
        else if (working_set_sizes.at(pv) < specs_.EffectiveSize(pv).Get()
                                            * specs_.MinUtilization(pv).Get())
        {
          success = false;
          if (break_on_failure)
            break;
        }
      }
    }
  }
  else  // sharing_type == DataSpaceIDSharing::Shared
  {
    if (specs_.Size().IsSpecified())
    {
      // Ugh. If we can do a distributed multicast from this level,
      // then the required size may be smaller. However, that depends
      // on the multicast factor etc. that we don't know at this point.
      // Use a very loose filter and fail this check only if there's
      // no chance that this mapping can fit.
      auto available_capacity = specs_.EffectiveSize().Get();
      if (DistributedMulticastSupported())
      {
        available_capacity *= specs_.Instances().Get();
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
        // std::cerr << "CAPACITY FAIL " << specs_.level_name << " req = " << required_capacity << " avail = " << available_capacity << std::endl;
        success = false;
      }
      else if (required_capacity < specs_.EffectiveSize().Get()
                                   * specs_.MinUtilization().Get())
      {
        success = false;
      }
    }
  }

  return success;  
}

//
// Heavyweight Evaluate() function.
// FIXME: Derive FanoutX, FanoutY, MeshX, MeshY from mapping if unspecified.
//
bool BufferLevel::Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                           const double inner_tile_area, const std::uint64_t compute_cycles,
                           const bool break_on_failure)
{
  bool success = ComputeAccesses(tile, mask, break_on_failure);
  if (!break_on_failure || success)
  {
    ComputeArea();
    ComputeBufferEnergy();
    ComputeReductionEnergy();
    ComputeAddrGenEnergy();
    ComputeNetworkEnergy(inner_tile_area);
    ComputePerformance(compute_cycles);
  }
  return success;
}

bool BufferLevel::ComputeAccesses(const tiling::CompoundTile& tile,
                                  const tiling::CompoundMask& mask,
                                  const bool break_on_failure)
{
  bool success = true;
  
  // Subnest FSM should be same for each problem::Shape::DataSpaceID in the list,
  // so just copy it from datatype #0.
  subnest_ = tile[0].subnest;

  // Stats are always collected per-DataSpaceID.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    //
    // Collect stats.
    //
    stats_.keep[pv] = mask[pv];
    
    stats_.partition_size[pv] = tile[pvi].partition_size;
    stats_.utilized_capacity[pv] = tile[pvi].size;
    stats_.utilized_instances[pv] = tile[pvi].replication_factor;

    assert((tile[pvi].size == 0) == (tile[pvi].content_accesses == 0));

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // First epoch is an Update, all subsequent epochs are Read-Modify-Update.
      assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

      stats_.reads[pv] = tile[pvi].content_accesses - tile[pvi].size;
      stats_.updates[pv] = tile[pvi].content_accesses;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar

      // FIXME: temporal reduction and network costs if hardware reduction isn't
      // supported appears to be wonky - network costs may need to trickle down
      // all the way to the level that has the reduction hardware.
      stats_.temporal_reductions[pv] = tile[pvi].content_accesses - tile[pvi].size;

      // Network access-count calculation for Read-Modify-Write datatypes depends on
      // whether the unit receiving a Read-Write datatype has the ability to do
      // a Read-Modify-Write (e.g. accumulate) locally. If the unit isn't capable
      // of doing this, we need to account for additional network traffic.
      // FIXME: take this information from an explicit arch spec.
      // FIXME: need to account for the case when this level is bypassed. In this
      //        case we'll have to query a different level. Also size will be 0,
      //        we may have to maintain a network_size.
      bool hw_reduction_supported =
        !(specs_.Tech(pv).IsSpecified() && specs_.Tech(pv).Get() == Technology::DRAM);
      
      if (hw_reduction_supported)
      {
        stats_.network.ingresses[pv] = tile[pvi].accesses;
      }
      else
      {
        stats_.network.ingresses[pv].resize(tile[pvi].accesses.size());
        for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
        {
          if (tile[pvi].accesses[i] > 0)
          {
            assert(tile[pvi].size == 0 || tile[pvi].accesses[i] % tile[pvi].size == 0);
            stats_.network.ingresses[pv][i] = 2*tile[pvi].accesses[i] - tile[pvi].size;
          }
          else
          {
            stats_.network.ingresses[pv][i] = 0;
          }
        }
      }
    }
    else // Read-only data type.
    {
      stats_.reads[pv] = tile[pvi].content_accesses;
      stats_.updates[pv] = 0;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.reads[pv] + stats_.fills[pv]; // scalar
      stats_.temporal_reductions[pv] = 0;
      stats_.network.ingresses[pv] = tile[pvi].accesses;
    }

    stats_.network.spatial_reductions[pv] = 0;
    stats_.network.distributed_multicast[pv] = tile[pvi].distributed_multicast;
    stats_.network.avg_hops[pv].resize(tile[pvi].accesses.size());
    for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
    {
      if (tile[pvi].accesses[i] > 0)
      {
        stats_.network.avg_hops[pv][i] = tile[pvi].cumulative_hops[i] / double(tile[pvi].scatter_factors[i]);
      }
    }
    
    // FIXME: issues with link-transfer modeling:
    // 1. link transfers should result in buffer accesses to a peer.
    // 2. should reductions via link transfers be counted as spatial or temporal?
    stats_.network.link_transfers[pv] = tile[pvi].link_transfers;
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.network.spatial_reductions[pv] += tile[pvi].link_transfers;
    }

    stats_.network.fanout[pv] = tile[pvi].fanout;
    if (stats_.network.distributed_multicast.at(pv))
      stats_.network.distributed_fanout[pv] = tile[pvi].distributed_fanout;
    else
      stats_.network.distributed_fanout[pv] = 0;

    // FIXME: multicast factor can be heterogeneous. This is correctly
    // handled by energy calculations, but not correctly reported out
    // in the stats.
    stats_.network.multicast_factor[pv] = 0;

    for (unsigned i = 0; i < stats_.network.ingresses[pv].size(); i++)
    {
      if (stats_.network.ingresses[pv][i] > 0)
      {
        auto factor = i + 1;
        if (factor > stats_.network.multicast_factor[pv])
        {
          stats_.network.multicast_factor[pv] = factor;
        }
        if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
        {
          stats_.network.spatial_reductions[pv] += (i * stats_.network.ingresses[pv][i]);
        }
      }
    }

  }

  // Derive/validate architecture specs based on stats.
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      
      if (!specs_.Tech(pv).IsSpecified())
        specs_.Tech(pv) = Technology::SRAM;
      
      if (!specs_.NumPorts(pv).IsSpecified()) {
        specs_.NumPorts(pv) = 2;
      }
        
      if (!specs_.NumBanks(pv).IsSpecified()) {
        specs_.NumBanks(pv) = 2; //FIXME: default 2 banks
      }
      
      if (!specs_.Size(pv).IsSpecified())
        specs_.Size(pv) = std::ceil(stats_.utilized_capacity.at(pv)
                                    * specs_.MultipleBuffering(pv).Get());
      else if (stats_.utilized_capacity.at(pv) > specs_.EffectiveSize(pv).Get())
        success = false;
      else if (stats_.utilized_capacity.at(pv) < specs_.EffectiveSize(pv).Get()
                                                 * specs_.MinUtilization(pv).Get())
        success = false;

      
      assert (specs_.BlockSize(pv).IsSpecified());
      
      assert (specs_.ClusterSize(pv).IsSpecified());

      specs_.AddrGenBits(pv) = static_cast<unsigned long>(
        std::ceil(std::log2(
                    static_cast<double>(specs_.Size(pv).Get()) / specs_.BlockSize(pv).Get()
                    )));
      
      if (!specs_.Instances(pv).IsSpecified())
        specs_.Instances(pv) = stats_.utilized_instances.at(pv);
      else if (stats_.utilized_instances.at(pv) > specs_.Instances(pv).Get())
        success = false;
      
      if (!specs_.Fanout(pv).IsSpecified())
        specs_.Fanout(pv) = stats_.network.fanout.at(pv);
      else if (stats_.network.fanout.at(pv) > specs_.Fanout(pv).Get())
        success = false;

      // Bandwidth constraints cannot be checked/inherited at this point
      // because the calculation is a little more involved. We will do
      // this later in the ComputePerformance() function.
      
      if (break_on_failure && !success)
        break;
    }
  }
  else  // sharing_type == DataSpaceIDSharing::Shared
  {
    if (!specs_.Tech().IsSpecified())
      specs_.Tech() = Technology::SRAM;
      
    if (!specs_.NumPorts().IsSpecified()) {
      specs_.NumPorts() = 2;
    }
      
    if (!specs_.NumBanks().IsSpecified()) {
      specs_.NumBanks() = 2; //FIXME: default 2 banks
    }
      
    auto total_utilized_capacity = std::accumulate(stats_.utilized_capacity.begin(),
                                                   stats_.utilized_capacity.end(),
                                                   0ULL);
    if (!specs_.Size().IsSpecified())
      specs_.Size() = std::ceil(total_utilized_capacity
                                * specs_.MultipleBuffering().Get());
    else if (total_utilized_capacity > specs_.EffectiveSize().Get())
    {
      // std::cout << specs_.Name().Get() << std::endl;
      // std::cout << "Tile size = " << std::endl;
      // std::cout << stats_.utilized_capacity;
      // std::cout << "Size = " << specs_.Size().Get() << std::endl;
      success = false;
    }
    else if (total_utilized_capacity < specs_.EffectiveSize().Get()
                                       * specs_.MinUtilization().Get())
      success = false;

    assert (specs_.BlockSize().IsSpecified());
    
    assert (specs_.ClusterSize().IsSpecified());
      
    specs_.AddrGenBits() = static_cast<unsigned long>(
      std::ceil(std::log2(
                  static_cast<double>(specs_.Size().Get()) / specs_.BlockSize().Get()
                  )));
      
    if (!specs_.Instances().IsSpecified())
      specs_.Instances() = stats_.utilized_instances.Max();
    else if (stats_.utilized_instances.Max() > specs_.Instances().Get())
      success = false;

    // Note: the following calculation uses the Max of fanouts
    // because it assumes that the fanout network is shared if the buffer
    // itself is shared. We should perhaps relax this.
    if (!specs_.Fanout().IsSpecified())
      specs_.Fanout() = stats_.network.fanout.Max();
    else if (stats_.network.fanout.Max() > specs_.Fanout().Get())
    {
      // This CANNOT happen, mapspace should have taken care of this.
      // assert(false);
      std::cerr << "WARNING: fanout FAIL level = " << specs_.level_name << " req = "
                << stats_.network.fanout.Max() << " avail = "
                << specs_.Fanout().Get() << std::endl;
      std::cerr << "I'm invalidating this mapping, but the mapping constructor "
                << "in the mapspace should have taken care of this." << std::endl;
      success = false;
    }

    // Bandwidth constraints cannot be checked/inherited at this point
    // because the calculation is a little more involved. We will do
    // this later in the ComputePerformance() function.      
  }

  // Compute utilized clusters.
  // FIXME: should derive this from precise spatial mapping.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // The following equation assumes fully condensed mapping. Do a ceil-div.
    // stats_.utilized_clusters[pv] = 1 + (stats_.utilized_instances[pv] - 1) /
    //    specs_.ClusterSize(pv).Get();
    // Assume utilized instances are sprinkled uniformly across all clusters.
    auto num_clusters = specs_.Instances(pv).Get() / specs_.ClusterSize(pv).Get();
    stats_.utilized_clusters[pv] = std::min(stats_.utilized_instances[pv],
                                            num_clusters);
  }

  is_evaluated_ = success;
    
  return success;
}

void BufferLevel::ComputeArea()
{
  // YUCK. FIXME. The area is now already stored in a specs_ attribute.
  // The stats_ here are just a copy of the specs_. Do we really need both?
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    stats_.area.SetPerDataSpace();
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      stats_.area[pv] =  specs_.StorageArea(pv).Get();
    }
  }
  else  // sharing_type == DataSpaceIDSharing::Shared
  {
    // Store the total area in the Num/All slot. We used to split it up between the
    // various data-types depending on their contribution towards utilization, but
    // this gives incorrect area if the structure is underutilized, especially when
    // the level is completely bypassed.
    stats_.area.SetShared(specs_.StorageArea().Get());
  }
}

// Compute buffer energy.
void BufferLevel::ComputeBufferEnergy()
{
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto instance_accesses = stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv);

    auto block_size = specs_.BlockSize(pv).Get();
    double vector_accesses =
      (instance_accesses % block_size == 0) ?
      (instance_accesses / block_size)      :
      (instance_accesses / block_size) + 1;
    
    double cluster_access_energy = vector_accesses *
      specs_.VectorAccessEnergy(pv).Get();

    // Spread out the cost between the utilized instances in each cluster.
    // This is because all the later stat-processing is per-instance.
    double cluster_utilization = double(stats_.utilized_instances.at(pv)) /
      double(stats_.utilized_clusters.at(pv));
    stats_.energy[pv] = cluster_access_energy / cluster_utilization;
    stats_.energy_per_access[pv] = stats_.energy.at(pv) / instance_accesses;
  }
}

// Compute network energy.
void BufferLevel::ComputeNetworkEnergy(const double inner_tile_area)
{
#define PROBABILISTIC_MULTICAST 0
#define PRECISE_MULTICAST 1
#define EYERISS_HACK_MULTICAST 2  

#define MULTICAST_MODEL PROBABILISTIC_MULTICAST
  
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    double energy_per_hop =
            WireEnergyPerHop(specs_.NetworkWordBits(pv).Get(), inner_tile_area);
    double energy_per_router = specs_.RouterEnergy(pv).Get();
    
    auto fanout = stats_.network.distributed_multicast.at(pv) ?
      stats_.network.distributed_fanout.at(pv) :
      stats_.network.fanout.at(pv);

    double total_wire_hops = 0;
    std::uint64_t total_routers_touched = 0;
    double total_ingresses = 0;
    
    for (unsigned i = 0; i < stats_.network.ingresses[pv].size(); i++)
    {
      auto ingresses = stats_.network.ingresses.at(pv).at(i);
      total_ingresses += ingresses;
      if (ingresses > 0)
      {
        auto multicast_factor = i + 1;

#if MULTICAST_MODEL == PROBABILISTIC_MULTICAST

        auto num_hops = NumHops(multicast_factor, fanout);
        total_routers_touched += (1 + num_hops) * ingresses;

#elif MULTICAST_MODEL == PRECISE_MULTICAST

        (void)fanout;
        (void)multicast_factor;
        if (stats_.network.distributed_multicast.at(pv))
        {
          std::cerr << "ERROR: precise multicast calculation does not work with distributed multicast." << std::endl;
          exit(1);
        }
        auto num_hops = stats_.network.avg_hops.at(pv).at(i);
        total_routers_touched += (1 + std::uint64_t(std::floor(num_hops))) * ingresses;

#elif MULTICAST_MODEL == EYERISS_HACK_MULTICAST

        (void)fanout;
        unsigned num_hops = 0;
        
        // Weights are multicast, and energy is already captured in array access.
        if (pv != problem::Shape::DataSpaceID::Weight)
        {
          // Input and Output activations are forwarded between neighboring PEs,
          // so the number of link transfers is equal to the multicast factor-1.
          num_hops = multicast_factor - 1;
        }
        
        // We pick up the router energy from the .cfg file as the "array" energy
        // as defined in the Eyeriss paper, so we don't add a 1 to the multicast
        // factor.
        total_routers_touched += num_hops * ingresses;

#else
#error undefined MULTICAST_MODEL
#endif        

        total_wire_hops += num_hops * ingresses;
      }
    }

    stats_.network.energy_per_hop[pv] = energy_per_hop;
    stats_.network.num_hops[pv] = total_ingresses > 0 ? total_wire_hops / total_ingresses : 0;
    stats_.network.energy[pv] =
      total_wire_hops * energy_per_hop + // wire energy
      total_routers_touched * energy_per_router; // router energy

    stats_.network.link_transfer_energy[pv] =
      stats_.network.link_transfers.at(pv) * (energy_per_hop + 2*energy_per_router);
  }
}

//
// Compute reduction energy.
//
void BufferLevel::ComputeReductionEnergy()
{
  // Temporal reduction: add a value coming in on the network to a value stored locally.
  // Spatial reduction: add two values in the network.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.temporal_reduction_energy[pv] = stats_.temporal_reductions[pv] * 
        pat::AdderEnergy(specs_.WordBits(pv).Get(), specs_.NetworkWordBits(pv).Get());
      
      stats_.network.spatial_reduction_energy[pv] = stats_.network.spatial_reductions[pv] * 
        pat::AdderEnergy(specs_.NetworkWordBits(pv).Get(), specs_.NetworkWordBits(pv).Get());
    }
    else
    {
      stats_.temporal_reduction_energy[pv] = 0;
      stats_.network.spatial_reduction_energy[pv] = 0;
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
      pat::AdderEnergy(specs_.AddrGenBits(pv).Get(), specs_.AddrGenBits(pv).Get());
  }
}

//
// Compute performance.
//
void BufferLevel::ComputePerformance(const std::uint64_t compute_cycles)
{
  // Ugh... have to fix per-datatype word size.
  double word_size = 0;
  for (unsigned pvi = 0; pvi < specs_.NumPartitions(); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    word_size = std::max(word_size, double(specs_.WordBits(pv).Get()) / 8);
  }
  
  //
  // Step 1: Compute unconstrained bandwidth demand.
  //
  problem::PerDataSpace<double> unconstrained_read_bandwidth;
  problem::PerDataSpace<double> unconstrained_write_bandwidth;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // FIXME: move the following code to Network bandwidth calculation.
    // auto total_ingresses =
    //   std::accumulate(stats_.network.ingresses.at(pv).begin(),
    //                   stats_.network.ingresses.at(pv).end(), static_cast<std::uint64_t>(0));
    auto total_read_accesses    =   stats_.reads.at(pv);
    auto total_write_accesses   =   stats_.updates.at(pv) + stats_.fills.at(pv);
    unconstrained_read_bandwidth[pv]  = (double(total_read_accesses)  / compute_cycles) * word_size;
    unconstrained_write_bandwidth[pv] = (double(total_write_accesses) / compute_cycles) * word_size;
  }

  //
  // Step 2: Compare vs. specified bandwidth and calculate slowdown.
  //
  stats_.slowdown = 1.0;
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    // Find worst slowdown (if bandwidth was specified).
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      if (specs_.ReadBandwidth(pv).IsSpecified() &&
          specs_.ReadBandwidth(pv).Get() < unconstrained_read_bandwidth.at(pv))
      {
        stats_.slowdown =
          std::min(stats_.slowdown,
                   specs_.ReadBandwidth(pv).Get() / unconstrained_read_bandwidth.at(pv));
      }
      if (specs_.WriteBandwidth(pv).IsSpecified() &&
          specs_.WriteBandwidth(pv).Get() < unconstrained_write_bandwidth.at(pv))
      {
        stats_.slowdown =
          std::min(stats_.slowdown,
                   specs_.WriteBandwidth(pv).Get() / unconstrained_write_bandwidth.at(pv));
      }
    }

  }
  else // specs_.sharing_type == DataSpaceIDSharing::Shared
  {
    // Find slowdown.
    auto total_unconstrained_read_bandwidth  = std::accumulate(unconstrained_read_bandwidth.begin(),  unconstrained_read_bandwidth.end(),  0.0);
    auto total_unconstrained_write_bandwidth = std::accumulate(unconstrained_write_bandwidth.begin(), unconstrained_write_bandwidth.end(), 0.0);

    if (specs_.ReadBandwidth().IsSpecified() &&
        specs_.ReadBandwidth().Get() < total_unconstrained_read_bandwidth)
    {
        stats_.slowdown =
          std::min(stats_.slowdown,
                   specs_.ReadBandwidth().Get() / total_unconstrained_read_bandwidth);
    }
    if (specs_.WriteBandwidth().IsSpecified() &&
        specs_.WriteBandwidth().Get() < total_unconstrained_write_bandwidth)
    {
        stats_.slowdown =
          std::min(stats_.slowdown,
                   specs_.WriteBandwidth().Get() / total_unconstrained_write_bandwidth);
    }
  }

  //
  // Step 3: For both shared and partitioned buffers,
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
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      if (!specs_.ReadBandwidth(pv).IsSpecified())
        specs_.ReadBandwidth(pv) = stats_.read_bandwidth.at(pv);        
      if (!specs_.WriteBandwidth(pv).IsSpecified())
        specs_.WriteBandwidth(pv) = stats_.write_bandwidth.at(pv);        
    }
  }
  else
  {
    if (!specs_.ReadBandwidth().IsSpecified())
      specs_.ReadBandwidth() = std::accumulate(stats_.read_bandwidth.begin(), stats_.read_bandwidth.end(), 0.0);
    if (!specs_.WriteBandwidth().IsSpecified())
      specs_.WriteBandwidth() = std::accumulate(stats_.write_bandwidth.begin(), stats_.write_bandwidth.end(), 0.0);
  }
}

//
// Accessors.
//

#define STAT_ACCESSOR(Type, FuncName, Expression)                                     \
Type BufferLevel::FuncName(problem::Shape::DataSpaceID pv) const                      \
{                                                                                     \
  if (pv != problem::GetShape()->NumDataSpaces)                                       \
  {                                                                                   \
    return Expression;                                                                \
  }                                                                                   \
  else                                                                                \
  {                                                                                   \
    Type stat = 0;                                                                    \
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++) \
    {                                                                                 \
      stat += FuncName(problem::Shape::DataSpaceID(pvi));                             \
    }                                                                                 \
    return stat;                                                                      \
  }                                                                                   \
}

// double BufferLevel::StorageEnergy(problem::Shape::DataSpaceID pv) const
// {                                                                         
//   if (pv != problem::GetShape()->NumDataSpaces)                                       
//   {                                                                       
//     return stats_.energy.at(pv) * stats_.utilized_instances.at(pv);                                                    
//   }                                                                       
//   else                                                                    
//   {                                                                       
//     double stat = 0;                                                        
//     for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++) 
//     {                                                                     
//       stat += StorageEnergy(problem::Shape::DataSpaceID(pvi));                           
//     }                                                                     
//     return stat;                                                          
//   }                                                                       
// }

STAT_ACCESSOR(double, StorageEnergy, stats_.energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, NetworkEnergy, (stats_.network.link_transfer_energy.at(pv) + stats_.network.energy.at(pv)) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, TemporalReductionEnergy, stats_.temporal_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, SpatialReductionEnergy, stats_.network.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, AddrGenEnergy, stats_.addr_gen_energy.at(pv) * stats_.utilized_clusters.at(pv)) // Note!!! clusters, not instances.
STAT_ACCESSOR(double, Energy,
              StorageEnergy(pv) +
              NetworkEnergy(pv) +
              TemporalReductionEnergy(pv) +
              SpatialReductionEnergy(pv) +
              AddrGenEnergy(pv))

STAT_ACCESSOR(std::uint64_t, Accesses, stats_.utilized_instances.at(pv) * (stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv)))
STAT_ACCESSOR(std::uint64_t, UtilizedCapacity, stats_.utilized_capacity.at(pv))

std::string BufferLevel::Name() const
{
  return (specs_.sharing_type == DataSpaceIDSharing::Shared ?
          specs_.Name().Get() :
          specs_.Name(problem::Shape::DataSpaceID(0)).Get());  
}

double BufferLevel::Area() const
{
  double area = 0;
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      area += specs_.StorageArea(pv).Get() * specs_.Instances(pv).Get();
    }
  }
  else
  {
    area += specs_.StorageArea().Get() * specs_.Instances().Get();
  }
  return area;
}

double BufferLevel::AreaPerInstance() const
{
  double area = 0;
  if (specs_.sharing_type == DataSpaceIDSharing::Partitioned)
  {
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      area += stats_.area.at(pv);
    }
  }
  else
  {
    auto pv = problem::GetShape()->NumDataSpaces;
    area += stats_.area.at(pv);
  }  
  return area;
}

double BufferLevel::Size() const
{
  // FIXME: this is per-instance. This is inconsistent with the naming
  // convention of some of the other methods, which are summed across instances.
  double size = 0;
  for (unsigned pvi = 0; pvi < specs_.NumPartitions(); pvi++)
    size += specs_.Size(problem::Shape::DataSpaceID(pvi)).Get();
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

  double total_capacity = Size() * specs_.Instances().Get();

  return utilized_capacity / total_capacity;
}

std::uint64_t BufferLevel::Cycles() const
{
  return stats_.cycles;
}

// ----------------------
//    PAT interfacing
// ----------------------

double BufferLevel::WireEnergyPerHop(std::uint64_t word_bits, double inner_tile_area)
{
  // Assuming square modules
  double inner_tile_width = std::sqrt(inner_tile_area);  // um
  inner_tile_width /= 1000;                              // mm
  return pat::WireEnergy(word_bits, inner_tile_width);
}

double BufferLevel::NumHops(std::uint32_t multicast_factor, std::uint32_t fanout)
{
  // Assuming central/side entry point.
  double root_f = std::sqrt(multicast_factor);
  double root_n = std::sqrt(fanout);
  return (root_n*root_f) + 0.5*(root_n-root_f) - (root_n/root_f) + 0.5;
  // return (root_n*root_f);
}

std::ostream& operator<<(std::ostream& out, const BufferLevel::Technology& tech)
{
  switch (tech)
  {
    case BufferLevel::Technology::SRAM: out << "SRAM"; break;
    case BufferLevel::Technology::DRAM: out << "DRAM"; break;
  }
  return out;
}

// ---------------
//    Printers
// ---------------

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
  unsigned start_pv = specs.DataSpaceIDIteratorStart();
  unsigned end_pv = specs.DataSpaceIDIteratorEnd();

  for (unsigned pvi = start_pv; pvi < end_pv; pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    if (specs.sharing_type == BufferLevel::DataSpaceIDSharing::Partitioned && specs.Name(pv).IsSpecified())
    {
      out << indent << "= " << specs.Name(pv).Get() << " =" << std::endl;
    }

    if (pv == problem::GetShape()->NumDataSpaces)
      out << indent << "Shared:" << std::endl;
    else
      out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;      
    out << indent << indent << "Technology           : " << specs.Tech(pv) << std::endl;
    out << indent << indent << "Size                 : " << specs.Size(pv) << std::endl;
    out << indent << indent << "Word bits            : " << specs.WordBits(pv) << std::endl;    
    out << indent << indent << "Block size           : " << specs.BlockSize(pv) << std::endl;
    out << indent << indent << "Cluster size         : " << specs.ClusterSize(pv) << std::endl;
    out << indent << indent << "Instances            : " << specs.Instances(pv) << " ("
        << specs.MeshX(pv) << "*" << specs.MeshY(pv) << ")" << std::endl;
    out << indent << indent << "Fanout               : " << specs.Fanout(pv) << " ("
        << specs.FanoutX(pv) << "*" << specs.FanoutY(pv) << ")" << std::endl;
    out << indent << indent << "Read bandwidth       : " << specs.ReadBandwidth(pv) << std::endl;    
    out << indent << indent << "Write bandwidth      : " << specs.WriteBandwidth(pv) << std::endl;    
    out << indent << indent << "Multiple buffering   : " << specs.MultipleBuffering(pv) << std::endl;
    out << indent << indent << "Effective size       : " << specs.EffectiveSize(pv) << std::endl;
    out << indent << indent << "Min utilization      : " << specs.MinUtilization(pv) << std::endl;
    out << indent << indent << "Vector access energy : " << specs.VectorAccessEnergy(pv) << " pJ" << std::endl;
    out << indent << indent << "Area                 : " << specs.StorageArea(pv) << " um^2" << std::endl;
  }
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

      // Partition size calculation is incorrect in nest-analysis.cpp, so do NOT print it out.
      // out << indent + indent << "Partition size                           : " << stats.partition_size.at(pv) << std::endl;
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

    if (specs.sharing_type == BufferLevel::DataSpaceIDSharing::Partitioned)
    {
      if (stats.utilized_capacity.at(pv) == 0 && specs.Size(pv).IsSpecified() && specs.Size(pv).Get() > 0)
        out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << std::endl;
      
      if (stats.utilized_capacity.at(pv) > 0 || (specs.Size(pv).IsSpecified() && specs.Size(pv).Get() > 0))
        out << indent + indent << "Area (per-instance)                      : " << stats.area.at(pv) << " um2" << std::endl;
    }      
  }

  // Print area for shared buffers.
  if (specs.sharing_type == BufferLevel::DataSpaceIDSharing::Shared)
  {
    auto pv = problem::GetShape()->NumDataSpaces;
    out << indent << "Shared:" << std::endl;
    out << indent + indent << "Area (per-instance)                      : " << stats.area.at(pv) << " um2" << std::endl;
  }
  
  out << std::endl;

  out << indent << "NETWORK STATS" << std::endl;
  out << indent << "-------------" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

    out << indent + indent << "Fanout                                  : "
        << stats.network.fanout.at(pv) << std::endl;
    out << indent + indent << "Fanout (distributed)                    : "
        << stats.network.distributed_fanout.at(pv) << std::endl;
    if (stats.network.distributed_multicast.at(pv))
      out << indent + indent << "Multicast factor (distributed)          : ";
    else
      out << indent + indent << "Multicast factor                        : ";
    out << stats.network.multicast_factor.at(pv) << std::endl;
      
    auto total_accesses =
      std::accumulate(stats.network.ingresses.at(pv).begin(),
                      stats.network.ingresses.at(pv).end(),
                      static_cast<std::uint64_t>(0));
    out << indent + indent << "Ingresses                               : " << total_accesses << std::endl;
    
    std::string mcast_type = "@multicast ";
    if (stats.network.distributed_multicast.at(pv))
      mcast_type += "(distributed) ";
    for (std::uint64_t i = 0; i < stats.network.ingresses.at(pv).size(); i++)
      if (stats.network.ingresses.at(pv)[i] != 0)
        out << indent + indent + indent << mcast_type << i + 1 << ": "
            << stats.network.ingresses.at(pv)[i] << std::endl;

    out << indent + indent << "Link transfers                          : "
        << stats.network.link_transfers.at(pv) << std::endl;
    out << indent + indent << "Spatial reductions                      : "
        << stats.network.spatial_reductions.at(pv) << std::endl;
    
    out << indent + indent << "Average number of hops                  : "
        << stats.network.num_hops.at(pv) << std::endl;
    
    out << indent + indent << "Energy (per-hop)                        : "
        << stats.network.energy_per_hop.at(pv)*1000 << " fJ" << std::endl;

    out << indent + indent << "Energy (per-instance)                   : "
        << stats.network.energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Energy (total)                          : "
        << stats.network.energy.at(pv) * stats.utilized_instances.at(pv)
        << " pJ" << std::endl;
    out << indent + indent << "Link transfer energy (per-instance)     : "
        << stats.network.link_transfer_energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Link transfer energy (total)            : "
        << stats.network.link_transfer_energy.at(pv) * stats.utilized_instances.at(pv)
        << " pJ" << std::endl;    
    out << indent + indent << "Spatial Reduction Energy (per-instance) : "
        << stats.network.spatial_reduction_energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Spatial Reduction Energy (total)        : "
        << stats.network.spatial_reduction_energy.at(pv) * stats.utilized_instances.at(pv)
        << " pJ" << std::endl;
  }
  
  out << std::endl;
}

}  // namespace model
