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
#include <cmath>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/buffer.hpp"
//BOOST_CLASS_EXPORT(model::BufferLevel::Specs)
BOOST_CLASS_EXPORT(model::BufferLevel)

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "pat/pat.hpp"
#include "topology.hpp"

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
  is_specced_ = true;
  is_evaluated_ = false;
}

BufferLevel::~BufferLevel()
{ }

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.
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
    assert(block_size != 0);
  }

  // MetaData Block size
  std::uint32_t metadata_block_size;
  specs.metadata_block_size = 1;
  if (buffer.lookupValue("metadata-block-size", metadata_block_size))
  {
    specs.metadata_block_size = metadata_block_size;
  }

  // Metadata data width
  // we currently consider metadata to be stored in the same storage
  // metadata data width is important to get a realistic size for the metadata
  // default to 0 -> no metadata
  // FIXME: consider metatdata as its own dataspace
  std::uint32_t metadata_word_bits;
  specs.metadata_word_bits = 0;
  if (buffer.lookupValue("metadata_datawidth", metadata_word_bits)){
     specs.metadata_word_bits = metadata_word_bits;
  }

  // Cluster size.
  std::uint32_t cluster_size;
  specs.cluster_size = 1;
  std::uint32_t width;
  if (buffer.lookupValue("cluster-size", cluster_size))
  {
    specs.cluster_size = cluster_size;
  }
  else if (buffer.lookupValue("width", width)||
           buffer.lookupValue("memory_width", width))
  {
    word_bits = specs.word_bits.Get();
    block_size = specs.block_size.Get();
    if (width % (word_bits * block_size)  != 0){
      std::cout << "ERROR: width: " << width << "  block_size: " << block_size << "  word_bits: " << word_bits << std::endl;
    }
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
  if (className.find("DRAM") != std::string::npos) specs.technology = Technology::DRAM;

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

  // Network names;
  std::string read_network_name;
  if (buffer.lookupValue("network_read", read_network_name))
  {
    specs.read_network_name = read_network_name;
  }

  std::string fill_network_name;
  if (buffer.lookupValue("network_fill", fill_network_name))
  {
    specs.fill_network_name = fill_network_name;
  }

  std::string drain_network_name;
  if (buffer.lookupValue("network_drain", drain_network_name))
  {
    specs.drain_network_name = drain_network_name;
  }

  std::string update_network_name;
  if (buffer.lookupValue("network_update", update_network_name))
  {
    specs.update_network_name = update_network_name;
  }

  bool allow_overbooking = false;
  if (buffer.lookupValue("allow_overbooking", allow_overbooking)){
     specs.allow_overbooking = allow_overbooking;
  } else {
     specs.allow_overbooking = false;
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

  // Allow user to override the addr gen energy.
  double tmp_addr_gen_energy = -0.1;
  buffer.lookupValue("addr-gen-energy", tmp_addr_gen_energy);
  specs.addr_gen_energy = tmp_addr_gen_energy;

  // Allow user to override the cluster area.
  double tmp_cluster_area = 0;
  buffer.lookupValue("cluster-area", tmp_cluster_area);
  if (tmp_cluster_area > 0)
    tmp_storage_area = tmp_cluster_area / specs.cluster_size.Get();

  // Set final physical dimensions and energy.
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


void BufferLevel::PopulateEnergyPerOp(unsigned num_ops){

  if (! populate_energy_per_op){

    double ert_energy_per_op;
    bool  ert_energy_found;
    std::vector<std::string> ert_action_names;
    std::string op_name;

    for (unsigned op_id = 0; op_id < num_ops; op_id++){
      // go through all op types
      ert_energy_per_op = 0;
      ert_energy_found = false;
      op_name = tiling::storageOperationTypes[op_id];

     // initialize to the pat values or zero in case no mapping is found
      if (op_name.find("random_read") != std::string::npos
          || op_name.find("random_fill") != std::string::npos
          || op_name.find("random_update") != std::string::npos){
            // use the max if no mapping is found for regular memory actions
            ert_energy_per_op = specs_.vector_access_energy.Get();
      } else {
          // use zero if no mapping is found for matadata/gated/skipped/decompression/compression actions
          ert_energy_per_op = 0;
      }

      // go through ERT entries and look for appopriate energy values
      // std::cout <<"operation name: " << op_name << std::endl;
      ert_action_names = model::storageOperationMappings.at(op_name);
      for (auto it = ert_action_names.begin(); it != ert_action_names.end(); it++){
        if(specs_.ERT_entries.count(*it)>0 && (!ert_energy_found)){
          ert_energy_per_op = specs_.ERT_entries.at(*it);
          ert_energy_found = true;
        }
      }
      // populate the op_energy_map data structure for easier future energy search
      specs_.op_energy_map[op_name] = ert_energy_per_op;
  }
  populate_energy_per_op = true;

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
  const problem::Workload* workload,
  const sparse::PerStorageLevelCompressionInfo per_level_compression_info,
  const double confidence_threshold,
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
    if (network_read_->DistributedMulticastSupported())
    {
      available_capacity *= specs_.instances.Get();
    }

    // Find the total capacity required by all un-masked data types.
    std::size_t required_capacity = 0;
    double confidence_constraint = ! specs_.allow_overbooking.Get() ? 1.0 : confidence_threshold;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      if (mask[pvi])
      {
        auto dense_working_set_size = working_set_sizes.at(problem::Shape::DataSpaceID(pvi));
        auto working_set_size = dense_working_set_size;

        std::string data_space_name = problem::GetShape()->DataSpaceIDToName.at(pvi);

        if (per_level_compression_info.find(data_space_name) != per_level_compression_info.end()
            && per_level_compression_info.at(data_space_name).compressed){
            working_set_size = workload->GetDensity(pvi)->GetTileOccupancyByConfidence(dense_working_set_size, confidence_constraint);
        } else {
            working_set_size = ceil(dense_working_set_size * confidence_constraint);
        }
        required_capacity += working_set_size;
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
                                 const double confidence_threshold, const std::uint64_t compute_cycles,
                                 const bool break_on_failure)
{
  auto eval_status = ComputeScalarAccesses(tile.data_movement_info, mask, confidence_threshold, break_on_failure);
  if (!break_on_failure || eval_status.success)
  {
    ComputeVectorAccesses(tile.data_movement_info);
    ComputeBufferEnergy(tile.data_movement_info);
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

void BufferLevel::ConnectRead(std::shared_ptr<Network> network)
{
  network_read_ = network;
}

void BufferLevel::ConnectFill(std::shared_ptr<Network> network)
{
  network_fill_ = network;
}

void BufferLevel::ConnectUpdate(std::shared_ptr<Network> network)
{
  network_update_ = network;
}

void BufferLevel::ConnectDrain(std::shared_ptr<Network> network)
{
  network_drain_ = network;
}


uint64_t GetMetaDataTileSize(tiling::DataMovementInfo per_datatype_tile_info, double tile_density){

    uint64_t metadata_tile_size;

    if (!per_datatype_tile_info.compressed){
       return 0;
    }

    // compute the corresponding metadata tile size
    if (per_datatype_tile_info.metadata_format == "bitmask"){
       metadata_tile_size = per_datatype_tile_info.size;

    } else if (per_datatype_tile_info.metadata_format == "RLE"){
       // metadata_tile_size = compressed_tile_size;
       metadata_tile_size = ceil(per_datatype_tile_info.size * tile_density);

    } else if (per_datatype_tile_info.metadata_format == "CSR"){
       metadata_tile_size = per_datatype_tile_info.dense_rank1_fills
                            + per_datatype_tile_info.dense_rank0_fills * tile_density;

    } else {
       metadata_tile_size = 0;
    }

    return metadata_tile_size;
}

void BufferLevel::ComputeTileOccupancyAndConfidence(const tiling::CompoundDataMovementInfo& tile,
                                                    const double confidence_threshold){

  // collect tile sizes (data + metadata) for all dataspaces stored at the storage level
  // used for better distribution storage capacity to different dataspaces stored at this level
  double total_tile_size = 0;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++){
      if (tile[pvi].compressed){
           total_tile_size += tile[pvi].size * tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size);
           total_tile_size += ceil(GetMetaDataTileSize(tile[pvi], tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size))
                                   * 1.0 * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());
      } else {
           total_tile_size += tile[pvi].size;
      }
  }

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    stats_.tile_size[pv] = tile[pvi].size;

    double tile_confidence = 1.0;
    uint64_t compressed_tile_size = 0;
    uint64_t metadata_tile_size = 0;
    double stored_data_density = 1.0;

    // compute tile occupancy and associated confidence
    // if the tile is not compressed, the confidence is just zero or one derived directly by a comparison of tile shape and allocated capacity
    if (tile[pvi].compressed)
    {

      if (specs_.effective_size.IsSpecified()){

        uint64_t allocated_effective_buffer_size; // buffer capacity allocated to this dataspace

        uint64_t equivalent_metadata_tile_size; // metadata tile size normalized to data bitwidth
        uint64_t updated_equivalent_metadata_tile_size; // updated metadata tile size generated by updated tile size

        // initialize metadata tile size to be the expected value, as we do not know how much should be overbooked at this point
        metadata_tile_size = GetMetaDataTileSize(tile[pvi], tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size));
        equivalent_metadata_tile_size = ceil((double)metadata_tile_size * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());

        // assign the dataspace storage capacity according to its tile size
        if (total_tile_size != 0){
          double ratio = (tile[pvi].size * tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size) + equivalent_metadata_tile_size)/ total_tile_size;
          allocated_effective_buffer_size = ceil(specs_.effective_size.Get() * ratio);
        } else {
          allocated_effective_buffer_size = specs_.effective_size.Get() ;
        }

        // confidence constraint is only useful when we actually allow overbooking for this memory level
        double confidence_constraint = specs_.allow_overbooking.Get() ? confidence_threshold: 1.0;
        tile_confidence = tile[pvi].tile_density->GetTileConfidenceByAllocatedCapacity(tile[pvi].size, allocated_effective_buffer_size - equivalent_metadata_tile_size);
        stored_data_density = tile[pvi].tile_density->GetTileDensityByConfidence(tile[pvi].size,
                                                                                tile_confidence,
                                                                                allocated_effective_buffer_size - equivalent_metadata_tile_size);
        compressed_tile_size = ceil(tile[pvi].size * stored_data_density);
        metadata_tile_size = GetMetaDataTileSize(tile[pvi], stored_data_density);
        equivalent_metadata_tile_size = ceil((double)metadata_tile_size * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());


        // Regenerate a conservative estimation if
        // 1) tile takes too much space
        // 2) there is still hope to get a more conservative tile occupancy (tile_confidence != 0)
        // 3) there is still margin comparing to the confidence constraint (tile_confidence > confidence_constraint)
        if (equivalent_metadata_tile_size + compressed_tile_size > allocated_effective_buffer_size
            && tile_confidence != 0 && tile_confidence > confidence_constraint){

           tile_confidence = tile[pvi].tile_density->GetTileConfidenceByAllocatedCapacity(tile[pvi].size, allocated_effective_buffer_size - equivalent_metadata_tile_size);
           stored_data_density = tile[pvi].tile_density->GetTileDensityByConfidence(tile[pvi].size,
                                                                                   tile_confidence,
                                                                                   allocated_effective_buffer_size - equivalent_metadata_tile_size);

           compressed_tile_size = ceil(tile[pvi].size * stored_data_density);
           metadata_tile_size = GetMetaDataTileSize(tile[pvi], stored_data_density);
           updated_equivalent_metadata_tile_size = ceil((double)metadata_tile_size * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());

           assert(updated_equivalent_metadata_tile_size + compressed_tile_size <= allocated_effective_buffer_size);

        } else {
           updated_equivalent_metadata_tile_size = equivalent_metadata_tile_size;
        }

        // if the occupancy is larger than allocated capacity, it means even with the currently assigned occupancy,
        // the confidence constraint is not met -> no need to search for any other options, will be rejected

        // if the occupancy is smaller than allocated capacity, it means we can strive to have larger occupancy to fit
        // in the allocated capacity, leading to higher confidence and better utilization of memory -> search for it

        // keep looking for better occupancy values (that result in higher tile confidences)
        // given the available allocated storage capacity if the above estimation is too conservative
        // stop searching when
        //  1) the tile occupancy is larger than 0.99 of the allocated effective buffer size -- good enough
        //  2) cannot find better storage occupancy values

        uint64_t tmp_metadata_tile_size;
        double tmp_tile_confidence;
        double tmp_stored_data_density;
        uint64_t tmp_compressed_tile_size;

        if (updated_equivalent_metadata_tile_size + compressed_tile_size <= 0.99 * allocated_effective_buffer_size){

          while( updated_equivalent_metadata_tile_size + compressed_tile_size <= 0.99 * allocated_effective_buffer_size &&
                 updated_equivalent_metadata_tile_size != equivalent_metadata_tile_size){

            equivalent_metadata_tile_size = updated_equivalent_metadata_tile_size;
            tmp_tile_confidence = tile[pvi].tile_density->GetTileConfidenceByAllocatedCapacity(tile[pvi].size, allocated_effective_buffer_size - equivalent_metadata_tile_size);
            tmp_stored_data_density = tile[pvi].tile_density->GetTileDensityByConfidence(tile[pvi].size,
                                                                                        tmp_tile_confidence,
                                                                                        allocated_effective_buffer_size - equivalent_metadata_tile_size);
            tmp_compressed_tile_size = ceil(tile[pvi].size * tmp_stored_data_density);
            tmp_metadata_tile_size = GetMetaDataTileSize(tile[pvi], tmp_stored_data_density);

            updated_equivalent_metadata_tile_size = ceil(tmp_metadata_tile_size * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());

            if (updated_equivalent_metadata_tile_size + tmp_compressed_tile_size > allocated_effective_buffer_size){
                updated_equivalent_metadata_tile_size = equivalent_metadata_tile_size;
            }

            if (updated_equivalent_metadata_tile_size != equivalent_metadata_tile_size){
                // the search is only supposed to find better tile confidence values
                assert(tmp_tile_confidence >= tile_confidence);
                metadata_tile_size = tmp_metadata_tile_size;
                compressed_tile_size = tmp_compressed_tile_size;
                stored_data_density = tmp_stored_data_density;
                tile_confidence = tmp_tile_confidence;
            }
          }

          // the updated occupancy must meet capacity requirement
          assert(updated_equivalent_metadata_tile_size + compressed_tile_size <= allocated_effective_buffer_size);

        }
      } else {
        // infinite memory size, e.g., DRAM, can fit for sure
        tile_confidence = 1.0;
        stored_data_density = tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size);
        compressed_tile_size = ceil(tile[pvi].size * stored_data_density);
        metadata_tile_size = GetMetaDataTileSize(tile[pvi], stored_data_density);
      }

    } else { // no compression

       if (tile[pvi].metadata_format == "bitmask"){
         metadata_tile_size = tile[pvi].size;
       }
      compressed_tile_size = tile[pvi].size;
    }

    stats_.tile_confidence[pv] = tile_confidence;
    stats_.compressed_tile_size[pv] = compressed_tile_size;
    stats_.metadata_tile_size[pv] = metadata_tile_size;
    stats_.tile_max_density[pv] = stored_data_density;
    stats_.tile_density_distribution[pv]  = tile[pvi].tile_density->GetDistributionType();

    stats_.utilized_capacity[pv] = compressed_tile_size
                                   + ceil((double)metadata_tile_size
                                     * specs_.metadata_word_bits.Get() / specs_.word_bits.Get());
  }
}

EvalStatus BufferLevel::ComputeScalarAccesses(const tiling::CompoundDataMovementInfo& tile,
                                              const tiling::CompoundMask& mask,
                                              const double confidence_threshold,
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
    stats_.tile_size[pv] = tile[pvi].size;
    stats_.utilized_instances[pv] = tile[pvi].replication_factor;

    assert((tile[pvi].size == 0) == (tile[pvi].content_accesses == 0));

    //
    // the commented calculations below is now moved to tiling.cpp
    //
   
    // if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    // {
    //   // First epoch is an Update, all subsequent epochs are Read-Modify-Update.

    //   // The following assertion is *incorrect* for coefficients (e.g. stride, pad) > 1.
    //   // FIXME: find a safety check that works with coefficients > 1.
    //   // assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

    //   stats_.reads[pv] = tile[pvi].content_accesses - tile[pvi].partition_size + tile[pvi].peer_accesses;
    //   stats_.updates[pv] = tile[pvi].content_accesses;
    //   stats_.fills[pv] = tile[pvi].fills + tile[pvi].peer_fills;
    //   stats_.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar

    //   // FIXME: temporal reduction and network costs if hardware reduction isn't
    //   // supported appears to be wonky - network costs may need to trickle down
    //   // all the way to the level that has the reduction hardware.
    //   stats_.temporal_reductions[pv] = tile[pvi].content_accesses - tile[pvi].partition_size;
    //   std::cout << "stats: reads, updates, fills, address_generations " 
    //   << stats_.reads[pv] << " " << stats_.updates[pv]<< " " << stats_.fills[pv] << " " << stats_.address_generations[pv] <<std::endl;
    // }
    // else // Read-only data type.
    // {
    //   stats_.reads[pv] = tile[pvi].content_accesses + tile[pvi].peer_accesses;
    //   stats_.updates[pv] = 0;
    //   stats_.fills[pv] = tile[pvi].fills + tile[pvi].peer_fills;
    //   stats_.address_generations[pv] = stats_.reads[pv] + stats_.fills[pv]; // scalar
    //   stats_.temporal_reductions[pv] = 0;
    // }
    // original high-level actions
    stats_.reads[pv] = tile[pvi].reads;
    stats_.updates[pv] = tile[pvi].updates;
    stats_.fills[pv] = tile[pvi].fills;
    stats_.temporal_reductions[pv] = tile[pvi].temporal_reductions;
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv)) 
      stats_.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // FIXME? we want address generation be accounted for in energy/compound action?
    else
      stats_.address_generations[pv] = stats_.reads[pv] + stats_.fills[pv]; // FIXME? we want address generation be accounted for in energy/compound action?

    stats_.metadata_reads[pv] = tile[pvi].metadata_reads;
    stats_.metadata_fills[pv] = tile[pvi].metadata_fills;
    stats_.metadata_updates[pv] = tile[pvi].metadata_updates;

    // populate fine-grained scalar accesses
    // vector access computation is more involved, will be performed in ComputeVectorAccesses if mapping valid
    for(auto iter = tile[pvi].fine_grained_accesses.begin(); iter != tile[pvi].fine_grained_accesses.end(); ++iter)
    {
       stats_.fine_grained_scalar_accesses[pvi][iter->first] = iter->second;
    }

  }

  // compute the tile occupancy and (if applicable) confidence (considers compression and metadata overhead)
  ComputeTileOccupancyAndConfidence(tile, confidence_threshold);

  //
  // 2. Derive/validate architecture specs based on stats.
  //      
  auto total_utilized_capacity = std::accumulate(stats_.utilized_capacity.begin(),
                                                 stats_.utilized_capacity.end(),
                                                 0ULL);
  if (!specs_.size.IsSpecified())
  {
#ifdef UPDATE_UNSPECIFIED_SPECS
    specs_.size = std::ceil(total_utilized_capacity * specs_.multiple_buffering.Get());
#endif
  }
  else if (total_utilized_capacity > specs_.effective_size.Get() && !specs_.allow_overbooking.Get() )
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

  // check if tile confidence meets user-defined constraints
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    if (confidence_threshold > stats_.tile_confidence[pvi] ||
       (specs_.size.IsSpecified() && total_utilized_capacity > specs_.effective_size.Get() && specs_.allow_overbooking.Get())){
      success = false;
      fail_reason << "best tile confidence is less than constrained "
                  << "minimum tile confidence " << confidence_threshold;
    }
  }

  assert (specs_.block_size.IsSpecified());
    
  assert (specs_.cluster_size.IsSpecified());
   
  // Compute address-generation bits.
  if (specs_.size.IsSpecified())
  {
    double address_range = std::ceil(static_cast<double>(specs_.size.Get()) / specs_.block_size.Get());
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
  }
  else if (specs_.technology.Get() == Technology::SRAM)
  {
    // Use utilized capacity as proxy for size.
    double address_range = std::ceil(static_cast<double>(total_utilized_capacity) / specs_.block_size.Get());
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
  }
  else // DRAM.
  {
#ifdef FIXED_DRAM_SIZE_IF_UNSPECIFIED
    // DRAM of un-specified size, use 48-bit physical address.
    specs_.addr_gen_bits = 48;
#else
    // Use utilized capacity as proxy for size.
    double address_range = std::ceil(static_cast<double>(total_utilized_capacity) / specs_.block_size.Get());
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
#endif
  }
  if (!specs_.instances.IsSpecified())
  {
#ifdef UPDATE_UNSPECIFIED_SPECS
    specs_.instances = stats_.utilized_instances.Max();
#endif
  }
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


void BufferLevel::ComputeVectorAccesses(const tiling::CompoundDataMovementInfo& tile){

  // calculate fine-grained vector accesses
  auto block_size = specs_.block_size.Get();
  auto metadata_block_size = specs_.metadata_block_size.Get();

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++){
    double tile_mean_density = tile[pvi].tile_density->GetTileExpectedDensity(tile[pvi].size);

    // flag to signify choice of vector access calculations
    bool data_storage_naive = true;

    // post process child level size
    uint64_t tile_size = tile[pvi].size;

    // determine whether naive model is enough
    // naive calculation of vector accesses is applicable if
    //    (1) tile is uncompressed
    //    (2) tile is dense
    //    (3) vector width is 1
    //    (4) tile shape/vector width exceed certain threshold values (see model::VectorWidthCoefficientTable)
    if (tile[pvi].compressed && tile_size != 0 && block_size > 1 && tile_mean_density < 1.0){

      assert(block_size % 2 == 0);

      double lookup_density_idx = floor(tile_mean_density/0.1)-1; // 0.1 is at idx 0
      double vector_width_threshold = VectorWidthCoefficientTable.at(block_size).at(lookup_density_idx);

      if (vector_width_threshold > tile_size/block_size){
        data_storage_naive = false;
      }
    }

    // calculate the scaling factor based on the tile distribution
    double ratio = 1.0;
    if (!data_storage_naive){
      // #define VALIDATE_SCNN_TIMELOOP_LITE   // ENV VAR for performing SCNN validation on timeloop-lite
      #ifdef VALIDATE_SCNN_TIMELOOP_LITE
      std::uint64_t workload_tensor_size = tile[pvi].tile_density->GetWorkloadTensorSize();
      problem::Hypergeometric stats_model(workload_tensor_size, tile_mean_density);
      #endif

      double total = 0;
      // number of possible nonzero values can only be discrete
      for (std::uint64_t nnz = 1; nnz <= tile_size; nnz++){

        #ifdef VALIDATE_SCNN_TIMELOOP_LITE
        // SCNN validation on timeloop-lite (enforces hypergeometric modeling no regardless of the tile density model)
        double prob = stats_model->GetProbability(tile_size, nnz);
        #else
        double prob = tile[pvi].tile_density->GetProbability(tile_size, nnz);
        #endif

        total += ceil(double(nnz)/block_size) * prob;
      }
      double naive_total = tile_size * tile_mean_density/block_size;
      ratio = total/naive_total;
    }

    // adjust the sparse modeling traffic based on the calculated ratio
    // metadata accesses are scaled similarly as they are also dependent on the number of nonzero values in the tile
    for (auto iter = tile[pvi].fine_grained_accesses.begin(); iter != tile[pvi].fine_grained_accesses.end(); ++iter)
    {
       if (iter->first.find("count") == std::string::npos && iter->second != 0 && tile_size != 0) {

          bool metadata_action = (iter->first.find("metadata") != std::string::npos) ? true : false;

           double total_naive_accesses;
           if (!metadata_action) {
              total_naive_accesses = (iter->second % block_size == 0) ? iter->second / block_size : iter->second / block_size + 1;
           } else {
              total_naive_accesses = (iter->second % metadata_block_size == 0) ? iter->second / metadata_block_size : iter->second / metadata_block_size + 1;
           }

           stats_.fine_grained_vector_accesses[pvi][iter->first] = total_naive_accesses * ratio;

       } else {
          // decompression counts are not related to block size
          stats_.fine_grained_vector_accesses[pvi][iter->first] = iter->second;
       }
    }
  }
}

// Compute buffer energy.
void BufferLevel::ComputeBufferEnergy(const tiling::CompoundDataMovementInfo& data_movement_info) {
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++) {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // move all orginal number of vector access computation to the ComputeVectorAccesses function
    // prepare for speculation energy calculation
    stats_.parent_level_name[pvi] = "";
    stats_.parent_level_id[pvi] = data_movement_info[pvi].parent_level;
    if (data_movement_info[pvi].parent_level != std::numeric_limits<unsigned>::max()) {
      stats_.parent_level_name[pvi] = data_movement_info[pvi].parent_level_name;
    }

    // compute in terms of fine-grained action types
    std::string op_name;
    double cluster_access_energy = 0;
    for (unsigned op_id = 0; op_id < unsigned(tiling::GetNumOpTypes("storage")); op_id++) {
      op_name = tiling::storageOperationTypes[op_id];
      // directly fetch the populated vector access numbers instead of using explicit action names
      cluster_access_energy += stats_.fine_grained_vector_accesses[pv].at(op_name) * specs_.op_energy_map.at(op_name)
                               * stats_.tile_confidence[pvi];
    }
    stats_.cluster_access_energy[pv] = cluster_access_energy;
    stats_.cluster_access_energy_due_to_overflow[pv] = 0; // will be populated (if any) in the ComputeEnergyDueToChildLevelOverflow
    // per-instance energy will be finalized in the FinalizeBufferEnergy function
  }
}

void BufferLevel::ComputeEnergyDueToChildLevelOverflow(Stats child_level_stats, unsigned data_space_id){

  double cluster_access_energy_due_to_overflow = 0;

  for (unsigned op_id = 0; op_id < unsigned( tiling::GetNumOpTypes("storage")); op_id++) {
    std::string op_name = tiling::storageOperationTypes[op_id];

    // random reads (of data and metadata) can be read from parent level dependent on confidence
    // (skipped and gated do not need to be propagated to parent level)
    if (op_name.find("read") != std::string::npos && op_name.find("random") != std::string::npos) {
      // for random data read and metadata read actions
      cluster_access_energy_due_to_overflow += child_level_stats.fine_grained_vector_accesses[data_space_id].at(op_name)
                                         * specs_.op_energy_map.at(op_name)
                                         * (1 - child_level_stats.tile_confidence[data_space_id]);
    }
  }

  stats_.cluster_access_energy[data_space_id] += cluster_access_energy_due_to_overflow;
  stats_.cluster_access_energy_due_to_overflow[data_space_id] += cluster_access_energy_due_to_overflow;

}


void BufferLevel::FinalizeBufferEnergy() {

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++) {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto instance_accesses = stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv);
    double cluster_utilization = double(stats_.utilized_instances.at(pv)) /
                                 double(stats_.utilized_clusters.at(pv));

    // Spread out the cost between the utilized instances in each cluster
    if (stats_.utilized_instances.at(pvi) > 0) {
      stats_.energy[pv] = stats_.cluster_access_energy.at(pv) / cluster_utilization;
      stats_.energy_per_access[pv] = stats_.energy.at(pv) / instance_accesses;
      stats_.energy_due_to_overflow[pv] = stats_.cluster_access_energy_due_to_overflow.at(pv) / cluster_utilization;
    } else {
      stats_.energy[pv] = 0;
      stats_.energy_per_access[pv] = 0;
      stats_.energy_due_to_overflow[pv] = 0;
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
        pat::AdderEnergy(specs_.word_bits.Get(), network_update_->WordBits());
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
    if (specs_.addr_gen_energy.Get() < 0.0) { 
      stats_.addr_gen_energy[pv] = stats_.address_generations[pv] *
        pat::AdderEnergy(specs_.addr_gen_bits.Get(), specs_.addr_gen_bits.Get());
    }
    else
    {
       stats_.addr_gen_energy[pv] = stats_.address_generations[pv] * specs_.addr_gen_energy.Get();
    }
  }
}

//
// Compute performance.
//
void BufferLevel::ComputePerformance(const std::uint64_t compute_cycles)
{
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
    unconstrained_read_bandwidth[pv]  = (double(total_read_accesses)  / compute_cycles);
    unconstrained_write_bandwidth[pv] = (double(total_write_accesses) / compute_cycles);
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
#ifdef UPDATE_UNSPECIFIED_SPECS
  if (!specs_.read_bandwidth.IsSpecified())
    specs_.read_bandwidth = std::accumulate(stats_.read_bandwidth.begin(), stats_.read_bandwidth.end(), 0.0);
  if (!specs_.write_bandwidth.IsSpecified())
    specs_.write_bandwidth = std::accumulate(stats_.write_bandwidth.begin(), stats_.write_bandwidth.end(), 0.0);
#endif
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
STAT_ACCESSOR(std::uint64_t, BufferLevel, TileSize, stats_.tile_size.at(pv))
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
  area += specs_.storage_area.Get();
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

// flag to print verbose sparse stats or dense stats
// #define PRINT_SPARSE_STATS
#ifdef PRINT_SPARSE_STATS
  out << indent << indent << "Technology                   : " << specs.technology << std::endl;
  out << indent << indent << "Size                         : " << specs.size << std::endl;
  out << indent << indent << "Word bits                    : " << specs.word_bits << std::endl;
  out << indent << indent << "Block size                   : " << specs.block_size << std::endl;
  out << indent << indent << "Metadata word bits           : " << specs.metadata_word_bits << std::endl;
  out << indent << indent << "Metadata block size          : " << specs.metadata_block_size << std::endl;
  out << indent << indent << "Cluster size                 : " << specs.cluster_size << std::endl;
  out << indent << indent << "Instances                    : " << specs.instances << " ("
      << specs.meshX << "*" << specs.meshY << ")" << std::endl;
  out << indent << indent << "Read bandwidth               : " << specs.read_bandwidth << std::endl;    
  out << indent << indent << "Write bandwidth              : " << specs.write_bandwidth << std::endl;    
  out << indent << indent << "Multiple buffering           : " << specs.multiple_buffering << std::endl;
  out << indent << indent << "Allow overbooking            : " << specs.allow_overbooking << std::endl;
  out << indent << indent << "Effective size               : " << specs.effective_size << std::endl;
  out << indent << indent << "Min utilization              : " << specs.min_utilization << std::endl;
  out << indent << indent << "Vector access energy(max)    : " << specs.vector_access_energy << " pJ" << std::endl;
  out << indent << indent << "Vector gated read energy     : " << specs.op_energy_map.at("gated_read") << " pJ" << std::endl;
  out << indent << indent << "Vector skipped read energy   : " << specs.op_energy_map.at("skipped_read") << " pJ" << std::endl;
  out << indent << indent << "Vector random read energy    : " << specs.op_energy_map.at("random_read") << " pJ" << std::endl;
  out << indent << indent << "Vector gated write energy    : " << specs.op_energy_map.at("gated_fill") << " pJ" << std::endl;
  out << indent << indent << "Vector skipped write energy  : " << specs.op_energy_map.at("skipped_fill") << " pJ" << std::endl;
  out << indent << indent << "Vector random write energy   : " << specs.op_energy_map.at("random_fill") << " pJ" << std::endl;
//  out << indent << indent << "Vector gated update energy   : " << specs.op_energy_map.at("gated_update") << " pJ" << std::endl;
//  out << indent << indent << "Vector skipped update energy : " << specs.op_energy_map.at("skipped_update") << " pJ" << std::endl;
//  out << indent << indent << "Vector random update energy  : " << specs.op_energy_map.at("random_update") << " pJ" << std::endl;
  out << indent << indent << "Vector metadata read energy  : " << specs.op_energy_map.at("metadata_read") << " pJ" << std::endl;
  out << indent << indent << "Vector metadata write energy : " << specs.op_energy_map.at("metadata_fill") << " pJ" << std::endl;
  out << indent << indent << "(De)compression energy       : " << specs.op_energy_map.at("decompression_count") << " pJ" << std::endl;
  out << indent << indent << "Area                         : " << specs.storage_area << " um^2" << std::endl;

  out << std::endl;

#else
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
#endif

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

// flag to print verbose sparse stats or dense stats
//#define PRINT_SPARSE_STATS
#ifdef PRINT_SPARSE_STATS
      out << indent + indent << "Partition size                                        : " << stats.partition_size.at(pv) << std::endl;
      out << indent + indent << "Parent level name                                     : " << stats.parent_level_name.at(pv) << std::endl;
      out << indent + indent << "Overbooked proportion                                 : " << 1.0 - stats.tile_confidence.at(pv) << std::endl;
      out << indent + indent << "Max tile density                                      : " << stats.tile_max_density.at(pv) << std::endl;
      out << indent + indent << "Tile density distribution                             : " << stats.tile_density_distribution.at(pv) << std::endl;
      out << indent + indent << "Tile size                                             : " << stats.tile_size.at(pv) << std::endl;
      out << indent + indent << "Max total utilized capacity                           : " << stats.utilized_capacity.at(pv) << std::endl;
      out << indent + indent << "Utilized instances (max)                              : " << stats.utilized_instances.at(pv) << std::endl;
      out << indent + indent << "Utilized clusters (max)                               : " << stats.utilized_clusters.at(pv) << std::endl;
      out << indent + indent << "Max metadata tile size                                : " << stats.metadata_tile_size.at(pv) << std::endl;
      out << indent + indent << "Max metadata utilized capacity                        : " << int(ceil((double)stats.metadata_tile_size.at(pv) * specs_.metadata_word_bits.Get()/specs_.word_bits.Get())) << std::endl;
      out << indent + indent << "Total scalar reads (per-instance)                     : " << stats.reads.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar skipped reads (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("skipped_read") << std::endl;
      out << indent + indent + indent << "Scalar gated reads (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_read") << std::endl;
      out << indent + indent + indent << "Scalar random reads (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("random_read") << std::endl;
      out << indent + indent << "Total scalar fills (per-instance)                     : " << stats.fills.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar skipped fills (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("skipped_fill") << std::endl;
      out << indent + indent + indent << "Scalar gated fills (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_fill") << std::endl;
      out << indent + indent + indent << "Scalar random fills (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("random_fill") << std::endl;
      out << indent + indent << "Total scalar updates (per-instance)                   : " << stats.updates.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar skipped  updates (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("skipped_update") << std::endl;
      out << indent + indent + indent << "Scalar gated  updates (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_update") << std::endl;
      out << indent + indent + indent << "Scalar random  updates (per-instance): " << stats.fine_grained_scalar_accesses.at(pv).at("random_update") << std::endl;
      out << indent + indent << "Vector random reads (per-instance): " << stats.fine_grained_vector_accesses.at(pv).at("random_read") << std::endl;
      out << indent + indent << "Vector random fills (per-instance): " << stats.fine_grained_vector_accesses.at(pv).at("random_fill") << std::endl;
      out << indent + indent << "Vector random updates (per-instance): " << stats.fine_grained_vector_accesses.at(pv).at("random_update") << std::endl;
      out << indent + indent << "Temporal reductions (per-instance)                    : " << stats.temporal_reductions.at(pv) << std::endl;
      out << indent + indent << "Address generations (per-cluster)                     : " << stats.address_generations.at(pv) << std::endl;
      out << indent + indent << "Total scalar metadata reads (per-cluster)             : " << stats.metadata_reads.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar metadata random reads (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("metadata_read") << std::endl;
      out << indent + indent + indent << "Scalar metadata gated reads (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_metadata_read")<< std::endl;
      out << indent + indent << "Total scalar metadata fills (per-cluster)             : " << stats.metadata_fills.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar metadata random fills (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("metadata_fill") << std::endl;
      out << indent + indent + indent << "Scalar metadata gated fills (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_metadata_fill") << std::endl;
      out << indent + indent << "Total scalar metadata updates (per-cluster)           : " << stats.metadata_updates.at(pv) << std::endl;
      out << indent + indent + indent << "Scalar metadata random updates (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("metadata_update") << std::endl;
      out << indent + indent + indent << "Scalar metadata gated updates (per-cluster): " << stats.fine_grained_scalar_accesses.at(pv).at("gated_metadata_update") << std::endl;
      out << indent + indent << "Scalar decompression counts (per-cluster)             : " << stats.fine_grained_scalar_accesses.at(pv).at("decompression_count") << std::endl;
      out << indent + indent << "Scalar compression counts (per-cluster)               : " << stats.fine_grained_scalar_accesses.at(pv).at("compression_count") << std::endl;
      out << indent + indent << "Energy (per-scalar-access)                            : " << stats.energy_per_access.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Energy (per-instance)                                 : " << stats.energy.at(pv) << " pJ" << std::endl;
      out << indent + indent + indent << "Energy due to current level accesses (per-instance): "  << stats.energy.at(pv) - stats.energy_due_to_overflow.at(pv)<< std::endl;
      out << indent + indent + indent << "Energy due to child level overflow (per-instance): "  << stats.energy_due_to_overflow.at(pv)<< std::endl;
      out << indent + indent << "Energy (total)                                        : " << stats.energy.at(pv) * stats.utilized_instances.at(pv) << " pJ" << std::endl;
      out << indent + indent + indent << "Energy due to current level accesses (total): "  << stats.cluster_access_energy.at(pv) - stats.cluster_access_energy_due_to_overflow.at(pv)<< std::endl;
      out << indent + indent + indent << "Energy due to child level overflow (total): "  << stats.cluster_access_energy_due_to_overflow.at(pv)<< std::endl;
      out << indent + indent << "Temporal Reduction Energy (per-instance)              : " << stats.temporal_reduction_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Temporal Reduction Energy (total)                     : " << stats.temporal_reduction_energy.at(pv) * stats.utilized_instances.at(pv) << " pJ" << std::endl;

      // out << indent + indent << "Address Generation Energy (per-cluster)  : "
      //     << stats.addr_gen_energy.at(pv) << " pJ" << std::endl;
      // out << indent + indent << "Address Generation Energy (total)        : "
      //     << stats.addr_gen_energy.at(pv) * stats.utilized_clusters.at(pv)
      //     << " pJ" << std::endl;
      out << indent + indent << "Read Bandwidth (per-instance)                         : " << stats.read_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Read Bandwidth (total)                                : " << stats.read_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (per-instance)                        : " << stats.write_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (total)                               : " << stats.write_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;

#else
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
      out << indent + indent << "Read Bandwidth (per-instance)            : " << stats.read_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Read Bandwidth (total)                   : " << stats.read_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (per-instance)           : " << stats.write_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (total)                  : " << stats.write_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;
#endif
    }
  }

  out << std::endl;
}

}  // namespace model
