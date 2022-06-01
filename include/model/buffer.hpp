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

#pragma once

#include <iostream>
#include <boost/serialization/export.hpp>

#include "model/model-base.hpp"
#include "model/level.hpp"
#include "loop-analysis/tiling.hpp"
#include "mapping/nest.hpp"
#include "compound-config/compound-config.hpp"
#include "model/util.hpp"
#include "model/network.hpp"
#include "workload/density-models/density-distribution.hpp"

namespace model
{

  //
  // mean density - tile_size/vectorwidth table
  //

  // once the tile shape exceeds vec_width * threshold, no statistical modeling is needed for number of vector accesses
  // in that case, number of vector accesses = ceil(number of scalar access / block size)

  static std::map<unsigned, std::vector<double>>  VectorWidthCoefficientTable

    //                 threshold ratio for densities
    //vec_width    0.1, 0.2, 0.3, ...,                1.0)

    = {{2,        {251, 125, 84, 63, 51, 42, 36, 32, 28, 1}},
       {4,        {375, 188, 125, 94, 75, 63, 54, 47, 42, 1}},
       {8,        {438, 219, 146, 110, 88, 73, 63, 55, 49, 1}},
       {16,       {469, 235, 157, 118, 94, 79, 67, 59, 52, 1}},
       {32,       {485, 243, 162, 122, 97, 81, 70, 61, 52, 1}}
     };

//--------------------------------------------//
//                 BufferLevel                //
//--------------------------------------------//

class BufferLevel : public Level
{

  //
  // Types.
  //

 public:
  
  // Memory technology (FIXME: separate latch arrays).
  enum class Technology { SRAM, DRAM };
  friend std::ostream& operator<<(std::ostream& out, const Technology& tech);

  //
  // Specs.
  //
  struct Specs : public LevelSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;
    const std::string Type() const override { return "BufferLevel"; }
    
    Attribute<std::string> name;
    Attribute<Technology> technology;
    Attribute<std::uint64_t> size;
    Attribute<std::uint64_t> md_size;
    Attribute<std::uint64_t> md_size_bits;
    Attribute<std::uint64_t> word_bits;
    Attribute<std::uint64_t> addr_gen_bits;
    Attribute<std::uint64_t> block_size;
    Attribute<std::uint64_t> cluster_size;
    Attribute<std::uint64_t> instances;    
    Attribute<std::uint64_t> meshX;
    Attribute<std::uint64_t> meshY;
    Attribute<double> shared_bandwidth;
    Attribute<double> read_bandwidth;
    Attribute<double> write_bandwidth;
    Attribute<double> multiple_buffering;
    Attribute<std::uint64_t> effective_size;
    Attribute<std::uint64_t> effective_md_size;
    Attribute<std::uint64_t> effective_md_size_bits;
    Attribute<double> min_utilization;
    Attribute<std::uint64_t> num_ports;
    Attribute<std::uint64_t> num_banks;
    Attribute<bool> reduction_supported;

    // compression related
    Attribute<bool> concordant_compressed_tile_traversal;
    Attribute<bool> tile_partition_supported;
    Attribute<bool> decompression_supported;
    Attribute<bool> compression_supported;

    // metadata storage related
    Attribute<std::uint64_t> metadata_storage_width;
    Attribute<std::uint64_t> metadata_storage_depth;

    Attribute<bool> unified_data_md_storage;

    Attribute<std::uint64_t> default_md_block_size;
    Attribute<std::uint64_t> default_md_word_bits;

    Attribute<std::string> read_network_name;
    Attribute<std::string> fill_network_name;
    Attribute<std::string> drain_network_name;
    Attribute<std::string> update_network_name;    

    // for ERT parsing
    std::map<std::string, double> ERT_entries;
    std::map<std::string, double> op_energy_map;

    // for overflow evaluation
    Attribute<bool> allow_overbooking;

    // Physical Attributes (derived from technology model).
    // FIXME: move into separate struct?
    Attribute<double> vector_access_energy; // pJ
    Attribute<double> storage_area; // um^2
    Attribute<double> addr_gen_energy; // pJ

    Attribute<bool> is_sparse_module;
    
    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(LevelSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(name);
        ar& BOOST_SERIALIZATION_NVP(technology);
        ar& BOOST_SERIALIZATION_NVP(size);
        ar& BOOST_SERIALIZATION_NVP(word_bits);
        ar& BOOST_SERIALIZATION_NVP(addr_gen_bits);
        ar& BOOST_SERIALIZATION_NVP(block_size);
        ar& BOOST_SERIALIZATION_NVP(cluster_size);
        ar& BOOST_SERIALIZATION_NVP(instances);    
        ar& BOOST_SERIALIZATION_NVP(meshX);
        ar& BOOST_SERIALIZATION_NVP(meshY);
        ar& BOOST_SERIALIZATION_NVP(shared_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(multiple_buffering);
        ar& BOOST_SERIALIZATION_NVP(min_utilization);
        ar& BOOST_SERIALIZATION_NVP(num_ports);
        ar& BOOST_SERIALIZATION_NVP(num_banks);

        ar& BOOST_SERIALIZATION_NVP(read_network_name);
        ar& BOOST_SERIALIZATION_NVP(fill_network_name);
        ar& BOOST_SERIALIZATION_NVP(drain_network_name);
        ar& BOOST_SERIALIZATION_NVP(update_network_name);
      }
    }

   public:
    std::shared_ptr<LevelSpecs> Clone() const override
    {
      return std::static_pointer_cast<LevelSpecs>(std::make_shared<Specs>(*this));
    }

    void UpdateOpEnergyViaERT(const std::map<std::string, double>& ERT_entries, const double max_energy) override;
    void UpdateAreaViaART(const double component_area) override;
  };
  
  //
  // Stats.
  //
  struct Stats
  {
    problem::PerDataSpace<bool> keep;
    problem::PerDataSpace<std::uint64_t> partition_size;
    problem::PerDataSpace<std::uint64_t> utilized_capacity;
    problem::PerDataSpace<std::uint64_t> utilized_md_capacity_bits;
    problem::PerDataSpace<std::uint64_t> tile_size;
    problem::PerDataSpace<double> utilized_instances;
    problem::PerDataSpace<std::uint64_t> utilized_x_expansion;
    problem::PerDataSpace<std::uint64_t> utilized_y_expansion;
    problem::PerDataSpace<std::uint64_t> utilized_clusters;
<<<<<<< HEAD
    problem::PerDataSpace<double> reads;
    problem::PerDataSpace<double> updates;
    problem::PerDataSpace<double> fills;
    problem::PerDataSpace<double> address_generations;
    problem::PerDataSpace<double> temporal_reductions;
=======
    problem::PerDataSpace<std::uint64_t> reads;
    problem::PerDataSpace<std::uint64_t> updates;
    problem::PerDataSpace<std::uint64_t> fills;
    problem::PerDataSpace<std::uint64_t> address_generations;
    problem::PerDataSpace<std::uint64_t> temporal_reductions;
    problem::PerDataSpace<double> shared_bandwidth;
>>>>>>> 8b4ff49a370bdc251d2c16bceadeba5739644cd5
    problem::PerDataSpace<double> read_bandwidth;
    problem::PerDataSpace<double> write_bandwidth;
    problem::PerDataSpace<double> energy_per_algorithmic_access;
    problem::PerDataSpace<double> energy_per_access;
    problem::PerDataSpace<double> energy;
    problem::PerDataSpace<double> temporal_reduction_energy;
    problem::PerDataSpace<double> addr_gen_energy;
    problem::PerDataSpace<double> cluster_access_energy;
    problem::PerDataSpace<double> cluster_access_energy_due_to_overflow;
    problem::PerDataSpace<double> energy_due_to_overflow;

    problem::PerDataSpace<std::uint64_t> tile_shape;
    problem::PerDataSpace<std::uint64_t> data_tile_size;
    problem::PerDataSpace<bool> compressed;
    //problem::PerDataSpace<std::uint64_t> metadata_tile_size;
    problem::PerDataSpace<std::vector<std::vector<std::uint64_t>>> metadata_tile_size;
    
    problem::PerDataSpace<std::uint64_t> metadata_tile_size_bits;
    problem::PerDataSpace<std::string> metadata_format;
    problem::PerDataSpace<double> tile_confidence;
    problem::PerDataSpace<std::string> parent_level_name;
    problem::PerDataSpace<unsigned> parent_level_id;
    problem::PerDataSpace<std::string> tile_density_distribution;
    problem::PerDataSpace<double> format_shared_bandwidth_ratio;
    problem::PerDataSpace<double> format_read_bandwidth_ratio;
    problem::PerDataSpace<double> format_write_bandwidth_ratio;
    
    // fine-grained action stats
<<<<<<< HEAD
    problem::PerDataSpace<std::map<std::string, double>> fine_grained_scalar_accesses;
    problem::PerDataSpace<std::map<std::string, double>> fine_grained_vector_accesses;
    problem::PerDataSpace<double> gated_reads;
    problem::PerDataSpace<double> skipped_reads;
    problem::PerDataSpace<double> random_reads;

    problem::PerDataSpace<double> gated_fills;
    problem::PerDataSpace<double> skipped_fills;
    problem::PerDataSpace<double> random_fills;

    problem::PerDataSpace<double> gated_updates;
    problem::PerDataSpace<double> skipped_updates;
    problem::PerDataSpace<double> random_updates;

    problem::PerDataSpace<double> metadata_reads;
    problem::PerDataSpace<double> random_metadata_reads;
    problem::PerDataSpace<double> gated_metadata_reads;
    problem::PerDataSpace<double> skipped_metadata_reads;
    problem::PerDataSpace<double> metadata_fills;
    problem::PerDataSpace<double> random_metadata_fills;
    problem::PerDataSpace<double> gated_metadata_fills;
    problem::PerDataSpace<double> skipped_metadata_fills;
    problem::PerDataSpace<double> metadata_updates;
    problem::PerDataSpace<double> random_metadata_updates;
    problem::PerDataSpace<double> gated_metadata_updates;
    problem::PerDataSpace<double> skipped_metadata_updates;

    problem::PerDataSpace<double> decompression_counts;
    problem::PerDataSpace<double> compression_counts;
=======
    problem::PerDataSpace<std::map<std::string, std::uint64_t>> fine_grained_scalar_accesses;
    problem::PerDataSpace<std::map<std::string, tiling::PerTileFormatAccesses>> fine_grained_format_scalar_accesses;
    problem::PerDataSpace<std::map<std::string, double>> fine_grained_vector_accesses;
    problem::PerDataSpace<std::map<std::string, std::uint64_t>> fine_grained_fromat_accesses_bits;


    problem::PerDataSpace<std::uint64_t> gated_reads;
    problem::PerDataSpace<std::uint64_t> skipped_reads;
    problem::PerDataSpace<std::uint64_t> random_reads;

    problem::PerDataSpace<std::uint64_t> gated_fills;
    problem::PerDataSpace<std::uint64_t> skipped_fills;
    problem::PerDataSpace<std::uint64_t> random_fills;

    problem::PerDataSpace<std::uint64_t> gated_updates;
    problem::PerDataSpace<std::uint64_t> skipped_updates;
    problem::PerDataSpace<std::uint64_t> random_updates;

    problem::PerDataSpace<tiling::PerTileFormatAccesses> random_format_reads;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> skipped_format_reads;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> gated_format_reads;

    problem::PerDataSpace<tiling::PerTileFormatAccesses> random_format_fills;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> skipped_format_fills;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> gated_format_fills;
 
    problem::PerDataSpace<tiling::PerTileFormatAccesses> random_format_updates;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> skipped_format_updates;
    problem::PerDataSpace<tiling::PerTileFormatAccesses> gated_format_updates;
       
    //problem::PerDataSpace<std::uint64_t> metadata_reads;
    //problem::PerDataSpace<std::uint64_t> random_metadata_reads;
    //problem::PerDataSpace<std::uint64_t> gated_metadata_reads;
    //problem::PerDataSpace<std::uint64_t> skipped_metadata_reads;
    //problem::PerDataSpace<std::uint64_t> metadata_fills;
    //problem::PerDataSpace<std::uint64_t> random_metadata_fills;
    //problem::PerDataSpace<std::uint64_t> gated_metadata_fills;
    //problem::PerDataSpace<std::uint64_t> skipped_metadata_fills;
    //problem::PerDataSpace<std::uint64_t> metadata_updates;
    //problem::PerDataSpace<std::uint64_t> random_metadata_updates;
    //problem::PerDataSpace<std::uint64_t> gated_metadata_updates;
    //problem::PerDataSpace<std::uint64_t> skipped_metadata_updates;

    problem::PerDataSpace<std::uint64_t> decompression_counts;
    problem::PerDataSpace<std::uint64_t> compression_counts;
>>>>>>> 8b4ff49a370bdc251d2c16bceadeba5739644cd5

    std::uint64_t cycles;
    double slowdown;

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(keep);
        ar& BOOST_SERIALIZATION_NVP(partition_size);
        ar& BOOST_SERIALIZATION_NVP(utilized_capacity);
        ar& BOOST_SERIALIZATION_NVP(utilized_instances);
        ar& BOOST_SERIALIZATION_NVP(utilized_clusters);
        ar& BOOST_SERIALIZATION_NVP(reads);
        ar& BOOST_SERIALIZATION_NVP(updates);
        ar& BOOST_SERIALIZATION_NVP(fills);
        
        // fine grained accesses
        ar& BOOST_SERIALIZATION_NVP(gated_reads);
        ar& BOOST_SERIALIZATION_NVP(skipped_reads);
        ar& BOOST_SERIALIZATION_NVP(random_reads);

        ar& BOOST_SERIALIZATION_NVP(gated_fills);
        ar& BOOST_SERIALIZATION_NVP(skipped_fills);
        ar& BOOST_SERIALIZATION_NVP(random_fills);

        ar& BOOST_SERIALIZATION_NVP(gated_updates);
        ar& BOOST_SERIALIZATION_NVP(skipped_updates);
        ar& BOOST_SERIALIZATION_NVP(random_updates);
       
        ar& BOOST_SERIALIZATION_NVP(random_format_reads);
        ar& BOOST_SERIALIZATION_NVP(gated_format_reads);
        ar& BOOST_SERIALIZATION_NVP(skipped_format_reads);

        ar& BOOST_SERIALIZATION_NVP(random_format_fills);
        ar& BOOST_SERIALIZATION_NVP(gated_format_fills);
        ar& BOOST_SERIALIZATION_NVP(skipped_format_fills);

        ar& BOOST_SERIALIZATION_NVP(random_format_updates);
        ar& BOOST_SERIALIZATION_NVP(gated_format_updates);
        ar& BOOST_SERIALIZATION_NVP(skipped_format_updates);
        
        ar& BOOST_SERIALIZATION_NVP(decompression_counts);
        ar& BOOST_SERIALIZATION_NVP(compression_counts);

        ar& BOOST_SERIALIZATION_NVP(address_generations);
        ar& BOOST_SERIALIZATION_NVP(temporal_reductions);
        ar& BOOST_SERIALIZATION_NVP(shared_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(energy_per_algorithmic_access);
        ar& BOOST_SERIALIZATION_NVP(energy_per_access);
        ar& BOOST_SERIALIZATION_NVP(energy);
        ar& BOOST_SERIALIZATION_NVP(temporal_reduction_energy);
        ar& BOOST_SERIALIZATION_NVP(addr_gen_energy);
        ar& BOOST_SERIALIZATION_NVP(cycles);
        ar& BOOST_SERIALIZATION_NVP(slowdown);
        ar& BOOST_SERIALIZATION_NVP(format_shared_bandwidth_ratio);
        ar& BOOST_SERIALIZATION_NVP(format_read_bandwidth_ratio);
        ar& BOOST_SERIALIZATION_NVP(format_write_bandwidth_ratio);
      }
    }
  };

  //
  // Data
  //
  
 private:

  std::vector<loop::Descriptor> subnest_;
  Stats stats_;
  Specs specs_;

  bool populate_energy_per_op = false;

  // Network endpoints.
  std::shared_ptr<Network> network_read_;
  std::shared_ptr<Network> network_fill_;
  std::shared_ptr<Network> network_update_;
  std::shared_ptr<Network> network_drain_;

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(Level);
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(subnest_);
      ar& BOOST_SERIALIZATION_NVP(specs_);
      ar& BOOST_SERIALIZATION_NVP(stats_);
    }
  }

  //
  // Private helpers.
  //

 private:
  EvalStatus ComputeScalarAccesses(const tiling::CompoundDataMovementInfo& tile, const tiling::CompoundMask& mask,
                                   const double confidence_threshold,
                                   const bool break_on_failure);
  void ComputeVectorAccesses(const tiling::CompoundDataMovementInfo& tile);
  void ComputeTileOccupancyAndConfidence(const tiling::CompoundDataMovementInfo& tile, const double confidence_threshold);
  std::uint64_t ComputeMetaDataTileSizeInBits (const tiling::MetaDataTileOccupancy metadata_occupancy) const;
  std::uint64_t ComputeMetaDataTileSize(const tiling::MetaDataTileOccupancy metadata_occupancy) const;
  void ComputePerformance(const std::uint64_t compute_cycles);
  // void ComputeBufferEnergy();
  void ComputeBufferEnergy(const tiling::CompoundDataMovementInfo& data_movement_info);
  void ComputeReductionEnergy();
  void ComputeAddrGenEnergy();

  double StorageEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double TemporalReductionEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double AddrGenEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;

  //
  // API
  //

 public:
  BufferLevel();
  BufferLevel(const Specs & specs);
  ~BufferLevel();

  std::shared_ptr<Level> Clone() const override
  {
    return std::static_pointer_cast<Level>(std::make_shared<BufferLevel>(*this));
  }

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the constructor when an object is actually created.
  static Specs ParseSpecs(config::CompoundConfigNode setting, std::uint64_t n_elements, bool is_sparse_module);
  static void ParseBufferSpecs(config::CompoundConfigNode buffer, std::uint64_t n_elements,
                               problem::Shape::DataSpaceID pv, Specs& specs);
  static void ValidateTopology(BufferLevel::Specs& specs);

  void PopulateEnergyPerOp(unsigned num_ops);

  Specs& GetSpecs() { return specs_; }
  Stats& GetStats() { return stats_;}
  
  bool HardwareReductionSupported() override;

  // Connect to networks.
  void ConnectRead(std::shared_ptr<Network> network);
  void ConnectFill(std::shared_ptr<Network> network);
  void ConnectUpdate(std::shared_ptr<Network> network);
  void ConnectDrain(std::shared_ptr<Network> network);
  std::shared_ptr<Network> GetReadNetwork() { return network_read_; }
  std::shared_ptr<Network> GetUpdateNetwork() { return network_update_; }
 
  // Evaluation functions.
  EvalStatus PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                                const tiling::CompoundMask mask,
                                const problem::Workload* workload,
                                const sparse::PerStorageLevelCompressionInfo per_level_compression_info,
                                const double confidence_threshold,
                                const bool break_on_failure) override;
  EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                      const double confidence_threshold, const std::uint64_t compute_cycles,
                      const bool break_on_failure) override;

  // Energy calculation functions that are externally accessed in topology.cpp
  void ComputeEnergyDueToChildLevelOverflow(Stats child_level_stats, unsigned data_space_id);
  void FinalizeBufferEnergy();

  // Accessors (post-evaluation).
  
  double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
 
  std::string Name() const override;
  double Area() const override;
  double AreaPerInstance() const override;
  double Size() const;
  std::uint64_t Cycles() const override;
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double CapacityUtilization() const override;
  std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  std::uint64_t TileSize(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  std::uint64_t UtilizedInstances(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  
  // Printers.
  void Print(std::ostream& out) const;
  friend std::ostream& operator << (std::ostream& out, const BufferLevel& buffer_level);
};

}  // namespace model
