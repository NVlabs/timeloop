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

#pragma once

#include <iostream>
#include <libconfig.h++>
#include <boost/serialization/export.hpp>

#include "model/model-base.hpp"
#include "model/level.hpp"
#include "loop-analysis/tiling.hpp"
#include "mapping/nest.hpp"

namespace model
{

// A special-purpose class with a std::map-like interface used to hold
// *either* a collection of values of type T, one for each data space,
// *or* a single value of type T accessed with the key DataSpaceID::Num.
template<class T>
class PerDataSpaceOrShared
{
 private:
  problem::PerDataSpace<T> per_data_space;
  T shared;
  bool is_per_data_space = false;
  bool is_shared = false;

 public:
  PerDataSpaceOrShared()
  {
  }

  void SetPerDataSpace() {
    assert(!is_shared);
    is_per_data_space = true;
    // Construct a separate T value for each element.
    for (T& val : per_data_space)
    {
      val = T();
    }
  }

  void SetShared(T val=T()) {
    assert(!is_per_data_space);
    is_shared = true;
    shared = val;
  }

  T & operator [] (problem::Shape::DataSpaceID pv)
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  T & at(problem::Shape::DataSpaceID pv)
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  const T & at(problem::Shape::DataSpaceID pv) const
  {
    if (pv == problem::GetShape()->NumDataSpaces)
    {
      assert(is_shared);
      return shared;
    }
    else
    {
      assert(pv < problem::GetShape()->NumDataSpaces);
      assert(is_per_data_space);
      return per_data_space[pv];
    }
  }

  T Max() const
  {
    if (is_shared)
    {
      return shared;
    }
    else
    {
      assert(is_per_data_space);
      return per_data_space.Max();
    }
  }

  friend std::ostream& operator << (std::ostream& out, const PerDataSpaceOrShared<T>& x)
  {
    if (x.is_per_data_space)
    {
      out << "PerDataSpace:" << std::endl;
      out << x.per_data_space;
    }
    else
    {
      assert(x.is_shared);
      out << "Shared: " << x.shared << std::endl;
    }
    return out;
  }

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      if (is_per_data_space)
      {
        ar& BOOST_SERIALIZATION_NVP(per_data_space);
      }
      else
      {
        assert(is_shared);
        ar& BOOST_SERIALIZATION_NVP(shared);
      }
    }
  }

};

template<class T>
std::ostream& operator<<(std::ostream& out, const PerDataSpaceOrShared<T>& px);

//--------------------------------------------//
//                 BufferLevel                //
//--------------------------------------------//

class BufferLevel : public Level
{
 public:
  
  // DataSpaceID sharing.
  enum class DataSpaceIDSharing
  {
    Partitioned,
    Shared
  };

  // Memory technology (FIXME: separate latch arrays).
  enum class Technology { SRAM, DRAM };
  friend std::ostream& operator<<(std::ostream& out, const Technology& tech);

  // Helper iterator.
  template<typename Functor>
  static void ForEachDataSpaceID(Functor functor, DataSpaceIDSharing sharing)
  {
    unsigned start_pvi, end_pvi;
    if (sharing == DataSpaceIDSharing::Shared)
    {
      start_pvi = end_pvi = unsigned(problem::GetShape()->NumDataSpaces);
    }
    else
    {
      start_pvi = 0;
      end_pvi = unsigned(problem::GetShape()->NumDataSpaces)-1;
    }

    for (unsigned pvi = start_pvi; pvi <= end_pvi; pvi++)
    {
      functor(problem::Shape::DataSpaceID(pvi));
    }
  }

  //
  // Specs.
  //
  struct Specs : public LevelSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;
    const std::string Type() const override { return "BufferLevel"; }
    
    DataSpaceIDSharing sharing_type;

    PerDataSpaceOrShared<Attribute<std::string>> name;
    PerDataSpaceOrShared<Attribute<Technology>> technology;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> size;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> word_bits;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> addr_gen_bits;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> block_size;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> cluster_size;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> instances;    
    PerDataSpaceOrShared<Attribute<std::uint64_t>> meshX;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> meshY;
    // PerDataSpaceOrShared<Attribute<double>> bandwidth;
    PerDataSpaceOrShared<Attribute<double>> read_bandwidth;
    PerDataSpaceOrShared<Attribute<double>> write_bandwidth;
    PerDataSpaceOrShared<Attribute<double>> multiple_buffering;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> effective_size;
    PerDataSpaceOrShared<Attribute<double>> min_utilization;
    // Sophia
    PerDataSpaceOrShared<Attribute<double>> vector_access_energy;
    PerDataSpaceOrShared<Attribute<double>> storage_area;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> num_ports;
    PerDataSpaceOrShared<Attribute<std::uint64_t>> num_banks;

    struct Network
    {
      enum class Type
      {
        OneToOne,
        OneToMany,
        ManyToMany
      };
      PerDataSpaceOrShared<Attribute<Type>> type;
      PerDataSpaceOrShared<Attribute<std::uint64_t>> word_bits;
      PerDataSpaceOrShared<Attribute<std::uint64_t>> fanout;
      PerDataSpaceOrShared<Attribute<std::uint64_t>> fanoutX;
      PerDataSpaceOrShared<Attribute<std::uint64_t>> fanoutY;
      PerDataSpaceOrShared<Attribute<double>> routerEnergy;

      // Serialization
      friend class boost::serialization::access;

      template <class Archive>
      void serialize(Archive& ar, const unsigned int version = 0)
      {
        if (version == 0)
        {
          ar& BOOST_SERIALIZATION_NVP(type);
          ar& BOOST_SERIALIZATION_NVP(word_bits);
          ar& BOOST_SERIALIZATION_NVP(fanout);
          ar& BOOST_SERIALIZATION_NVP(fanoutX);
          ar& BOOST_SERIALIZATION_NVP(fanoutY);
          ar& BOOST_SERIALIZATION_NVP(routerEnergy);
        }
      }
    } network;

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(LevelSpecs);
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
        // ar& BOOST_SERIALIZATION_NVP(bandwidth);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(multiple_buffering);
        ar& BOOST_SERIALIZATION_NVP(network);
        ar& BOOST_SERIALIZATION_NVP(min_utilization);
        ar& BOOST_SERIALIZATION_NVP(vector_access_energy);
        ar& BOOST_SERIALIZATION_NVP(storage_area);
        ar& BOOST_SERIALIZATION_NVP(num_ports);
        ar& BOOST_SERIALIZATION_NVP(num_banks);
      }
    }
    
    Specs() :
        sharing_type(DataSpaceIDSharing::Shared)
    {
      Init();
    }
    
    Specs(DataSpaceIDSharing sharing) :
        sharing_type(sharing)
    {
      Init();
    }
    
    void Init()
    {
      // Initialize all parameters to "unspecified" default.
      if (sharing_type == DataSpaceIDSharing::Partitioned)
      {
        name.SetPerDataSpace();
        technology.SetPerDataSpace();
        size.SetPerDataSpace();
        word_bits.SetPerDataSpace();
        addr_gen_bits.SetPerDataSpace();
        block_size.SetPerDataSpace();
        cluster_size.SetPerDataSpace();
        instances.SetPerDataSpace();
        meshX.SetPerDataSpace();
        meshY.SetPerDataSpace();
        // bandwidth.SetPerDataSpace();
        read_bandwidth.SetPerDataSpace();
        write_bandwidth.SetPerDataSpace();
        multiple_buffering.SetPerDataSpace();
        effective_size.SetPerDataSpace();
        min_utilization.SetPerDataSpace();
        vector_access_energy.SetPerDataSpace();
        storage_area.SetPerDataSpace();
        num_ports.SetPerDataSpace();
        num_banks.SetPerDataSpace();

        network.type.SetPerDataSpace();
        network.word_bits.SetPerDataSpace();
        network.fanout.SetPerDataSpace();
        network.fanoutX.SetPerDataSpace();
        network.fanoutY.SetPerDataSpace();
        network.routerEnergy.SetPerDataSpace();
      }
      else // sharing_type == DataSpaceIDSharing::Shared
      {
        name.SetShared();
        technology.SetShared();
        size.SetShared();
        word_bits.SetShared();
        addr_gen_bits.SetShared();
        block_size.SetShared();
        cluster_size.SetShared();
        instances.SetShared();
        meshX.SetShared();
        meshY.SetShared();
        // bandwidth.SetShared();
        read_bandwidth.SetShared();
        write_bandwidth.SetShared();
        multiple_buffering.SetShared();
        effective_size.SetShared();
        min_utilization.SetShared();
        vector_access_energy.SetShared();
        storage_area.SetShared();
        num_ports.SetShared();
        num_banks.SetShared();

        network.type.SetShared();
        network.word_bits.SetShared();
        network.fanout.SetShared();
        network.fanoutX.SetShared();
        network.fanoutY.SetShared();
        network.routerEnergy.SetShared();
      }
    
    }

    // ----- Macro to add Accessors -----
#define ADD_ACCESSORS(FuncName, MemberName, Type)                          \
    Attribute<Type> & FuncName(problem::Shape::DataSpaceID pv)             \
    {                                                                      \
      return (sharing_type == DataSpaceIDSharing::Partitioned)             \
             ? MemberName[pv]                                              \
             : MemberName[problem::GetShape()->NumDataSpaces];             \
    }                                                                      \
                                                                           \
    const Attribute<Type> & FuncName(problem::Shape::DataSpaceID pv) const \
    {                                                                      \
      return (sharing_type == DataSpaceIDSharing::Partitioned)             \
             ? MemberName.at(pv)                                           \
             : MemberName.at(problem::GetShape()->NumDataSpaces);          \
    }                                                                      \
                                                                           \
    Attribute<Type> & FuncName()                                           \
    {                                                                      \
      assert(sharing_type == DataSpaceIDSharing::Shared);                  \
      return MemberName[problem::GetShape()->NumDataSpaces];               \
    }                                                                      \
                                                                           \
    const Attribute<Type> & FuncName() const                               \
    {                                                                      \
      assert(sharing_type == DataSpaceIDSharing::Shared);                  \
      return MemberName.at(problem::GetShape()->NumDataSpaces);            \
    }                                                               
    // ----- End Macro -----

    DataSpaceIDSharing SharingType() const { return sharing_type; }

    unsigned DataSpaceIDIteratorStart() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? unsigned(problem::GetShape()->NumDataSpaces)
                             : 0;
    }

    unsigned DataSpaceIDIteratorEnd() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? unsigned(problem::GetShape()->NumDataSpaces) + 1
                             : unsigned(problem::GetShape()->NumDataSpaces);
    }

    size_t NumPartitions() const
    {
      return sharing_type == DataSpaceIDSharing::Shared
                             ? 1
                             : size_t(problem::GetShape()->NumDataSpaces);
    }

    ADD_ACCESSORS(Name, name, std::string)
    ADD_ACCESSORS(Tech, technology, Technology)
    ADD_ACCESSORS(Size, size, std::uint64_t)
    ADD_ACCESSORS(WordBits, word_bits, std::uint64_t)
    ADD_ACCESSORS(AddrGenBits, addr_gen_bits, std::uint64_t)
    ADD_ACCESSORS(BlockSize, block_size, std::uint64_t)
    ADD_ACCESSORS(ClusterSize, cluster_size, std::uint64_t)
    ADD_ACCESSORS(Instances, instances, std::uint64_t)
    ADD_ACCESSORS(MeshX, meshX, std::uint64_t)
    ADD_ACCESSORS(MeshY, meshY, std::uint64_t)
    // ADD_ACCESSORS(Bandwidth, bandwidth, double)    
    ADD_ACCESSORS(ReadBandwidth, read_bandwidth, double)    
    ADD_ACCESSORS(WriteBandwidth, write_bandwidth, double)    
    ADD_ACCESSORS(MultipleBuffering, multiple_buffering, double)
    ADD_ACCESSORS(EffectiveSize, effective_size, std::uint64_t)
    ADD_ACCESSORS(MinUtilization, min_utilization, double)
    ADD_ACCESSORS(VectorAccessEnergy, vector_access_energy, double)
    ADD_ACCESSORS(StorageArea, storage_area, double)
    ADD_ACCESSORS(NumPorts, num_ports, std::uint64_t)
    ADD_ACCESSORS(NumBanks, num_banks, std::uint64_t)

    ADD_ACCESSORS(NetworkType, network.type, Network::Type)
    ADD_ACCESSORS(NetworkWordBits, network.word_bits, std::uint64_t)
    ADD_ACCESSORS(Fanout, network.fanout, std::uint64_t)
    ADD_ACCESSORS(FanoutX, network.fanoutX, std::uint64_t)
    ADD_ACCESSORS(FanoutY, network.fanoutY, std::uint64_t)
    ADD_ACCESSORS(RouterEnergy, network.routerEnergy, double)
  };
  
  struct Stats
  {
    problem::PerDataSpace<bool> keep;
    problem::PerDataSpace<std::uint64_t> partition_size;
    problem::PerDataSpace<std::uint64_t> utilized_capacity;
    problem::PerDataSpace<std::uint64_t> utilized_instances;
    problem::PerDataSpace<std::uint64_t> utilized_clusters;
    problem::PerDataSpace<unsigned long> reads;
    problem::PerDataSpace<unsigned long> updates;
    problem::PerDataSpace<unsigned long> fills;
    problem::PerDataSpace<unsigned long> address_generations;
    problem::PerDataSpace<unsigned long> temporal_reductions;
    // problem::PerDataSpace<double> bandwidth;
    problem::PerDataSpace<double> read_bandwidth;
    problem::PerDataSpace<double> write_bandwidth;
    problem::PerDataSpace<double> energy_per_access;
    problem::PerDataSpace<double> energy;
    problem::PerDataSpace<double> temporal_reduction_energy;
    problem::PerDataSpace<double> addr_gen_energy;
    PerDataSpaceOrShared<double> area;
    std::uint64_t cycles;
    double slowdown;

    struct Network
    {
      problem::PerDataSpace<std::uint64_t> fanout;
      problem::PerDataSpace<std::uint64_t> distributed_fanout;
      problem::PerDataSpace<std::uint64_t> multicast_factor;
      problem::PerDataSpace<std::vector<unsigned long>> ingresses;
      problem::PerDataSpace<bool> distributed_multicast;
      problem::PerDataSpace<unsigned long> link_transfers;
      problem::PerDataSpace<unsigned long> spatial_reductions;
      problem::PerDataSpace<double> link_transfer_energy;
      problem::PerDataSpace<double> num_hops;
      problem::PerDataSpace<std::vector<double>> avg_hops;
      problem::PerDataSpace<double> energy_per_hop;
      problem::PerDataSpace<double> energy;
      problem::PerDataSpace<double> spatial_reduction_energy;

      // Serialization
      friend class boost::serialization::access;
      
      template <class Archive>
      void serialize(Archive& ar, const unsigned int version = 0)
      {
        if (version == 0)
        {
          ar& BOOST_SERIALIZATION_NVP(fanout);
          ar& BOOST_SERIALIZATION_NVP(distributed_fanout);
          ar& BOOST_SERIALIZATION_NVP(multicast_factor);
          ar& BOOST_SERIALIZATION_NVP(ingresses);
          ar& BOOST_SERIALIZATION_NVP(distributed_multicast);
          ar& BOOST_SERIALIZATION_NVP(link_transfers);
          ar& BOOST_SERIALIZATION_NVP(spatial_reductions);
          ar& BOOST_SERIALIZATION_NVP(link_transfer_energy);
          ar& BOOST_SERIALIZATION_NVP(num_hops);
          ar& BOOST_SERIALIZATION_NVP(energy_per_hop);
          ar& BOOST_SERIALIZATION_NVP(energy);
          ar& BOOST_SERIALIZATION_NVP(spatial_reduction_energy);
        }
      }      
    } network;

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
        ar& BOOST_SERIALIZATION_NVP(address_generations);
        ar& BOOST_SERIALIZATION_NVP(temporal_reductions);
        // ar& BOOST_SERIALIZATION_NVP(bandwidth);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(energy_per_access);
        ar& BOOST_SERIALIZATION_NVP(energy);
        ar& BOOST_SERIALIZATION_NVP(temporal_reduction_energy);
        ar& BOOST_SERIALIZATION_NVP(addr_gen_energy);
        ar& BOOST_SERIALIZATION_NVP(area);
        ar& BOOST_SERIALIZATION_NVP(cycles);
        ar& BOOST_SERIALIZATION_NVP(slowdown);
        ar& BOOST_SERIALIZATION_NVP(network);
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
  
  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Level);
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(subnest_);
      ar& BOOST_SERIALIZATION_NVP(specs_);
      ar& BOOST_SERIALIZATION_NVP(stats_);
    }
  }

 public:
  BufferLevel() { }
  BufferLevel(const Specs & specs);
  ~BufferLevel() { }

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the dynamic Spec() call later.
  static Specs ParseSpecs(libconfig::Setting& setting);
  static void ParseBufferSpecs(libconfig::Setting& buffer, problem::Shape::DataSpaceID pv, Specs& specs);
  static void ValidateTopology(BufferLevel::Specs& specs);
  
  bool DistributedMulticastSupported() override;
  
  // Evaluation functions.
  bool PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                          const tiling::CompoundMask mask,
                          const bool break_on_failure) override;
  bool Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                const double inner_tile_area, const std::uint64_t compute_cycles,
                const bool break_on_failure) override;

  // Private helpers.
  bool ComputeAccesses(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                       const bool break_on_failure);
  void ComputeArea();
  void ComputePerformance(const std::uint64_t compute_cycles);
  void ComputeNetworkEnergy(const double inner_tile_area);
  void ComputeBufferEnergy();
  void ComputeReductionEnergy();
  void ComputeAddrGenEnergy();

  // Accessors (post-evaluation).
  double StorageEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double NetworkEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double TemporalReductionEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double SpatialReductionEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double AddrGenEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  
  double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
 
  std::string Name() const override;
  double Area() const override;
  double AreaPerInstance() const override;
  double Size() const;
  std::uint64_t Cycles() const override;
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double CapacityUtilization() const override;
  std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  
  std::uint64_t MaxFanout() const override
  {
    // FIXME: remove this function, it's used only once.
    return stats_.network.fanout.Max();
  }

  // PAT interface.
  static double WireEnergyPerHop(std::uint64_t word_bits, const double inner_tile_area);
  static double NumHops(std::uint32_t multicast_factor, std::uint32_t fanout);

  // Printers.
  void Print(std::ostream& out) const;
  friend std::ostream& operator<<(std::ostream& out, const BufferLevel& buffer_level);  
};

}  // namespace model

//BOOST_CLASS_EXPORT(model::BufferLevel::Specs)
