#include <iostream>
#include <csignal>
#include <cstring>
#include <functional>

#include <boost/range/adaptor/map.hpp>

#include "einsum-graph/einsum-graph.hpp"
#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include "util/args.hpp"
#include "util/hashable-set.hpp"
#include "util/pareto-frontier.hpp"

extern bool gTerminateEval;

using problem::EinsumId;
using problem::DimensionId;

using EinsumSet = HashableSet<EinsumId>;
using DataSpaceSet = HashableSet<problem::DataSpaceId>;

size_t factorial(size_t n)
{
  if (n == 1)
  {
    return 1;
  }
  return n*factorial(n-1);
}

struct EinsumDimGraph
{
  std::set<DimensionId>& TilableDimensions(const std::set<EinsumId>& einsums);
};

struct HeadEinsumTracker
{
  HeadEinsumTracker(const problem::FusedWorkload& workload) :
    workload_(workload)
  {
    using namespace boost::adaptors;
    auto range = workload.EinsumNameToId() | map_values;
    rest_of_einsums_ = std::set<EinsumId>(range.begin(), range.end());


    for (auto einsum_id : rest_of_einsums_)
    {
      auto is_head = true;

      for (auto out_dspace : workload.TensorsWrittenByEinsum(einsum_id))
      {
        for (auto consumer_einsum : workload.ReaderEinsums(out_dspace))
        {
          auto it = rest_of_einsums_.find(consumer_einsum);
          if (it != rest_of_einsums_.end())
          {
            is_head = false;
            break;
          }
        }
        if (!is_head)
        {
          break;
        }
      }

      if (is_head)
      {
        next_einsums_.insert(einsum_id);
      }
    }
  }

  void TakeEinsum(EinsumId einsum)
  {
    auto n_erased = next_einsums_.erase(einsum);
    if (n_erased != 1)
    {
      throw std::logic_error("taking non-head einsum");
    }

    rest_of_einsums_.erase(einsum);
    taken_einsums_.emplace(einsum);

    for (const auto& tensor : workload_.TensorsReadByEinsum(einsum))
    {
      auto produced_by_head_einsum = true;
      for (const auto& reader_einsum : workload_.ReaderEinsums(tensor))
      {
        auto it = rest_of_einsums_.find(reader_einsum);
        if (it != rest_of_einsums_.end())
        {
          produced_by_head_einsum = false;
          break;
        }
      }

      if (produced_by_head_einsum)
      {
        auto new_head_einsum_opt = workload_.WriterEinsum(tensor);
        if (new_head_einsum_opt)
        {
          next_einsums_.emplace(*new_head_einsum_opt);
        }
      }
    }
  }

  void UntakeEinsum(EinsumId einsum)
  {
    const auto& read_by_einsum = workload_.TensorsReadByEinsum(einsum);

    auto new_next_einsums = std::set<EinsumId>();

    for (const auto& cur_head_einsum : next_einsums_)
    {
      bool still_head_einsum = true;
      for (const auto& t : workload_.TensorsWrittenByEinsum(cur_head_einsum))
      {
        auto it = read_by_einsum.find(t);
        if (it != read_by_einsum.end())
        {
          still_head_einsum = false;
          break;
        }
      }

      if (still_head_einsum)
      {
        new_next_einsums.emplace(cur_head_einsum);
      }
    }

    auto new_einsum_is_head = true;
    const auto& written_by_einsum = workload_.TensorsWrittenByEinsum(einsum);
    for (const auto& cur_head_einsum : new_next_einsums)
    {
      auto read_by_cur_head = workload_.TensorsReadByEinsum(cur_head_einsum);
      for (const auto& t : written_by_einsum)
      {
        auto it = read_by_cur_head.find(t);
        if (it != read_by_cur_head.end())
        {
          new_einsum_is_head = false;
          break;
        }
      }

      if (!new_einsum_is_head)
      {
        break;
      }
    }

    if (new_einsum_is_head)
    {
      new_next_einsums.emplace(einsum);
    }

    rest_of_einsums_.emplace(einsum);
    taken_einsums_.erase(einsum);
    next_einsums_ = new_next_einsums;
  }

  const std::set<EinsumId>& NextEinsums() const
  {
    return next_einsums_;
  }
  const std::set<EinsumId>& RestOfEinsums() const
  {
    return rest_of_einsums_;
  }

 private:
  const problem::FusedWorkload& workload_;
  std::set<EinsumId> next_einsums_;
  std::set<EinsumId> rest_of_einsums_;
  std::set<EinsumId> taken_einsums_;
};

template<typename KeyT, typename ValT>
struct Memo
{
  Memo() : memo_() {}

  std::optional<std::reference_wrapper<ValT>>
  GetMemoizedValue(const KeyT& key)
  {
    auto it = memo_.find(key);
    if (it == memo_.end())
    {
      return std::nullopt;
    }
    else
    {
      return std::ref(it->second);
    }
  }

  void Memoize(const KeyT& key, const ValT& val)
  {
    memo_.emplace(key, val);
  }

 private:
  std::unordered_map<KeyT, ValT> memo_;
};

struct MapperResult
{
  size_t transfers;
  size_t capacity;
  size_t computation;
};

bool operator<(const MapperResult& res1, const MapperResult& res2)
{
  return res1.transfers < res2.transfers
         && res1.capacity < res2.capacity
         && res1.computation < res2.computation;
}

bool operator>(const MapperResult& res1, const MapperResult& res2)
{
  return res1.transfers > res2.transfers
         && res1.capacity > res2.capacity
         && res1.computation > res2.computation;
}

MapperResult CombineResults(const MapperResult& res1, const MapperResult& res2)
{
  return MapperResult
  {
    .transfers = res1.transfers + res2.transfers,
    .capacity = std::max(res1.capacity, res2.capacity),
    .computation = res1.computation + res2.computation
  };
}

struct Subproblem
{
  EinsumSet einsums;
  // We always go to DRAM between fused sets for now
  // DataSpaceSet dspaces_on_chip;
};

bool operator==(const Subproblem& sp1, const Subproblem& sp2)
{
  return sp1.einsums == sp2.einsums;
}

template<>
struct std::hash<Subproblem>
{
  std::size_t operator()(const Subproblem& subproblem) const noexcept
  {
    return std::hash<EinsumSet>()(subproblem.einsums);
  }
};

struct Mapper
{
  Mapper(const problem::FusedWorkload& workload) :
    workload_(workload), head_tracker_(workload), einsum_graph_(workload)
  {
    using namespace boost::adaptors;
    rest_of_einsums_ = EinsumSet(workload.EinsumNameToId() | map_values);
  }

  ParetoFrontier<MapperResult> Run()
  {
    return SearchRestOfEinsums();
  }

  ParetoFrontier<MapperResult> SearchRestOfEinsums()
  {
    std::cout << "rest of einsums: " << rest_of_einsums_ << std::endl;
    if (rest_of_einsums_.Size() == 0)
    {
      return ParetoFrontier<MapperResult>();
    }

    auto memoized_val_opt =
      memo_.GetMemoizedValue(Subproblem{rest_of_einsums_});
    if (memoized_val_opt)
    {
      return *memoized_val_opt;
    }

    auto cur_fused_set = std::set<EinsumId>();

    // Start a new fused set
    auto pareto = ParetoFrontier<MapperResult>();
    auto next_einsums = head_tracker_.NextEinsums();
    for (auto e : next_einsums)
    {
      std::cout << "next einsum: " << e << std::endl;
      head_tracker_.TakeEinsum(e);
      rest_of_einsums_ = Erase(std::move(rest_of_einsums_), e);
      cur_fused_set.emplace(e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set, e);
      pareto.Insert(cur_pareto);

      head_tracker_.UntakeEinsum(e);
      rest_of_einsums_ = Emplace(std::move(rest_of_einsums_), e);
      cur_fused_set.erase(e);
    }

    memo_.Memoize(Subproblem{rest_of_einsums_}, pareto);

    return pareto;
  }

  ParetoFrontier<MapperResult>
  SearchCurFusedSet(std::set<EinsumId>& cur_fused_set, EinsumId cur_head)
  {
    auto einsum_str = std::vector<std::string>();
    std::transform(cur_fused_set.begin(), cur_fused_set.end(),
                   std::back_inserter(einsum_str),
                   [](EinsumId e) { return std::to_string(e); });

    // Stop here
    auto cur_pareto = ExploreTilingAndReuseLevel(cur_fused_set, cur_head);
    auto rest_pareto = SearchRestOfEinsums();
    auto pareto =
      CombineParetoFrontiers<MapperResult>(cur_pareto, rest_pareto, CombineResults);
    
    // Keep going
    auto next_einsums = head_tracker_.NextEinsums();
    for (auto e : next_einsums)
    {
      std::cout << "next einsum: " << e << std::endl;

      cur_fused_set.emplace(e);
      head_tracker_.TakeEinsum(e);
      rest_of_einsums_ = Erase(std::move(rest_of_einsums_), e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set, cur_head);
      pareto.Insert(cur_pareto);

      cur_fused_set.erase(e);
      head_tracker_.UntakeEinsum(e);
      rest_of_einsums_ = Emplace(std::move(rest_of_einsums_), e);
    }

    return pareto;
  }

  ParetoFrontier<MapperResult>
  ExploreTilingAndReuseLevel(std::set<EinsumId>& fused_set, EinsumId cur_head)
  {
    auto einsum_str = std::vector<std::string>();
    std::transform(fused_set.begin(), fused_set.end(),
                   std::back_inserter(einsum_str),
                   [](EinsumId e) { return std::to_string(e); });
    
    std::cout << "exploring fused set: "
              << boost::join(einsum_str, ", ")
              << std::endl;
    
    auto intermediate_dspaces = std::set<EinsumId>();
    auto in_dspaces = std::set<EinsumId>();
    auto out_dspaces = std::set<EinsumId>();
    for (auto e : fused_set)
    {
      for (auto dspace : workload_.TensorsReadByEinsum(e))
      {
        auto it = out_dspaces.find(dspace);
        if (it != out_dspaces.end())
        {
          out_dspaces.erase(it);
          intermediate_dspaces.insert(dspace);
        }
        else
        {
          in_dspaces.insert(dspace);
        }
      }

      for (auto dspace : workload_.TensorsWrittenByEinsum(e))
      {
        auto it = in_dspaces.find(dspace);
        if (it != in_dspaces.end())
        {
          in_dspaces.erase(it);
          intermediate_dspaces.insert(dspace);
        }
        else
        {
          out_dspaces.insert(dspace);
        }
      }
    }

    // Determine dimensions we can tile
    auto tilable_dims = std::set<DimensionId>();
    for (const auto& [dim, _] : workload_.EinsumDimToIdx(cur_head))
    {
      auto tiled_einsums = einsum_graph_.TiledEinsums(dim);
      auto tilable = true;
      for (auto e : fused_set)
      {
        auto it = tiled_einsums.find(e);
        if (it == tiled_einsums.end())
        {
          tilable = false;
          break;
        }
      }
      if (tilable)
      {
        tilable_dims.emplace(dim);
      }
    }

    // At this point we have to make the following choices:
    // 1. choose reuse level of each data space.
    // 2. choose tiled dimension and order.
    // 3. choose tile size.

    // TODO: this is TERRIBLE!
    auto tile_sizes = std::map<DimensionId, std::vector<size_t>>({
      {0, {1, 4, 16, 64, 256, 512}},
      {1, {1, 4, 16, 64, 256, 512}},
      {2, {1, 4, 16, 64}},
      {9, {1, 4, 16, 64, 256, 512}},
      {10, {1, 4, 16, 64, 256, 512}},
      {11, {1, 4, 16, 64}},
    });

    auto n_dspaces = in_dspaces.size() + out_dspaces.size()
                    + intermediate_dspaces.size();
    auto n_evals = 0;
    auto tiling = std::vector<DimensionId>(tilable_dims.begin(),
                                           tilable_dims.end());
    auto keep_going = true;
    while (keep_going)
    {
      auto einsum_str = std::vector<std::string>();
      std::transform(tiling.begin(), tiling.end(),
                    std::back_inserter(einsum_str),
                    [](EinsumId e) { return std::to_string(e); });
      std::cout << boost::join(einsum_str, ",") << std::endl;

      // auto mapping = mapping::FusedMapping();
      // auto parent = mapping.GetRoot().id;
      // parent = mapping.AddChild<mapping::Storage>(parent, 1, dspace);
      auto n_tile_sizes = 1;
      for (auto d : tiling)
      {
        n_tile_sizes *= tile_sizes.at(d).size();
      }
      n_evals += pow(tiling.size(), n_dspaces)*n_tile_sizes;

      keep_going = std::next_permutation(tiling.begin(), tiling.end());
    }
    std::cout << "num evals: " << n_evals << std::endl;

    return ParetoFrontier<MapperResult>();
  }

 private:
  const problem::FusedWorkload& workload_;

  Memo<Subproblem, ParetoFrontier<MapperResult>> memo_;
  HeadEinsumTracker head_tracker_;
  EinsumGraph einsum_graph_;
  EinsumSet rest_of_einsums_;
};

void handler(int s)
{
  if (!gTerminateEval)
  {
    std::cerr << "First " << strsignal(s) << " caught. Abandoning "
              << "ongoing evaluation and terminating immediately."
              << std::endl;
    gTerminateEval = true;
  }
  else
  {
    std::cerr << "Second " << strsignal(s) << " caught. Existing disgracefully."
              << std::endl;
    exit(0);
  }
}


int main(int argc, char* argv[])
{
  assert(argc >= 2);

  struct sigaction action;
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, NULL);
  
  std::vector<std::string> input_files;
  std::string output_dir = ".";
  bool success = ParseArgs(argc, argv, input_files, output_dir);
  if (!success)
  {
    std::cerr << "ERROR: error parsing command line." << std::endl;
    exit(1);
  }

  auto config = config::CompoundConfig(input_files);
  auto root = config.getRoot();

  auto workload = problem::ParseFusedWorkload(root.lookup("problem"));

  auto mapper = Mapper(workload);
  mapper.Run();

  return 0;
}