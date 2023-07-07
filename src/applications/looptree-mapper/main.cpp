#include <iostream>
#include <csignal>
#include <cstring>
#include <functional>

#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include "util/args.hpp"

extern bool gTerminateEval;

using problem::EinsumId;
using problem::DimensionId;

struct EinsumSet
{
  void AddEinsum(EinsumId einsum)
  {
    einsums_.emplace(einsum);
    hash_ ^= einsum;
  }

  void RemoveEinsum(EinsumId einsum)
  {
    einsums_.erase(einsum);
    hash_ ^= einsum;
  }

  inline size_t GetHash() const
  {
    return hash_;
  }

  size_t Size() const
  {
    return einsums_.size();
  }

 private:
  std::set<EinsumId> einsums_;
  size_t hash_;
};

template<>
struct std::hash<EinsumSet>
{
  size_t operator()(const EinsumSet& set) const
  {
    return set.GetHash();
  }
};

struct EinsumDimGraph
{
  std::set<DimensionId>& TilableDimensions(const std::set<EinsumId>& einsums);
};

struct HeadEinsumTracker
{
  HeadEinsumTracker(const problem::FusedWorkload& workload) :
    workload_(workload)
  {
  }

  void AddEinsum(EinsumId einsum);

  void RemoveEinsum(EinsumId einsum);

  std::set<EinsumId> NextEinsums() const;

 private:
  const problem::FusedWorkload& workload_;
  EinsumSet einsum_set_;
};

template<typename T>
struct Memo
{
  std::optional<std::reference_wrapper<T>>
  GetMemoizedValue(const EinsumSet& einsum_set)
  {
    auto it = memo_.find(einsum_set);
    if (it == memo_.end())
    {
      return std::nullopt;
    }
    else
    {
      return std::ref(it->second);
    }
  }

  void Memoize(const EinsumSet& einsum_set, const T& val)
  {
    memo_.emplace(einsum_set, val);
  }

 private:
  std::unordered_map<EinsumSet, T> memo_;
};

struct MapperResult;

MapperResult CombineResults(const MapperResult& res1, const MapperResult& res2)
{

}

template<typename T>
struct ParetoFrontier
{
  void Insert(const T& element);

  void Insert(const ParetoFrontier<T>& other);
};

template<typename T>
ParetoFrontier<T>
CombineParetoFrontiers(const ParetoFrontier<T>& frontier1,
                       const ParetoFrontier<T>& frontier2,
                       const std::function<T(const T&, const T&)>& op);

struct Mapper
{
  ParetoFrontier<MapperResult> SearchRestOfEinsums()
  {
    if (rest_of_einsums_.Size() == 0)
    {
      return ParetoFrontier<MapperResult>();
    }

    auto memoized_val_opt = memo_.GetMemoizedValue(rest_of_einsums_);
    if (memoized_val_opt)
    {
      return *memoized_val_opt;
    }

    auto cur_fused_set = EinsumSet();

    // Start a new fused set
    auto pareto = ParetoFrontier<MapperResult>();
    auto next_einsums = head_tracker_.NextEinsums();
    for (auto e : next_einsums)
    {
      head_tracker_.RemoveEinsum(e);
      rest_of_einsums_.RemoveEinsum(e);
      cur_fused_set.AddEinsum(e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set);
      pareto.Insert(SearchCurFusedSet(cur_fused_set));

      head_tracker_.AddEinsum(e);
      rest_of_einsums_.AddEinsum(e);
      cur_fused_set.RemoveEinsum(e);
    }

    memo_.Memoize(rest_of_einsums_, pareto);

    return pareto;
  }

  ParetoFrontier<MapperResult> SearchCurFusedSet(EinsumSet& cur_fused_set)
  {
    // Stop here
    auto cur_pareto = ExploreTilingAndReuseLevel(cur_fused_set);
    auto rest_pareto = SearchRestOfEinsums();
    auto pareto =
      CombineParetoFrontiers(cur_pareto, rest_pareto, CombineResults);
    
    // Keep going
    auto next_einsums = head_tracker_.NextEinsums();
    for (auto e : next_einsums)
    {
      cur_fused_set.AddEinsum(e);
      head_tracker_.RemoveEinsum(e);
      rest_of_einsums_.RemoveEinsum(e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set);
      pareto.Insert(SearchCurFusedSet(cur_fused_set));

      cur_fused_set.RemoveEinsum(e);
      head_tracker_.AddEinsum(e);
      rest_of_einsums_.AddEinsum(e);
    }

    return pareto;
  }

  ParetoFrontier<MapperResult>
  ExploreTilingAndReuseLevel(const EinsumSet& fused_set);

 private:
  Memo<ParetoFrontier<MapperResult>> memo_;
  HeadEinsumTracker head_tracker_;
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

  std::cout << EnumerateMappings(workload, 3) << std::endl;

  return 0;
}