#include <iostream>
#include <csignal>
#include <cstring>
#include <functional>

#include <boost/range/adaptor/map.hpp>

#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include "util/args.hpp"
#include "util/hashable-set.hpp"

extern bool gTerminateEval;

using problem::EinsumId;
using problem::DimensionId;

using EinsumSet = HashableSet<EinsumId>;

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
  }

  void TakeEinsum(EinsumId einsum);
  void UntakeEinsum(EinsumId einsum);

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
    .capacity = res1.capacity + res2.capacity,
    .computation = res1.computation + res2.computation
  };
}

template<typename T>
struct ParetoFrontier
{
  struct Iterator
  {
    bool operator!=(const Iterator& other)
    {
      return it_ != other.it_;
    }
    bool operator==(const Iterator& other)
    {
      return it_ == other.it_;
    }

    const T& operator*()
    {
      return *(*it_);
    }

    Iterator& operator++()
    {
      ++it_;
      while (it_ != end_ && !(*it_))
      {
        ++it_;
      }
      return *this;
    }

   private:
    Iterator(typename std::vector<std::optional<T>>::const_iterator it,
             typename std::vector<std::optional<T>>::const_iterator end) :
      it_(std::move(it)), end_(std::move(end))
    {
    }

    typename std::vector<std::optional<T>>::const_iterator it_;
    typename std::vector<std::optional<T>>::const_iterator end_;

    friend ParetoFrontier;
  };

  ParetoFrontier() : arr_(), size_(0) {}

  Iterator begin() const
  {
    return Iterator(arr_.begin(), arr_.end());
  }

  Iterator end() const
  {
    return Iterator(arr_.end(), arr_.end());
  }

  void Insert(const T& element)
  {
    bool inserted = false;
    for (auto& e : arr_)
    {
      if (!e)
      {
        continue;
      }

      if (element > *e)
      {
        return;
      }
      else if (element < *e)
      {
        e = std::nullopt;
        size_ -= 1;
        if (!inserted)
        {
          e = element;
          inserted = true;
          size_ += 1;
        }
      }
    }
    if (!inserted)
    {
      arr_.emplace_back(element);
    }

    if (size_ < arr_.size() / 2)
    {
      size_t i = 0;
      for (; i < arr_.size(); ++i)
      {
        if (!arr_.at(i))
        {
          ++i;
          break;
        }
      }

      for (size_t j = i; j < arr_.size(); ++j)
      {
        if (arr_.at(j))
        {
          arr_.at(i) = arr_.at(j);
        }
      }

      assert(i == size_);
      arr_.resize(i);
    }
  }

  void Insert(const ParetoFrontier<T>& other)
  {
    for (const auto& e : other)
    {
      Insert(e);
    }
  }

 private:
  std::vector<std::optional<T>> arr_;
  size_t size_;
};

template<typename T>
ParetoFrontier<T>
CombineParetoFrontiers(const ParetoFrontier<T>& frontier1,
                       const ParetoFrontier<T>& frontier2,
                       const std::function<T(const T&, const T&)>& op)
{
  auto pareto = ParetoFrontier<T>();
  for (const auto& e1 : frontier1)
  {
    for (const auto& e2 : frontier2)
    {
      pareto.Insert(op(e1, e2));
    }
  }
  return pareto;
}

struct Mapper
{
  Mapper(const problem::FusedWorkload& workload) :
    head_tracker_(workload)
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
    if (rest_of_einsums_.Size() == 0)
    {
      return ParetoFrontier<MapperResult>();
    }

    auto memoized_val_opt = memo_.GetMemoizedValue(rest_of_einsums_);
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
      head_tracker_.TakeEinsum(e);
      rest_of_einsums_ = Erase(std::move(rest_of_einsums_), e);
      cur_fused_set.emplace(e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set);
      pareto.Insert(SearchCurFusedSet(cur_fused_set));

      head_tracker_.UntakeEinsum(e);
      rest_of_einsums_ = Emplace(std::move(rest_of_einsums_), e);
      cur_fused_set.erase(e);
    }

    memo_.Memoize(rest_of_einsums_, pareto);

    return pareto;
  }

  ParetoFrontier<MapperResult>
  SearchCurFusedSet(std::set<EinsumId>& cur_fused_set)
  {
    // Stop here
    auto cur_pareto = ExploreTilingAndReuseLevel(cur_fused_set);
    auto rest_pareto = SearchRestOfEinsums();
    auto pareto =
      CombineParetoFrontiers<MapperResult>(cur_pareto, rest_pareto, CombineResults);
    
    // Keep going
    auto next_einsums = head_tracker_.NextEinsums();
    for (auto e : next_einsums)
    {
      cur_fused_set.emplace(e);
      head_tracker_.TakeEinsum(e);
      rest_of_einsums_ = Erase(std::move(rest_of_einsums_), e);

      auto cur_pareto = SearchCurFusedSet(cur_fused_set);
      pareto.Insert(SearchCurFusedSet(cur_fused_set));

      cur_fused_set.erase(e);
      head_tracker_.UntakeEinsum(e);
      rest_of_einsums_ = Emplace(std::move(rest_of_einsums_), e);
    }

    return pareto;
  }

  ParetoFrontier<MapperResult>
  ExploreTilingAndReuseLevel(const EinsumSet& fused_set)
  {
    std::cout << "exploring fused set: " << fused_set << std::endl;
    return ParetoFrontier<MapperResult>();
  }

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

  auto mapper = Mapper(workload);

  return 0;
}