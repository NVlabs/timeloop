#include <iostream>
#include <csignal>
#include <cstring>

#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include "util/args.hpp"

extern bool gTerminateEval;

template<typename DimIterT>
struct TilingOrderIter
{
  using DimT = typename std::iterator_traits<DimIterT>::value_type;

  TilingOrderIter& operator++()
  {
    if (iters_.at(cur_depth_) == end_)
    {
      // end iterator
      return *this;
    }

    if (cur_depth_ < max_depth_)
    {
      cur_depth_++;
      iters_.at(cur_depth_) = begin_;
    }
    else
    {
      iters_.at(cur_depth_)++;
    }

    while (cur_depth_ > 0
           && (iters_.at(cur_depth_) == end_
               || iters_.at(cur_depth_) == iters_.at(cur_depth_-1)))
    {
      if (iters_.at(cur_depth_) == end_)
      {
        cur_depth_--;
      }
      iters_.at(cur_depth_)++;
    }

    return *this;
  }

  bool operator==(const TilingOrderIter& other) const
  {
    if (max_depth_ != other.max_depth_)
    {
      return false;
    }
    if (cur_depth_ != other.cur_depth_)
    {
      return false;
    }
    for (auto i = 0; i <= max_depth_; ++i)
    {
      if (iters_.at(i) != other.iters_.at(i))
      {
        return false;
      }
    }

    return true;
  }

  bool operator!=(const TilingOrderIter& other) const
  {
    return !(*this == other);
  }

  const std::vector<DimT>& operator*() const
  {
    tiling_order_.clear();
    for (auto i = 0; i <= cur_depth_; ++i)
    {
      tiling_order_.emplace_back(*iters_.at(i));
    }

    return tiling_order_;
  }

  static TilingOrderIter<DimIterT>
  begin(int max_depth, DimIterT begin, DimIterT end)
  {
    return TilingOrderIter(max_depth, begin, end);
  }

  static TilingOrderIter<DimIterT>
  end(int max_depth, DimIterT begin, DimIterT end)
  {
    auto iter = TilingOrderIter(max_depth, begin, end);
    for (auto i = 0; i <= max_depth; ++i)
    {
      iter.iters_.at(i) = end;
    }
    iter.cur_depth_ = 0;

    return iter;
  }

 private:
  const DimIterT begin_;
  const DimIterT end_;
  const int max_depth_;

  int cur_depth_;
  std::vector<DimIterT> iters_;
  mutable std::vector<DimT> tiling_order_;

  TilingOrderIter(int max_depth, DimIterT begin, DimIterT end) :
    begin_(begin), end_(end), max_depth_(max_depth),
    cur_depth_(0), iters_(max_depth+1, begin), tiling_order_()
  {
    tiling_order_.reserve(max_depth+1);
  }
};

template<typename DimIterT>
struct std::iterator_traits<TilingOrderIter<DimIterT>>
{
  using value_type = typename TilingOrderIter<DimIterT>::DimT;
};

template<typename DimIterT>
struct TilingOrders
{
  TilingOrders(int max_depth, DimIterT begin, DimIterT end) :
    begin_(TilingOrderIter<DimIterT>::begin(max_depth, begin, end)),
    end_(TilingOrderIter<DimIterT>::end(max_depth, begin, end))
  {
  }

  TilingOrderIter<DimIterT> begin() const
  {
    return begin_;
  }

  TilingOrderIter<DimIterT> end() const
  {
    return end_;
  }

 private:
  TilingOrderIter<DimIterT> begin_;
  TilingOrderIter<DimIterT> end_;
};

template<typename IterT>
TilingOrders<IterT> ForAllMeaningfulTilingOrder(int max_depth,
                                                IterT begin, IterT end)
{
  return TilingOrders(max_depth, begin, end);
}

size_t EnumerateMappings(const problem::FusedWorkload& workload,
                         int max_tile_depth)
{
  auto final_einsum_id = std::optional<problem::EinsumId>();
  for (const auto& [dspace_name, dspace_id] : workload.DataSpaceNameToId())
  {
    if (workload.ReaderEinsums(dspace_id).size() == 0)
    {
      final_einsum_id = workload.WriterEinsum(dspace_id);
      break;
    }
  }

  if (!final_einsum_id)
  {
    throw std::logic_error("Could not find final Einsum");
  }

  std::cout << "Final einsum: " << *final_einsum_id << std::endl;
  const auto& tilable_dims_to_idx = workload.EinsumDimToIdx(*final_einsum_id);

  std::cout << "There are " << tilable_dims_to_idx.size() << " dims." << std::endl;

  auto tilable_dims = std::vector<problem::DimensionId>();
  for (const auto& [dim, idx] : tilable_dims_to_idx)
  {
    tilable_dims.emplace_back(dim);
  }

  auto tiling_orders = ForAllMeaningfulTilingOrder(
    max_tile_depth, tilable_dims.begin(), tilable_dims.end()
  );

  auto count = 0;
  for (const auto& tiling_order : tiling_orders)
  {
    (void) tiling_order;
    count++;
  }

  return count;
}

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