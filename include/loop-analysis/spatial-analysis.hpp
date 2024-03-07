#pragma once

#include <memory>

#include "loop-analysis/isl-ir.hpp"
#include "isl/polynomial.h"

namespace analysis
{

/******************************************************************************
 * Interface
 *****************************************************************************/

struct TransferInfo
{
  Transfers fulfilled_fill;
  Reads parent_reads;
  Fill unfulfilled_fill;

  /***************** Compatibility with Timeloop v2.0 ************************/
  bool is_multicast = false;
  bool is_link_transfer = false;

  struct AccessStats
  {
    double accesses;
    double hops;
    double unicast_hops;
  };
  std::map<std::pair<uint64_t, uint64_t>, AccessStats> compat_access_stats;

  double total_child_accesses = 0;
  /***************************************************************************/
};

struct SpatialReuseModel
{
  virtual TransferInfo Apply(const Fill&, const Occupancy&) const = 0;

  virtual ~SpatialReuseModel() = default;
};


struct SpatialReuseInfo
{
  std::vector<TransferInfo> transfer_infos;
};

struct FillProvider
{
  std::unique_ptr<SpatialReuseModel> spatial_reuse_model;
  const LogicalBuffer buf;
  const Occupancy& occupancy;

  FillProvider(const LogicalBuffer& buf, const Occupancy& occ);

  template<typename ReuseModelT, typename... ArgsT>
  static FillProvider MakeFillProvider(
    const LogicalBuffer& buf,
    const Occupancy& occ,
    ArgsT&&... args
  )
  {
    auto fill_provider = FillProvider(buf, occ);
    fill_provider.spatial_reuse_model =
      std::make_unique<ReuseModelT>(std::forward<ArgsT>(args)...);
    return fill_provider;
  }
};


struct SpatialReuseAnalysisInput
{
  const LogicalBuffer buf;
  const Fill& children_fill;
  std::vector<FillProvider> fill_providers;

  SpatialReuseAnalysisInput(LogicalBuffer buf, const Fill& children_fill);
};


SpatialReuseInfo SpatialReuseAnalysis(const SpatialReuseAnalysisInput& input);


/******************************************************************************
 * Model Classes
 *****************************************************************************/

/**
 * @brief A link transfer model for 2-dimensional mesh interconnect.
 */
class SimpleLinkTransferModel final : public SpatialReuseModel
{
 public:
  SimpleLinkTransferModel();

  TransferInfo
  Apply(const Fill& fills, const Occupancy& occupancies) const override;

 private:
  isl::map connectivity_;
};


/**
 * @brief A multicast model for 2-dimensional array.
 *
 * @note Differs from Timeloop's original multicast model in terms of partial
 *   tile overlap multicast. This model assumes partial overlaps can benefit
 *   from multicasting. The original model does the opposite.
 * @note Does not directly model distributed multicast. Uses the same methods
 *   as the original multicast model (i.e., assumes parents have equally
 *   distributed tiles)
 */
class SimpleMulticastModel final : public SpatialReuseModel
{
 public:
  SimpleMulticastModel();

  TransferInfo
  Apply(const Fill& fills, const Occupancy& occupancy) const override;
};

}; // namespace analysis
