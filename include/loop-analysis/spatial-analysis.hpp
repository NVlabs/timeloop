#pragma once

#include <memory>

#include "loop-analysis/isl-ir.hpp"
#include "isl/polynomial.h"

namespace analysis
{

/******************************************************************************
 * Interface
 *****************************************************************************/

struct LinkTransferInfo
{
  Transfers link_transfer;
  Fill unfulfilled_fill;
};

struct LinkTransferModel
{
  virtual LinkTransferInfo Apply(const Fill&, const Occupancy&) const = 0;
};


struct MulticastInfo
{
  Reads reads;
  isl_pw_qpolynomial* p_hops;

  /***************** Compatibility with Timeloop v2.0 ************************/
  struct AccessStats
  {
    double accesses;
    double hops;
  };
  std::map<std::pair<uint64_t, uint64_t>, AccessStats> compat_access_stats;
  /***************************************************************************/

  ~MulticastInfo();
};

struct MulticastModel
{
  virtual MulticastInfo Apply(const Fill&) const = 0;
};


struct SpatialReuseInfo
{
  LinkTransferInfo link_transfer_info;
  MulticastInfo multicast_info;
};

struct SpatialReuseAnalysisInput
{
  const LogicalBuffer& buf;
  const Fill& children_fill;
  const Occupancy& children_occupancy;

  SpatialReuseAnalysisInput(const LogicalBuffer&,
                            const Fill&,
                            const Occupancy&);
};

struct SpatialReuseModels
{
  std::unique_ptr<LinkTransferModel> link_transfer_model;
  std::unique_ptr<MulticastModel> multicast_model;

  template<typename LinkTransferModelT, typename... ArgsT>
  void EmplaceLinkTransferModel(ArgsT&&... args)
  {
    link_transfer_model =
      std::make_unique<LinkTransferModelT>(std::forward(args)...);
  }

  template<typename MulticastModelT, typename... ArgsT>
  void EmplaceMulticastModel(ArgsT&&... args)
  {
    multicast_model = std::make_unique<MulticastModelT>(std::forward(args)...);
  }

  LinkTransferModel& GetLinkTransferModel();
  const LinkTransferModel& GetLinkTransferModel() const;

  MulticastModel& GetMulticastModel();
  const MulticastModel& GetMulticastModel() const;
};

SpatialReuseInfo SpatialReuseAnalysis(const SpatialReuseAnalysisInput& input,
                                      const SpatialReuseModels& models);


/******************************************************************************
 * Model Classes
 *****************************************************************************/

/**
 * @brief A link transfer model for 2-dimensional mesh interconnect.
 */
class SimpleLinkTransferModel : public LinkTransferModel
{
 public:
  SimpleLinkTransferModel();

  LinkTransferInfo
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
class SimpleMulticastModel : public MulticastModel
{
 public:
  SimpleMulticastModel();

  MulticastInfo Apply(const Fill& fills) const;

 private:
  isl::map connectivity_;
};

}