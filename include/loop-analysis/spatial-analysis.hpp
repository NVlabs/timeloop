#pragma once
#include "loop-analysis/isl-ir.hpp"
#include "isl/polynomial.h"

/******************************************************************************
 * Interface
 *****************************************************************************/

namespace analysis
{

struct LinkTransferInfo
{
  LogicalBufTransfers link_transfers;
  LogicalBufFills unfulfilled_fills;
};

struct LinkTransferModel
{
  virtual LinkTransferInfo
  Apply(LogicalBufFills&, LogicalBufOccupancies&) const = 0;
};

struct MulticastInfo
{
  std::map<LogicalBuffer, isl::map> reads;
  std::map<LogicalBuffer, isl_pw_qpolynomial*> p_hops;

  ~MulticastInfo();
};

struct MulticastModel
{
  virtual MulticastInfo
  Apply(LogicalBufFills&, LogicalBufOccupancies&) const = 0;
};

struct SpatialReuseInfo
{
  LinkTransferInfo link_transfer_info;
  MulticastInfo multicast_info;
};

SpatialReuseInfo SpatialReuseAnalysis(LogicalBufFills&,
                                      LogicalBufOccupancies&,
                                      const LinkTransferModel&,
                                      const MulticastModel&);

/******************************************************************************
 * Concrete Classes
 *****************************************************************************/

/**
 * @brief A link transfer model for 1- or 2-dimensional mesh interconnect.
 */
class SimpleLinkTransferModel : public LinkTransferModel
{
 public:
  SimpleLinkTransferModel(size_t n_spatial_dims);

  LinkTransferInfo
  Apply(LogicalBufFills& fills, LogicalBufOccupancies& occupancies) const;

 private:
  size_t n_spatial_dims_;
  isl::map connectivity_;
};

/**
 * @brief A multicast model for 1- or 2-dimensional array.
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
  SimpleMulticastModel(size_t n_spatial_dims);

  MulticastInfo
  Apply(LogicalBufFills& fills, LogicalBufOccupancies& occupancies) const;

 private:
  size_t n_spatial_dims_;
  isl::map connectivity_;
};
}