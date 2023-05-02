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