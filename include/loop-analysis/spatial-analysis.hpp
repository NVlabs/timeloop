#pragma once
#include "loop-analysis/isl-ir.hpp"

/******************************************************************************
 * Interface
 *****************************************************************************/

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
  LogicalBufTransfers multicasts;
};

struct MulticastModel
{
  virtual MulticastInfo
  Apply(LogicalBufFills&, LogicalBufOccupancies&) const = 0;
};

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