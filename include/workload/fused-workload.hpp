#pragma once

#include <cstddef>

#include "workload/shape-models/problem-shape.hpp"

namespace problem
{

// TODO: this Einsum class is the old Workload class
using EinsumID = std::size_t;
class Einsum;

// TODO: this should be the new Workload class
class FusedWorkload
{
  FusedWorkload();

  Einsum& GetEinsum(const EinsumID& einsum_id);
  const Einsum& GetEinsum(const EinsumID& einsum_id) const;

  std::vector<Einsum>::iterator
  GetDspaceConsumers(const Shape::DataSpaceID& dspace_id);
  std::vector<Einsum>::const_iterator
  GetDspaceConsumers(const Shape::DataSpaceID& dspace_id) const;

  Einsum& GetDSpaceProducer(const Shape::DataSpaceID& dspace_id);
  const Einsum& GetDSpaceProducer(const Shape::DataSpaceID& dspace_id) const;
};

};