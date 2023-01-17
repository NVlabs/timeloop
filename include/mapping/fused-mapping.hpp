#pragma once

#include <memory>

#include "isl-wrapper/isl-wrapper.hpp"
#include "workload/workload.hpp"

struct MappingNode
{
};

struct SingleChild : public MappingNode
{
  std::unique_ptr<MappingNode> child;
};

struct ManyChildren : public MappingNode
{
  std::vector<std::unique_ptr<MappingNode>> children;
};

struct Root : public SingleChild
{
};

struct Loop : public SingleChild
{
  problem::Shape::FlattenedDimensionID op_dim;
  IslAff begin;
  IslAff end;
};

struct For : public Loop
{
};

struct ParFor : public Loop
{
};

struct Store : public SingleChild
{
  std::string buffer;
  std::set<problem::Shape::DataSpaceID> dspaces;
};

struct Compute : public MappingNode
{
  // problem::KernelID kernel;
};

struct Pipeline : public ManyChildren
{
};
