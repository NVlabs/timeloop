#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>

#include "workload/fused-workload.hpp"

struct EinsumGraph
{
  EinsumGraph(const problem::FusedWorkload& workload);

  std::set<problem::EinsumId>
  TiledEinsums(const problem::DimensionId& tiled_dim) const;

 private:
  using Graph = boost::adjacency_list<>;
  using Vertex = Graph::vertex_descriptor;

  const problem::FusedWorkload& workload_;

  Graph graph_;
  std::map<Vertex, problem::DimensionId> vertex_to_dim_;
  std::map<problem::DimensionId, Vertex> dim_to_vertex_;
  std::map<Vertex, std::string> vertex_to_name_;

  struct TiledEinsumsDfsVisitor;

  friend void WriteGraphviz(std::ostream& os, const EinsumGraph& g);
};

void WriteGraphviz(std::ostream& os, const EinsumGraph& g);
