#include "einsum-graph/einsum-graph.hpp"

#include "isl-wrapper/isl-functions.hpp"

/******************************************************************************
 * Local declarations
 *****************************************************************************/

struct EinsumGraph::TiledEinsumsDfsVisitor : public boost::default_dfs_visitor
{
  TiledEinsumsDfsVisitor(std::set<problem::EinsumId>& visited_einsums,
                          const EinsumGraph& einsum_graph) :
    started_(false), ignore_(false), visited_einsums_(visited_einsums),
    einsum_graph_(einsum_graph)
  {
  }

  template<typename Vertex, typename Graph>
  void start_vertex(Vertex, const Graph&)
  {
    // The depth_first_search call goes through ALL vertices. This marks that
    // we are finished with connected component containing our starting vertex
    // and should ignore the rest.
    if (started_)
    {
      ignore_ = true;
    }
    started_ = true;
  }

  template<typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph&)
  {
    if (ignore_)
    {
      return;
    }

    auto dim = einsum_graph_.vertex_to_dim_.at(u);
    auto einsum_opt = einsum_graph_.workload_.GetEinsumWithDim(dim);

    if (einsum_opt)
    {
      visited_einsums_.emplace(*einsum_opt);
    }
  }

private:
  bool started_;
  bool ignore_;
  std::set<problem::EinsumId>& visited_einsums_;
  const EinsumGraph& einsum_graph_;
};

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

EinsumGraph::EinsumGraph(const problem::FusedWorkload& workload) :
  workload_(workload)
{
  auto dim_id_to_name =
    [&](problem::DimensionId d) { return workload_.GetDimensionName(d); };

  for (const auto& [_, dspace] : workload_.DataSpaceNameToId())
  {
    for (const auto& dspace_dim : workload_.DataSpaceDimensions(dspace))
    {
      auto vertex = boost::add_vertex(graph_);
      vertex_to_dim_[vertex] = dspace_dim;
      vertex_to_name_[vertex] = dim_id_to_name(dspace_dim);
      dim_to_vertex_[dspace_dim] = vertex;
    }
  }

  for (const auto& [_, einsum] : workload_.EinsumNameToId())
  {
    const auto& einsum_dim_to_idx = workload_.EinsumDimToIdx(einsum);

    for (const auto& [einsum_dim, _] : workload_.EinsumDimToIdx(einsum))
    {
      auto vertex = boost::add_vertex(graph_);
      vertex_to_dim_[vertex] = einsum_dim;
      vertex_to_name_[vertex] = dim_id_to_name(einsum_dim);
      dim_to_vertex_[einsum_dim] = vertex;
    }

    const auto& tensors_read_by_einsum = workload_.TensorsReadByEinsum(einsum);
    for (const auto& read_dspace : tensors_read_by_einsum)
    {
      const auto& dspace_dim_to_idx = workload_.DspaceDimToIdx(read_dspace);
      const auto& aff = workload_.ReadAccessesAff(einsum, read_dspace);

      for (const auto& [einsum_dim, einsum_dim_i] : einsum_dim_to_idx)
      {
        const auto& einsum_v = dim_to_vertex_.at(einsum_dim);
        for (const auto& [dspace_dim, dspace_dim_i] : dspace_dim_to_idx)
        {
          const auto& dspace_v = dim_to_vertex_.at(dspace_dim);

          auto coef = isl::val_to_double(isl_aff_get_coefficient_val(
            aff.at(dspace_dim_i).release(),
            isl_dim_in,
            einsum_dim_i
          ));

          if (coef != 0)
          {
            boost::add_edge(einsum_v, dspace_v, graph_);
          }
        }
      }
    }

    const auto& tensors_written_by_einsum =
      workload_.TensorsWrittenByEinsum(einsum);
    for (const auto& written_dspace : tensors_written_by_einsum)
    {
      const auto& dspace_dim_to_idx = workload_.DspaceDimToIdx(written_dspace);
      const auto& aff = workload_.WriteAccessesAff(einsum, written_dspace);

      for (const auto& [einsum_dim, einsum_dim_i] : einsum_dim_to_idx)
      {
        const auto& einsum_v = dim_to_vertex_.at(einsum_dim);
        for (const auto& [dspace_dim, dspace_dim_i] : dspace_dim_to_idx)
        {
          const auto& dspace_v = dim_to_vertex_.at(dspace_dim);

          auto coef = isl::val_to_double(isl_aff_get_coefficient_val(
            aff.at(dspace_dim_i).release(),
            isl_dim_in,
            einsum_dim_i
          ));

          if (coef != 0)
          {
            boost::add_edge(einsum_v, dspace_v, graph_);
            boost::add_edge(dspace_v, einsum_v, graph_);
          }
        }
      }
    }
  }
}

std::set<problem::EinsumId>
EinsumGraph::TiledEinsums(const problem::DimensionId& tiled_dim) const
{
  std::set<problem::EinsumId> tiled_einsums;
  auto vis = TiledEinsumsDfsVisitor(tiled_einsums, *this);

  boost::depth_first_search(graph_,
                            boost::visitor(vis).
                            root_vertex(dim_to_vertex_.at(tiled_dim)));

  return tiled_einsums;
}

void WriteGraphviz(std::ostream& os, const EinsumGraph& g)
{
  auto dim_id_to_name =
    [&](problem::DimensionId d) { return g.workload_.GetDimensionName(d); };

  os << "digraph G {" << std::endl;

  for (const auto& [dspace_name, dspace] : g.workload_.DataSpaceNameToId())
  {
    os << "subgraph cluster_" << dspace_name << " {" << std::endl;
    for (const auto& dspace_dim : g.workload_.DataSpaceDimensions(dspace))
    {
      const auto& dim_name = dim_id_to_name(dspace_dim);
      const auto& vertex = g.dim_to_vertex_.at(dspace_dim);
      os << vertex << "[label=\"" << dim_name << "\"];" << std::endl;
    }
    os << "label = \"" << dspace_name << "\";" << std::endl;
    os << "color = purple;" << std::endl;
    os << "}" << std::endl;
  }

  for (const auto& [einsum_name, einsum] : g.workload_.EinsumNameToId())
  {
    os << "subgraph cluster_" << einsum_name << " {" << std::endl;
    for (const auto& [einsum_dim, _] : g.workload_.EinsumDimToIdx(einsum))
    {
      const auto& dim_name = dim_id_to_name(einsum_dim);
      const auto& vertex = g.dim_to_vertex_.at(einsum_dim);
      os << vertex << "[label=\"" << dim_name << "\"];" << std::endl;
    }
    os << "label = \"" << einsum_name << "\";" << std::endl;
    os << "color = orange;" << std::endl;
    os << "}" << std::endl;
  }

  EinsumGraph::Graph::edge_iterator edge_it, edge_it_end;
  for (std::tie(edge_it, edge_it_end) = boost::edges(g.graph_);
       edge_it != edge_it_end;
       ++edge_it)
  {
    os << boost::source(*edge_it, g.graph_)
       << "->"
       << boost::target(*edge_it, g.graph_)
       << ";"
       << std::endl;
  }

  os << "}";
}
