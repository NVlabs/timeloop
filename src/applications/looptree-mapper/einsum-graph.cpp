#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "isl-wrapper/isl-functions.hpp"
#include "util/args.hpp"
#include "workload/fused-workload.hpp"

struct EinsumGraphLabelWriter;

struct EinsumGraph
{
  EinsumGraph(const problem::FusedWorkload& workload);

 private:
  using Graph = boost::adjacency_list<>;
  using Vertex = Graph::vertex_descriptor;

  const problem::FusedWorkload& workload_;

  Graph graph_;
  std::map<Vertex, problem::DimensionId> vertex_to_dim_;
  std::map<problem::DimensionId, Vertex> dim_to_vertex_;
  std::map<Vertex, std::string> vertex_to_name_;

  friend EinsumGraphLabelWriter;
  friend void WriteGraphviz(std::ostream& os, const EinsumGraph& g);
};

struct EinsumGraphLabelWriter
{
  EinsumGraphLabelWriter(const EinsumGraph& g) :
    g_(g)
  {
  }

  template<typename VertexOrEdge>
  void operator()(std::ostream& os, const VertexOrEdge& key)
  {
    os << "[label=\"" << g_.vertex_to_name_.at(key) << "\"]";
  }

 private:
  const EinsumGraph& g_;
};

void WriteGraphviz(std::ostream& os, const EinsumGraph& g)
{
  const auto& dim_id_to_name = g.workload_.DimensionIdToName();

  os << "digraph G {" << std::endl;

  for (const auto& [dspace_name, dspace] : g.workload_.DataSpaceNameToId())
  {
    os << "subgraph cluster_" << dspace_name << " {" << std::endl;
    for (const auto& dspace_dim : g.workload_.DataSpaceDimensions(dspace))
    {
      const auto& dim_name = dim_id_to_name.at(dspace_dim);
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
      const auto& dim_name = dim_id_to_name.at(einsum_dim);
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

EinsumGraph::EinsumGraph(const problem::FusedWorkload& workload) :
  workload_(workload)
{
  const auto& dim_id_to_name = workload_.DimensionIdToName();

  for (const auto& [_, dspace] : workload_.DataSpaceNameToId())
  {
    for (const auto& dspace_dim : workload_.DataSpaceDimensions(dspace))
    {
      auto vertex = boost::add_vertex(graph_);
      vertex_to_dim_[vertex] = dspace_dim;
      vertex_to_name_[vertex] = dim_id_to_name.at(dspace_dim);
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
      vertex_to_name_[vertex] = dim_id_to_name.at(einsum_dim);
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

int main(int argc, char* argv[])
{
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

  auto einsum_graph = EinsumGraph(workload);

  std::ofstream fout("out.dot");
  WriteGraphviz(fout, einsum_graph);
}