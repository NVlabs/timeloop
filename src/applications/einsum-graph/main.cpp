#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>

#include "einsum-graph/einsum-graph.hpp"
#include "isl-wrapper/isl-functions.hpp"
#include "util/args.hpp"
#include "workload/fused-workload.hpp"

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