#include <boost/test/unit_test.hpp>

#include <filesystem>

#include "isl-wrapper/ctx-manager.hpp"
#include "loop-analysis/mapping-to-isl/mapping-to-isl.hpp"

const auto TEST_CONFIG_PATH =
  std::filesystem::path(__FILE__).parent_path() / "configs";

BOOST_AUTO_TEST_CASE(TestMappingToIsl)
{
  const auto CONV1D_CONFIG_PATH = TEST_CONFIG_PATH / "conv1d.yaml";
  auto config = config::CompoundConfig({CONV1D_CONFIG_PATH.native()});
  auto workload = problem::Workload();
  problem::ParseWorkload(config.getRoot().lookup("problem"), workload);

  const auto rank_P = workload.GetShape()->FlattenedDimensionNameToID.at("P");
  const auto rank_R = workload.GetShape()->FlattenedDimensionNameToID.at("R");

  auto loop_nest = loop::Nest();
  loop_nest.AddLoop(rank_R, 0, 3, 1, spacetime::Dimension::SpaceX);
  loop_nest.AddLoop(rank_P, 0, 10, 1, spacetime::Dimension::Time, 6);
  loop_nest.AddStorageTilingBoundary();
  loop_nest.AddLoop(rank_P, 0, 2, 1, spacetime::Dimension::Time);
  loop_nest.AddStorageTilingBoundary();

  const auto occupancies = analysis::OccupanciesFromMapping(loop_nest,
                                                            workload);

  for (const auto& [buf, occ] : occupancies)
  {
    if (buf == analysis::LogicalBuffer(2, 2, 0))
    {
      BOOST_CHECK(occ.map.is_equal(
        isl::map(
          GetIslCtx().get(),
          "{ [0, 0, P1, 0, 0, P0, R, 0] -> [10*P1 + P0] : "
          "0 <= R < 3 and "
          "((P1 = 0 and 0 <= P0 < 10) or (P1 = 1 and 0 <= P0 < 6)) }"
        )
      ));
    }
  }
}