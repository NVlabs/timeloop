#include <boost/test/unit_test.hpp>

#include "isl-wrapper/ctx-manager.hpp"
#include "loop-analysis/spatial-analysis.hpp"

BOOST_AUTO_TEST_CASE(TestSimpleMulticastModel_0)
{
  using namespace analysis;

  auto fill = Fill(
    {Temporal(), Spatial(0), Spatial(1)},
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [t + x + y] : 0 <= x < 2 and 0 <= y < 2 and 0 <= t < 4 }"
    )
  );

  auto occ = Occupancy(fill.dim_in_tags, fill.map);

  auto multicast_model = SimpleMulticastModel(false);

  auto info = multicast_model.Apply(fill, occ);

  BOOST_CHECK(info.fulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [t + x + y] : 0 <= t < 4 and 0 <= x < 2 and 0 <= y < 2 }"
    )
  ));

  BOOST_CHECK(info.parent_reads.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t] -> [d] : 0 <= t < 4 and t <= d < t + 3 }"
    )
  ));

  BOOST_CHECK(info.compat_access_stats.size() == 1);
  for (const auto& [multicast_scatter, stats] : info.compat_access_stats)
  {
    auto [multicast, scatter] = multicast_scatter;

    BOOST_CHECK(multicast == 1);
    BOOST_CHECK(scatter == 1);
    BOOST_CHECK(stats.accesses == 12);
    BOOST_CHECK(stats.hops == 0);
  }

  multicast_model = SimpleMulticastModel(true);

  info = multicast_model.Apply(fill, occ);

  BOOST_CHECK(info.fulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [t + x + y] : 0 <= t < 4 and 0 <= x < 2 and 0 <= y < 2 }"
    )
  ));

  BOOST_CHECK(info.parent_reads.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t] -> [d] : 0 <= t < 4 and t <= d < t + 3 }"
    )
  ));

  BOOST_CHECK(info.compat_access_stats.size() == 1);
  for (const auto& [multicast_scatter, stats] : info.compat_access_stats)
  {
    auto [multicast, scatter] = multicast_scatter;

    BOOST_CHECK(multicast == 1);
    BOOST_CHECK(scatter == 1);
    BOOST_CHECK(stats.accesses == 12);
    BOOST_TEST(stats.hops == 3.667, boost::test_tools::tolerance(0.001));
  }
}

BOOST_AUTO_TEST_CASE(TestSimpleMulticastModel_SpatialPC)
{
  using namespace analysis;

  auto fill = Fill(
    {Temporal(), Spatial(0), Spatial(1)},
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [d, y] : 0 <= x < 4 and 0 <= y < 2 and 0 <= t < 4 and x <= d < x+2 }"
    )
  );

  auto occ = Occupancy(fill.dim_in_tags, fill.map);

  auto multicast_model = SimpleMulticastModel(true);

  auto info = multicast_model.Apply(fill, occ);

  BOOST_CHECK(info.fulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [d, y] : 0 <= x < 4 and 0 <= y < 2 and 0 <= t < 4 and x <= d < x+2 }"
    )
  ));

  BOOST_CHECK(info.parent_reads.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t] -> [d, y] : 0 <= y < 2 and 0 <= t < 4 and 0 <= d < 5 }"
    )
  ));

  BOOST_CHECK(info.compat_access_stats.size() == 1);
  for (const auto& [multicast_scatter, stats] : info.compat_access_stats)
  {
    auto [multicast, scatter] = multicast_scatter;

    BOOST_CHECK(multicast == 1);
    BOOST_CHECK(scatter == 1);
    BOOST_CHECK(stats.accesses == 40);
    BOOST_TEST(stats.hops == 5.2, boost::test_tools::tolerance(0.001));
  }
}