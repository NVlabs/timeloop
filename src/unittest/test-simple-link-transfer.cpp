#include <boost/test/unit_test.hpp>

#include "isl-wrapper/ctx-manager.hpp"
#include "loop-analysis/spatial-analysis.hpp"

BOOST_AUTO_TEST_CASE(TestSimpleLinkTransferModel)
{
  using namespace analysis;
  using namespace spacetime;

  auto fill = Fill(
    {Dimension::Time, Dimension::SpaceX, Dimension::SpaceY},
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [t+x+y] : 0 <= x < 2 and 0 <= y < 2 and 0 <= t < 2}"
    )
  );

  auto occ = Occupancy(fill.dim_in_tags, fill.map);

  auto link_transfer_model = SimpleLinkTransferModel();

  auto info = link_transfer_model.Apply(fill, occ);

  BOOST_CHECK(info.fulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t = 1, x, y = 0] -> [1 + x] : 0 <= x <= 1; "
      "  [t = 1, x = 0, y] -> [1 + y] : 0 <= y <= 1 }"
    )
  ));

  BOOST_CHECK(info.unfulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t = 0, x, y] -> [x + y] : 0 <= x <= 1 and 0 <= y <= 1; "
      "  [t = 1, x = 1, y = 1] -> [3] }"
    )
  ));
}