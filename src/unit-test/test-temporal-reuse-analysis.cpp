#include <boost/test/unit_test.hpp>

#include "isl-wrapper/ctx-manager.hpp"
#include "loop-analysis/temporal-analysis.hpp"


BOOST_AUTO_TEST_CASE(TestTemporalReuse_MultipleLoopReuse)
{
  using namespace analysis;

  Occupancy occ = Occupancy(
    {Temporal(), Spatial(0, 0), Temporal()},
    isl::map(
      GetIslCtx(),
      "{ [t1, x, t0] -> [d] : "
      "t0 <= d < t0+3 and "
      "0 <= t1 < 2 and 0 <= x < 2 and 0 <= t0 < 2 }"
    )
  );

  analysis::TemporalReuseAnalysisOutput result = analysis::TemporalReuseAnalysis(
      analysis::TemporalReuseAnalysisInput(
        occ,
        analysis::BufTemporalReuseOpts{
          .exploit_temporal_reuse=1,
          .multiple_loop_reuse=true
        }
      )
    );
  
  BOOST_CHECK(result.fill.map.is_equal(isl::map(
    GetIslCtx(),
    "{ [t1, x, t0] -> [d] : "
    "0 <= x < 2 and "
    "(((t1 = 0) and (t0 = 0) and (0 <= d < 3)) or "
    " ((t1 = 0) and (t0 = 1) and (d = 3)) or "
    " ((t1 = 1) and (t0 = 0) and (d = 0)) or "
    " ((t1 = 1) and (t0 = 1) and (d = 3))"
    ")}"
  )));
}