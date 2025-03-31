#include <limits>

#include <boost/test/unit_test.hpp>
namespace ut = boost::unit_test;
namespace tt = boost::test_tools;

#include <barvinok/isl.h>

#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

BOOST_AUTO_TEST_CASE(TestIslFunctions_aff_from_qpolynomial)
{
  isl_qpolynomial* p_qp = isl_pw_qpolynomial_as_qpolynomial(
    isl_pw_qpolynomial_read_from_str(
      GetIslCtx().get(),
      "{ [x] -> 2*x }"
    )
  );

  isl_aff* result = isl::aff_from_qpolynomial(p_qp);

  BOOST_CHECK(
    isl_aff_plain_is_equal(
      result,
      isl_aff_read_from_str(GetIslCtx().get(), "{ [x] -> [(2x)] }")
    )
  );

  p_qp = isl_pw_qpolynomial_as_qpolynomial(
    isl_pw_qpolynomial_read_from_str(
      GetIslCtx().get(),
      "{ [x] -> x*x }"
    )
  );

  result = isl::aff_from_qpolynomial(p_qp);

  BOOST_CHECK(result == nullptr);
}


BOOST_AUTO_TEST_CASE(TestIslFunctions_val_to_double,
                     * ut::tolerance(tt::fpc::percent_tolerance(0.01)))
{
  isl_ctx* ctx = GetIslCtx().get();
  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  isl_val* val = isl_val_int_from_ui(ctx, max_val);
  double double_val = max_val;

  auto res = isl::val_to_double(val);
  BOOST_TEST(res == double_val);

  double_val = (uint64_t)max_val + 1;
  val = isl_val_add_ui(val, 1);
  BOOST_TEST(res == double_val);
}
