#include <boost/test/unit_test.hpp>

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
