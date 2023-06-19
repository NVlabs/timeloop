#include "loop-analysis/spatial-analysis.hpp"

#include <isl/cpp.h>
#include <isl/map.h>
#include <isl/polynomial_type.h>
#include <barvinok/isl.h>

#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{

MulticastInfo::~MulticastInfo()
{
  // for (auto& [_, p_hop] : p_hops)
  // {
  //   isl_pw_qpolynomial_free(p_hop);
  // }
}


SpatialReuseAnalysisInput::SpatialReuseAnalysisInput(
  const LogicalBuffer& buf,
  const Fill& fill,
  const Occupancy& occupancy
) :
  buf(buf), children_fill(fill), children_occupancy(occupancy)
{
}


LinkTransferModel& SpatialReuseModels::GetLinkTransferModel()
{
  return *link_transfer_model;
}
const LinkTransferModel& SpatialReuseModels::GetLinkTransferModel() const
{
  return *link_transfer_model;
}

MulticastModel& SpatialReuseModels::GetMulticastModel()
{
  return *multicast_model;
}
const MulticastModel& SpatialReuseModels::GetMulticastModel() const
{
  return *multicast_model;
}


SpatialReuseInfo SpatialReuseAnalysis(const SpatialReuseAnalysisInput& input,
                                      const SpatialReuseModels& models)
{
  auto link_transfer_info =
    models.GetLinkTransferModel().Apply(input.children_fill,
                                        input.children_occupancy);

  auto multicast_info = models.GetMulticastModel().Apply(
    link_transfer_info.unfulfilled_fill
  );

  return SpatialReuseInfo{
    .link_transfer_info = std::move(link_transfer_info),
    .multicast_info = std::move(multicast_info)
  };
}


SimpleLinkTransferModel::SimpleLinkTransferModel()
{
  connectivity_ = isl::map(GetIslCtx(),
                          "{ [t, x, y] -> [t-1, x', y'] : "
                          " (y'=y and x'=x-1) or (y'=y and x'=x+1) "
                          " or (x'=x and y'=y-1) or (x'=x and y'=y+1) }");
}

LinkTransferInfo SimpleLinkTransferModel::Apply(
  const Fill& fill,
  const Occupancy& occupancy
) const
{
  if (fill.dim_in_tags.size() != occupancy.dim_in_tags.size())
  {
    throw std::logic_error("fill and occupancy have different sizes");
  }

  auto n = fill.dim_in_tags.size();
  if (n == 0 || fill.dim_in_tags.back() == spacetime::Dimension::Time)
  {
    throw std::logic_error("unreachable");
  }

  if (n < 3)
  {
    return LinkTransferInfo{
      .link_transfer=Transfers(fill.dim_in_tags, fill.map.subtract(fill.map)),
      .unfulfilled_fill=fill
    };
  }

  auto complete_connectivity =
    isl::insert_equal_dims(connectivity_, 0, 0, n - 3);
  std::cout << complete_connectivity << std::endl;
  std::cout << occupancy.map << std::endl;
  auto available_from_neighbors =
    complete_connectivity.apply_range(occupancy.map);
  auto fill_set = fill.map.intersect(available_from_neighbors);
  auto remaining_fill = fill.map.subtract(fill_set);

  return LinkTransferInfo{
    .link_transfer=Transfers(fill.dim_in_tags, fill_set),
    .unfulfilled_fill=Fill(fill.dim_in_tags, remaining_fill)
  };
}


struct HopsAccesses
{
  double hops;
  double accesses;

  /**
   * @brief Simple weighted average for hops and accumulation for accesses.
   */
  void InsertHopsAccesses(double extra_hops, double extra_accesses)
  {
    accesses += extra_accesses;
    hops += extra_hops;
  }
};

struct Accumulator
{
  std::map<uint64_t, HopsAccesses> multicast_to_hops_accesses;
  isl_pw_qpolynomial* p_time_data_to_hops;
};

/**
 * @brief Accumulates scatter, hops, and accesses for many multicast factors.
 *
 * @param p_domain A set with signature $\{ [st_{n-1},t_n] -> data \}$ where
 *                 data is some set of data.
 * @param p_multicast_factor A qpolynomial assumed to be constant that equals
 *                           the multicast factor.
 * @param p_voided_accumulator Voided pointer that is cast into Accumulator.
 */
isl_stat ComputeMulticastScatterHops(isl_set* p_domain,
                                     isl_qpolynomial* p_multicast_factor,
                                     void* p_voided_accumulator)
{
  auto& accumulator = *static_cast<Accumulator*>(p_voided_accumulator);
  // WARNING: assumes constant multicast factor over piecewise domain.
  // It is unclear what conditions may cause this to break.
  auto multicast_factor = isl::val_to_double(
    isl_qpolynomial_eval(p_multicast_factor,
                         isl_set_sample_point(isl_set_copy(p_domain)))
  );
  std::cout << "multicast factor: " << multicast_factor << std::endl;
  std::cout << "domain: " << isl_set_to_str(p_domain) << std::endl;
  auto& hops_accesses = accumulator.multicast_to_hops_accesses[multicast_factor];

  auto p_hops_pw_qp = isl_set_apply_pw_qpolynomial(
    isl_set_copy(p_domain),
    isl_pw_qpolynomial_copy(accumulator.p_time_data_to_hops)
  );
  if (isl_pw_qpolynomial_isa_qpolynomial(p_hops_pw_qp) == isl_bool_false)
  {
    throw std::runtime_error("accesses is not a single qpolynomial");
  }
  auto hops = isl::val_to_double(isl_qpolynomial_get_constant_val(
    isl_pw_qpolynomial_as_qpolynomial(p_hops_pw_qp)
  ));

  auto p_time_to_data = isl_set_unwrap(p_domain);
  auto p_accesses_pw_qp = isl_pw_qpolynomial_sum(isl_map_card(p_time_to_data));
  if (isl_pw_qpolynomial_isa_qpolynomial(p_accesses_pw_qp) == isl_bool_false)
  {
    throw std::runtime_error("accesses is not a single qpolynomial");
  }
  auto accesses = isl::val_to_double(isl_qpolynomial_get_constant_val(
    isl_pw_qpolynomial_as_qpolynomial(p_accesses_pw_qp)
  ));

  hops_accesses.InsertHopsAccesses(hops, accesses);

  return isl_stat_ok;
}

SimpleMulticastModel::SimpleMulticastModel()
{
}

MulticastInfo SimpleMulticastModel::Apply(const Fill& fill) const
{
  MulticastInfo multicast_info;

  auto n = isl::dim(fill.map, isl_dim_in);
  if (n == 0 || fill.dim_in_tags.back() == spacetime::Dimension::Time)
  {
    throw std::logic_error("fill spacetime missing spatial dimensions");
  }

  auto p_fill = fill.map.copy();
  auto p_wrapped_fill = isl_map_uncurry(isl_map_project_out(
    isl_map_reverse(isl_map_range_map(isl_map_reverse(p_fill))),
    isl_dim_in,
    n-2,
    2
  ));
  auto wrapped_fill = isl::manage(p_wrapped_fill);

  auto p_multicast_factor = isl_map_card(wrapped_fill.copy());
  std::cout << "multicast_qp: " << isl_pw_qpolynomial_to_str(p_multicast_factor) << std::endl;

  auto p_y_hops_cost = isl_pw_qpolynomial_from_qpolynomial(
    isl_qpolynomial_add(
      isl_qpolynomial_var_on_domain(
        isl_space_range(isl_map_get_space(wrapped_fill.get())),
        isl_dim_set,
        n-1
      ),
      isl_qpolynomial_one_on_domain(
        isl_space_range(isl_map_get_space(wrapped_fill.get()))
      )
    )
  );
  auto p_y_hops = isl_map_apply_pw_qpolynomial(wrapped_fill.copy(),
                                                p_y_hops_cost);

  // Remove y, leaving only x
  auto p_data_to_max_x = isl_map_lexmax(
    isl_map_project_out(wrapped_fill.copy(), isl_dim_out, n-1, 1)
  );
  auto p_x_hops_cost = isl_pw_qpolynomial_from_qpolynomial(
    isl_qpolynomial_add(
      isl_qpolynomial_var_on_domain(
        isl_space_range(isl_map_get_space(p_data_to_max_x)),
        isl_dim_set,
        n-2
      ),
      isl_qpolynomial_one_on_domain(
        isl_space_range(isl_map_get_space(p_data_to_max_x))
      )
    )
  );
  auto p_x_hops = isl_map_apply_pw_qpolynomial(p_data_to_max_x,
                                                p_x_hops_cost);

  auto p_hops = isl_pw_qpolynomial_add(p_y_hops, p_x_hops);

  auto accumulator = Accumulator();
  accumulator.p_time_data_to_hops = p_hops;
  isl_pw_qpolynomial_foreach_piece(p_multicast_factor,
                                    &ComputeMulticastScatterHops,
                                    static_cast<void*>(&accumulator));

  isl_pw_qpolynomial_free(p_multicast_factor);
  isl_pw_qpolynomial_free(accumulator.p_time_data_to_hops);

  for (const auto& [multicast, hops_accesses] :
        accumulator.multicast_to_hops_accesses)
  {
    auto& stats =
      multicast_info.compat_access_stats[std::make_pair(multicast, 1)];
    stats.accesses = hops_accesses.accesses;
    stats.hops = hops_accesses.hops / hops_accesses.accesses;
  }

  multicast_info.reads = Reads(
    fill.dim_in_tags,
    fill.map.subtract(fill.map)
  );

  return multicast_info;
}

} // namespace analysis