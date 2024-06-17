#include "loop-analysis/spatial-analysis.hpp"

#include <boost/range/adaptor/reversed.hpp>
#include <isl/cpp.h>
#include <isl/map.h>
#include <isl/polynomial_type.h>
#include <barvinok/isl.h>

#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

isl::map MakeMeshConnectivity(size_t num_spatial_dims);

size_t GetNumTrailingSpatialTags(const std::vector<SpaceTime>& tags);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

FillProvider::FillProvider(const LogicalBuffer& buf, const Occupancy& occ) : 
  buf(buf), occupancy(occ)
{
}


SpatialReuseAnalysisInput::SpatialReuseAnalysisInput(
  LogicalBuffer buf,
  const Fill& children_fill,
  bool count_hops
) : buf(buf), children_fill(children_fill), count_hops(count_hops)
{
}

SpatialReuseInfo SpatialReuseAnalysis(const SpatialReuseAnalysisInput& input)
{
  auto transfer_infos = std::vector<TransferInfo>();

  for (const auto& fill_provider : input.fill_providers)
  {
    transfer_infos.emplace_back(
      fill_provider.spatial_reuse_model->Apply(input.children_fill,
                                               fill_provider.occupancy)
    );
  }

  return SpatialReuseInfo{.transfer_infos = std::move(transfer_infos)};
}


SimpleLinkTransferModel::SimpleLinkTransferModel()
{
}

TransferInfo SimpleLinkTransferModel::Apply(
  const Fill& fill,
  const Occupancy& occupancy
) const
{
  if (fill.dim_in_tags.size() != occupancy.dim_in_tags.size())
  {
    throw std::logic_error("fill and occupancy have different sizes");
  }

  auto n = fill.dim_in_tags.size();
  if (n == 0 || std::holds_alternative<Temporal>(fill.dim_in_tags.back()))
  {
    throw std::logic_error("unreachable");
  }

  auto n_spatial_dims = GetNumTrailingSpatialTags(fill.dim_in_tags);

  auto connectivity = MakeMeshConnectivity(n_spatial_dims);

  auto transfer_info = TransferInfo();
  transfer_info.is_link_transfer = true;

  if (n < n_spatial_dims+1) // i.e., no temporal loop. Cannot fulfill via link transfers
  {
    transfer_info.fulfilled_fill =
      Transfers(fill.dim_in_tags, fill.map.subtract(fill.map)); // empty map
    transfer_info.parent_reads = 
      Reads(occupancy.dim_in_tags,
            occupancy.map.subtract(occupancy.map)); // empty map
    transfer_info.unfulfilled_fill = fill;

    return transfer_info;
  }

  auto complete_connectivity =
    isl::insert_equal_dims(connectivity, 0, 0, n - n_spatial_dims - 1);
  auto available_from_neighbors =
    complete_connectivity.apply_range(occupancy.map);
  auto fill_set = fill.map.intersect(available_from_neighbors);
  auto remaining_fill = fill.map.subtract(fill_set);

  auto p_fill_set = isl_map_wrap(fill_set.copy());
  auto p_hops = isl_pw_qpolynomial_from_qpolynomial(
    isl_qpolynomial_one_on_domain(isl_set_get_space(p_fill_set))
  );
  p_hops = isl_pw_qpolynomial_intersect_domain(p_hops, p_fill_set);

  transfer_info.fulfilled_fill = Transfers(fill.dim_in_tags, fill_set);
  // TODO: fill parent_reads. Below is just (wrong!) placeholder to make
  // isl::map copy ctor happy.
  transfer_info.parent_reads= Reads(fill.dim_in_tags, fill_set);
  transfer_info.unfulfilled_fill = Fill(fill.dim_in_tags, remaining_fill);
  transfer_info.p_hops = p_hops;

  return transfer_info;
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

SimpleMulticastModel::SimpleMulticastModel(bool count_hops) :
  count_hops_(count_hops)
{
}

TransferInfo
SimpleMulticastModel::Apply(const Fill& fill, const Occupancy& occ) const
{
  auto transfer_info = TransferInfo();
  transfer_info.is_multicast = true;

  auto n = isl::dim(fill.map, isl_dim_in);
  if (n == 0 || std::holds_alternative<Temporal>(fill.dim_in_tags.back()))
  {
    throw std::logic_error("fill spacetime missing spatial dimensions");
  }

  auto n_spatial_dims = GetNumTrailingSpatialTags(fill.dim_in_tags);

  auto p_fill = fill.map.copy();

  // Creates [[t] -> [data]] -> [t, x, y] when n_spatial_dims == 2
  // Creates [[t] -> [data]] -> [t, x] when n_spatial_dims == 1
  auto p_wrapped_fill = isl_map_uncurry(isl_map_project_out(
    isl_map_reverse(isl_map_range_map(isl_map_reverse(p_fill))),
    isl_dim_in,
    n-n_spatial_dims,
    n_spatial_dims
  ));
  auto wrapped_fill = isl::manage(p_wrapped_fill);

  auto p_multicast_factor = isl_map_card(wrapped_fill.copy());

  isl_pw_qpolynomial* p_hops = nullptr;
  if (count_hops_)
  {
    isl_map* p_data_to_x = nullptr;
    if (n_spatial_dims == 2)
    {
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
      p_hops = p_y_hops;
      p_data_to_x = isl_map_lexmax(
        isl_map_project_out(wrapped_fill.copy(), isl_dim_out, n-1, 1)
      );
    }
    else
    {
      p_data_to_x = wrapped_fill.copy();
    }

    // Remove y, leaving only x
    auto p_x_hops_cost = isl_pw_qpolynomial_from_qpolynomial(
      isl_qpolynomial_add(
        isl_qpolynomial_var_on_domain(
          isl_space_range(isl_map_get_space(p_data_to_x)),
          isl_dim_set,
          n-n_spatial_dims
        ),
        isl_qpolynomial_one_on_domain(
          isl_space_range(isl_map_get_space(p_data_to_x))
        )
      )
    );
    auto p_x_hops = isl_map_apply_pw_qpolynomial(p_data_to_x,
                                                 p_x_hops_cost);

    if (p_hops == nullptr)
    {
      p_hops = p_x_hops;
    }
    else
    {
      p_hops = isl_pw_qpolynomial_add(p_hops, p_x_hops);
    }
  }
  else
  {
    p_hops = isl_pw_qpolynomial_from_qpolynomial(
      isl_qpolynomial_zero_on_domain(isl_pw_qpolynomial_get_domain_space(
        p_multicast_factor
      ))
    );
  }

  // auto accumulator = Accumulator();
  // accumulator.p_time_data_to_hops = p_hops;
  // isl_pw_qpolynomial_foreach_piece(p_multicast_factor,
  //                                  &ComputeMulticastScatterHops,
  //                                  static_cast<void*>(&accumulator));

  // isl_pw_qpolynomial_free(p_multicast_factor);

  // for (const auto& [multicast, hops_accesses] :
  //       accumulator.multicast_to_hops_accesses)
  // {
  //   auto& stats =
  //     transfer_info.compat_access_stats[std::make_pair(multicast, 1)];
  //   stats.accesses = hops_accesses.accesses;
  //   stats.hops = hops_accesses.hops / hops_accesses.accesses;
  // }
  // Remove y, leaving only x

  auto total_accesses = isl::val_to_double(isl_qpolynomial_get_constant_val(
    isl_pw_qpolynomial_as_qpolynomial(
      isl_set_card(isl_pw_qpolynomial_domain(
        isl_pw_qpolynomial_copy(p_multicast_factor)
      ))
    )
  ));

  auto p_total_hops = isl_pw_qpolynomial_sum(isl_pw_qpolynomial_sum(
    isl_pw_qpolynomial_copy(p_hops)
  ));
  auto total_hops = isl::val_to_double(isl_qpolynomial_get_constant_val(
    isl_pw_qpolynomial_as_qpolynomial(p_total_hops)
  ));

  auto p_total_multicast = isl_pw_qpolynomial_sum(isl_pw_qpolynomial_sum(
    p_multicast_factor
  ));
  auto total_multicast = isl::val_to_double(isl_qpolynomial_get_constant_val(
    isl_pw_qpolynomial_as_qpolynomial(p_total_multicast)
  ));

  auto avg_multicast = total_multicast / total_accesses;
  auto avg_hops = total_hops / total_accesses;

  auto& stats = transfer_info.compat_access_stats[std::make_pair(
    avg_multicast,
    1
  )];
  stats.accesses = total_accesses;
  stats.hops = avg_hops;

  transfer_info.fulfilled_fill = Transfers(fill.dim_in_tags, fill.map);
  // TODO: this assumes no bypassing
  transfer_info.parent_reads = Reads(
    occ.dim_in_tags,
    isl::project_dim(fill.map, isl_dim_in, n-n_spatial_dims, n_spatial_dims)
  );
  transfer_info.unfulfilled_fill = Fill(
    fill.dim_in_tags,
    fill.map.subtract(fill.map)  // empty map
  );
  transfer_info.p_hops = p_hops;

  return transfer_info;
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/


isl::map MakeMeshConnectivity(size_t num_spatial_dims)
{
  if (num_spatial_dims == 2)
  {
    return isl::map(GetIslCtx(),
                    "{ [t, x, y] -> [t-1, x', y'] : "
                    " (y'=y and x'=x-1) or (y'=y and x'=x+1) "
                    " or (x'=x and y'=y-1) or (x'=x and y'=y+1) }");
  }
  else if (num_spatial_dims == 1)
  {
    return isl::map(GetIslCtx(),
                    "{ [t, x] -> [t-1, x'] : "
                    " (x'=x-1) or (x'=x+1) }");
  }

  auto err_msg = std::stringstream();
  err_msg << "Cannot make mesh with " << num_spatial_dims << "dims";
  throw std::logic_error(err_msg.str());
}

size_t GetNumTrailingSpatialTags(const std::vector<SpaceTime>& tags)
{
  size_t n_spatial_dims = 0;
  for (const auto& in_tags : tags | boost::adaptors::reversed)
  {
    if (std::holds_alternative<Spatial>(in_tags))
    {
      ++n_spatial_dims;
    }
    else
    {
      break;
    }
  }
  return n_spatial_dims;
}

} // namespace analysis