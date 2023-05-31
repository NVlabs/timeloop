#include "loop-analysis/spatial-analysis.hpp"

#include "isl-wrapper/ctx-manager.hpp"
#include "isl/cpp.h"
#include "isl/map.h"
#include "barvinok/isl.h"
#include "isl/polynomial_type.h"

namespace analysis
{

MulticastInfo::~MulticastInfo()
{
  // for (auto& [_, p_hop] : p_hops)
  // {
  //   isl_pw_qpolynomial_free(p_hop);
  // }
}

SpatialReuseInfo
SpatialReuseAnalysis(LogicalBufFills& fills,
                     LogicalBufOccupancies& occupancies,
                     const LinkTransferModel& link_transfer_model,
                     const MulticastModel& multicast_model)
{
  auto link_transfer_info = link_transfer_model.Apply(fills, occupancies);
  auto multicast_info = multicast_model.Apply(
    link_transfer_info.unfulfilled_fills,
    occupancies
  );

  return SpatialReuseInfo{
    .link_transfer_info = std::move(link_transfer_info),
    .multicast_info = std::move(multicast_info)
  };
}


SimpleLinkTransferModel::SimpleLinkTransferModel(size_t n_spatial_dims) :
  n_spatial_dims_(n_spatial_dims)
{
  if (n_spatial_dims == 1)
  {
      connectivity_ = isl::map(GetIslCtx(),
                              "{ [t, x] -> [t-1, x'] : x'=x-1 or x'=x+1 }");
  }
  else if (n_spatial_dims == 2)
  {
      connectivity_ = isl::map(GetIslCtx(),
                              "{ [t, x, y] -> [t-1, x', y'] : "
                              "  (x'=x-1 or x'=x+1) "
                              "  and (y'=y-1 or y'=y+1) }");
  }
  else
  {
      throw std::logic_error("unsupported");
  }
}

LinkTransferInfo SimpleLinkTransferModel::Apply(
  LogicalBufFills& fills,
  LogicalBufOccupancies& occupancies
) const
{
  LogicalBufTransfers transfers;
  LogicalBufFills remaining_fills;

  for (auto& [buf, fill] : fills)
  {
    auto n = fill.in_tags.size();
    if (n == 0 || fill.in_tags.back() == spacetime::Dimension::Time)
    {
      throw std::logic_error("unreachable");
    }
    else if (n < n_spatial_dims_ + 1)
    {
      remaining_fills.emplace(std::make_pair(buf, fill));
    }
    else
    {
      auto complete_connectivity =
        isl::insert_equal_dims(connectivity_, 0, 0, n - n_spatial_dims_ - 1);
      auto available_from_neighbors =
        complete_connectivity.apply_range(occupancies.at(buf).map);
      auto fill_set = fill.intersect(available_from_neighbors);
      auto remaining_fill = fill.subtract(fill_set.map);

      transfers.emplace(std::make_pair(std::make_pair(buf, buf), fill_set));
      remaining_fills.emplace(std::make_pair(buf, remaining_fill));
    }
  }

  return LinkTransferInfo{.link_transfers=std::move(transfers),
                          .unfulfilled_fills=std::move(remaining_fills)};
}

SimpleMulticastModel::SimpleMulticastModel(size_t n_spatial_dims) :
  n_spatial_dims_(n_spatial_dims) {}

MulticastInfo SimpleMulticastModel::Apply(
  LogicalBufFills& fills,
  LogicalBufOccupancies& occupancies
) const
{
  (void) occupancies;

  MulticastInfo multicast_info;

  if (n_spatial_dims_ == 1)
  {
    /**
    * Hops = \sum_{x}{x*|Fill(x) - \union_{x'>x}{Fill(x')}|}
    */
    for (auto& [buf, fill] : fills)
    {
      auto n = fill.dim(isl_dim_in);
      if (n == 0 || fill.in_tags.back() == spacetime::Dimension::Time)
      {
        throw std::logic_error("unreachable");
      }
      else
      {
        auto last_use = fill.subtract(
          map_to_all_after(fill.space().domain(), isl_dim_in, n-1)
            .apply_range(fill.map)
        );
        auto p_aff_cost = 
          isl_aff_zero_on_domain_space(fill.space().domain().release());
        p_aff_cost = isl_aff_set_constant_si(
          isl_aff_set_coefficient_si(p_aff_cost, isl_dim_in, n-1, 1),
          1
        );
        auto p_cost = isl_pw_qpolynomial_from_qpolynomial(
          isl_qpolynomial_from_aff(p_aff_cost)
        );
        auto p_count = isl_map_card(last_use.map.copy());
        auto p_hops = isl_pw_qpolynomial_mul(p_cost, p_count);

        auto p_coords = isl_map_from_multi_aff(
          isl_multi_aff_identity_on_domain_space(
            fill.map.space().domain().release()
          )
        );
        p_coords = isl_map_intersect_domain(p_coords,
                                            fill.map.domain().release());
        p_coords = isl_map_project_out(p_coords, isl_dim_in, n-1, 1);
        auto p_total_hops = isl_map_apply_pw_qpolynomial(p_coords, p_hops);

        multicast_info.reads.emplace(std::make_pair(
          LogicalBuffer(buf.buffer_id-1, buf.dspace_id, buf.branch_leaf_id),
          isl::project_last_dim(fill.map)
        ));
        multicast_info.p_hops.emplace(std::make_pair(
          LogicalBuffer(buf.buffer_id-1, buf.dspace_id, buf.branch_leaf_id),
          p_total_hops
        ));
      }
    }
  }
  else if (n_spatial_dims_ == 2)
  {
    /**
    * inject at (0, 0) along the y-axis first, then x-axis
    *   yfill(y) = \union_{x}{fill(x, y)}
    *   yhops = \sum_{y}{y*|yfill(y) - \union_{y'>y}{yfill(y')}|}
    *   xhops = \sum_{x, y}{x*|fill(x, y) - \union_{x'>x}{fill(x', y)}|}
    */
    for (auto& [buf, fill] : fills)
    {
      auto n = fill.dim(isl_dim_in);
      if (n == 0 || fill.in_tags.back() == spacetime::Dimension::Time)
      {
        throw std::logic_error("unreachable");
      }
      else
      {
        throw std::logic_error("unimplemented");
        auto y_fill = isl::project_dim(fill.map, isl_dim_in, n-1, 1);
        auto y_last_use = y_fill.subtract(
          map_to_all_after(y_fill.space().domain(), isl_dim_in, n-2)
            .apply_range(y_fill)
        );
        auto p_y_card = isl_map_card(y_last_use.release());
        auto p_y_aff_dist = 
          isl_aff_zero_on_domain_space(fill.space().domain().release());
        p_y_aff_dist = isl_aff_set_constant_si(
          isl_aff_set_coefficient_si(p_y_aff_dist, isl_dim_in, n-2, 1),
          1
        );
        auto p_y_dist = isl_pw_qpolynomial_from_qpolynomial(
          isl_qpolynomial_from_aff(p_y_aff_dist)
        );
        auto p_y_hops = isl_pw_qpolynomial_mul(p_y_dist, p_y_card);
        std::cout << "y hops: " << isl_pw_qpolynomial_to_str(p_y_hops) << std::endl;

        auto p_y_total_hops = isl_map_apply_pw_qpolynomial(
          isl::project_dim_in_after(fill.map, n-2).release(),
          p_y_hops
        );

        auto x_last_use = fill.subtract(
          map_to_all_after(fill.space().domain(), isl_dim_in, n-1)
            .apply_range(fill.map)
        );
        auto p_x_card = isl_map_card(x_last_use.map.copy());
        auto p_x_aff_dist = 
          isl_aff_zero_on_domain_space(fill.space().domain().release());
        p_x_aff_dist = isl_aff_set_constant_si(
          isl_aff_set_coefficient_si(p_x_aff_dist, isl_dim_in, n-1, 1),
          1
        );
        auto p_x_dist = isl_pw_qpolynomial_from_qpolynomial(
          isl_qpolynomial_from_aff(p_x_aff_dist)
        );
        auto p_x_hops = isl_pw_qpolynomial_mul(p_x_dist, p_x_card);
        std::cout << "x hops: " << isl_pw_qpolynomial_to_str(p_x_hops) << std::endl;

        auto p_x_total_hops = isl_map_apply_pw_qpolynomial(
          isl::project_dim_in_after(fill.map, n-2).release(),
          p_x_hops
        );

        multicast_info.reads.emplace(std::make_pair(
          LogicalBuffer(buf.buffer_id-1, buf.dspace_id, buf.branch_leaf_id),
          isl::project_last_dim(isl::project_last_dim(fill.map))
        ));
        multicast_info.p_hops.emplace(std::make_pair(
          LogicalBuffer(buf.buffer_id-1, buf.dspace_id, buf.branch_leaf_id),
          isl_pw_qpolynomial_add(p_y_total_hops, p_x_total_hops)
        ));
      }
    }
  }

  return multicast_info;
}

} // namespace analysis