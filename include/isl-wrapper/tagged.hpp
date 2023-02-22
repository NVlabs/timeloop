#pragma once

#include <vector>
#include <stdexcept>

#include <isl/space.h>

/******************************************************************************
 * Macros
 *****************************************************************************/
#define ISL_TAGGED_MAP_BINARY_OP_IMPL(NAME)                  \
  template<typename MapT, typename InTagT, typename OutTagT> \
  TaggedMap<MapT, InTagT, OutTagT>                           \
  NAME(TaggedMap<MapT, InTagT, OutTagT>&& map1,              \
       MapT&& map2)                                          \
  {                                                          \
    return TaggedMap<MapT, InTagT, OutTagT>(                 \
      NAME(std::move(map1.map), std::move(map2)),            \
      std::move(map1.in_tags),                               \
      std::move(map1.out_tags)                               \
    );                                                       \
  }                                                          \
  template<typename MapT, typename InTagT, typename OutTagT> \
  TaggedMap<MapT, InTagT, OutTagT>                           \
  NAME(TaggedMap<MapT, InTagT, OutTagT>&& map1,              \
       TaggedMap<MapT, InTagT, OutTagT>&& map2)              \
  {                                                          \
    return TaggedMap<MapT, InTagT, OutTagT>(                 \
      NAME(std::move(map1.map), std::move(map2.map)),        \
      std::move(map1.in_tags),                               \
      std::move(map1.out_tags)                               \
    );                                                       \
  }

/******************************************************************************
 * Classes
 *****************************************************************************/

template<typename T>
class TaggedMapDimIterator
{
 public:
  std::pair<std::size_t, T> operator*()
  {
    return std::make_pair(dim_, *iterator_);
  }

  TaggedMapDimIterator<T>& operator++()
  {
    ++iterator_;
    ++dim_;
    return *this;
  }

  bool operator!=(const TaggedMapDimIterator& other)
  {
    return iterator_ != other.iterator_;
  }

 private:
  using BaseIter = typename std::vector<T>::iterator;

  BaseIter iterator_;
  std::size_t dim_;
  std::size_t size_;

  TaggedMapDimIterator(const BaseIter& iter,
                       std::size_t start)
  : iterator_(iter), dim_(start) {}

  static TaggedMapDimIterator start(std::vector<T>& vec)
  {
    return TaggedMapDimIterator(vec.begin(), 0UL);
  }

  static TaggedMapDimIterator end(std::vector<T>& vec)
  {
    return TaggedMapDimIterator(vec.end(), 0UL);
  }

  template<typename MapT, typename InTagT, typename OutTagT>
  friend class TaggedMap;
};

template<typename T>
class ReverseTaggedMapDimIterator
{
 public:
  std::pair<std::size_t, T> operator*()
  {
    return std::make_pair(dim_, *iterator_);
  }

  ReverseTaggedMapDimIterator<T>& operator++()
  {
    ++iterator_;
    --dim_;
    return *this;
  }

  bool operator!=(const ReverseTaggedMapDimIterator& other)
  {
    return iterator_ != other.iterator_;
  }

 private:
  using BaseIter = typename std::vector<T>::reverse_iterator;

  BaseIter iterator_;
  std::size_t dim_;
  std::size_t size_;

  ReverseTaggedMapDimIterator(const BaseIter& iter,
                              std::size_t start)
  : iterator_(iter), dim_(start) {}

  static ReverseTaggedMapDimIterator start(std::vector<T>& vec)
  {
    return ReverseTaggedMapDimIterator(vec.rbegin(), vec.size()-1);
  }

  static ReverseTaggedMapDimIterator end(std::vector<T>& vec)
  {
    return ReverseTaggedMapDimIterator(vec.rend(), 0UL);
  }

  template<typename MapT, typename InTagT, typename OutTagT>
  friend class TaggedMap;
};

struct NoTag
{
};

template<typename MapT, typename InTagT, typename OutTagT = NoTag>
struct TaggedMap
{
  using Map    = MapT;
  using InTag  = InTagT;
  using OutTag = OutTagT;

  Map map;
  std::vector<InTag> in_tags;
  std::vector<OutTag> out_tags;

  TaggedMap(Map&& map,
            std::vector<InTag>&& in_tags,
            std::vector<OutTag>&& out_tags)
    : map(std::move(map)), in_tags(std::move(in_tags)),
      out_tags(std::move(out_tags))
  {
  }
  TaggedMap(Map&& map,
            std::vector<InTag>&& in_tags)
    : map(std::move(map)), in_tags(std::move(in_tags)),
      out_tags()
  {
  }
  TaggedMap(const TaggedMap<MapT, InTagT, OutTagT>& other)
    : map(other.map), in_tags(other.in_tags), out_tags(other.out_tags)
  {
  }

  TaggedMap<MapT, InTagT, OutTagT>&
  operator=(TaggedMap<MapT, InTagT, OutTagT>&& other)
  {
    map = std::move(other.map);
    in_tags = std::move(other.in_tags);
    out_tags = std::move(other.out_tags);
    return *this;
  }

  TaggedMap<MapT, InTag, OutTag> Copy() const
  {
    return TaggedMap(*this);
  }
  TaggedMap<MapT, InTag, OutTag> TagLikeThis(MapT&& map) const
  {
    return TaggedMap(std::move(map),
                     std::vector<InTag>(in_tags),
                     std::vector<OutTag>(out_tags));
  }

  TaggedMapDimIterator<InTag> in_begin()
  {
    return TaggedMapDimIterator<InTag>::start(in_tags);
  }
  TaggedMapDimIterator<InTag> in_end()
  {
    return TaggedMapDimIterator<InTag>::end(in_tags);
  }
  ReverseTaggedMapDimIterator<InTag> in_rbegin()
  {
    return ReverseTaggedMapDimIterator<InTag>::start(in_tags);
  }
  ReverseTaggedMapDimIterator<InTag> in_rend()
  {
    return ReverseTaggedMapDimIterator<InTag>::end(in_tags);
  }

  TaggedMapDimIterator<OutTag> out_begin()
  {
    return TaggedMapDimIterator<OutTag>::start(out_tags);
  }
  TaggedMapDimIterator<OutTag> out_end()
  {
    return TaggedMapDimIterator<OutTag>::end(out_tags);
  }
  ReverseTaggedMapDimIterator<OutTag> out_rbegin()
  {
    return ReverseTaggedMapDimIterator<OutTag>::start(out_tags);
  }
  ReverseTaggedMapDimIterator<OutTag> out_rend()
  {
    return ReverseTaggedMapDimIterator<OutTag>::end(out_tags);
  }


  bool InvolvesDims(isl_dim_type dim_type,
                    std::size_t first,
                    std::size_t n) const
  {
    return map.InvolvesDims(dim_type, first, n);
  }
  std::size_t NumDims(isl_dim_type dim_type) const
  {
    return map.NumDims(dim_type);
  }
};

template<typename MapT, typename InTagT, typename OutTagT>
std::ostream&
operator<<(std::ostream& os, const TaggedMap<MapT, InTagT, OutTagT>& map)
{
  os << map.map;
  return os;
}

ISL_TAGGED_MAP_BINARY_OP_IMPL(ApplyRange)
ISL_TAGGED_MAP_BINARY_OP_IMPL(Subtract)

template<typename MapT, typename InTagT, typename OutTagT>
TaggedMap<MapT, InTagT, OutTagT>
ProjectDims(TaggedMap<MapT, InTagT, OutTagT>&& map,
            isl_dim_type dim_type,
            std::size_t first,
            std::size_t n)
{
  if (dim_type == isl_dim_in)
  {
    auto first_it = map.in_tags.begin() + first;
    auto last_it = map.in_tags.begin() + first + n;
    map.in_tags.erase(first_it, last_it);
  }
  else if (dim_type == isl_dim_out)
  {
    auto first_it = map.out_tags.begin() + first;
    auto last_it = map.out_tags.begin() + first + n;
    map.out_tags.erase(first_it, last_it);
  }
  else
  {
    throw std::logic_error("unimplemented");
  }

  return TaggedMap<MapT, InTagT, OutTagT>(
    ProjectDims(std::move(map.map), dim_type, first, n),
    std::move(map.in_tags),
    std::move(map.out_tags)
  );
}
