#pragma once

#include <vector>
#include <stdexcept>
#include <ostream>

#include <isl/cpp.h>

#include "isl-wrapper/isl-functions.hpp"

/******************************************************************************
 * Macros
 *****************************************************************************/

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
    --dim_; return *this;
  }

  bool operator!=(const ReverseTaggedMapDimIterator& other)
  {
    return iterator_ != other.iterator_;
  }

 private:
  using BaseIter = typename std::vector<T>::reverse_iterator;

  BaseIter iterator_;
  std::size_t dim_;

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

std::ostream& operator<<(std::ostream& os, const NoTag& n);

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

  size_t dim(isl_dim_type dim_type) const
  {
    return isl_map_dim(map.get(), isl_dim_in);
  }

  inline TaggedMap<Map, InTag, NoTag>
  apply_range(const isl::map& other_map)
  {
    return TaggedMap(map.apply_range(other_map),
                     std::vector<InTag>(in_tags));
  }

  void project_dim_in(size_t pos, size_t n)
  {
    if (pos + n <= in_tags.size())
    {
      map = isl::project_dim(map, isl_dim_in, pos, n);
      in_tags.erase(in_tags.begin() + pos, in_tags.begin() + pos + n);
    }
    else
    {
      throw std::out_of_range("pos + n out of range");
    }
  }

  inline isl::space space() const
  {
    return map.space();
  }

  TaggedMap<MapT, InTag, OutTag> intersect(MapT other) const
  {
    return TaggedMap(map.intersect(other),
                     std::vector<InTag>(in_tags),
                     std::vector<OutTag>(out_tags));
  }

  TaggedMap<MapT, InTag, OutTag> subtract(MapT other) const
  {
    return TaggedMap(map.subtract(other),
                     std::vector<InTag>(in_tags),
                     std::vector<OutTag>(out_tags));
  }

  TaggedMap<MapT, InTag, OutTag> copy() const
  {
    return TaggedMap(*this);
  }
  TaggedMap<MapT, InTag, OutTag> tag_like_this(MapT&& map) const
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
};

template<typename MapT, typename InTagT, typename OutTagT>
std::ostream&
operator<<(std::ostream& os, const TaggedMap<MapT, InTagT, OutTagT>& map)
{
  os << "in: [";
  bool first_item = true;
  for (const auto& in_tag : map.in_tags)
  {
    if (!first_item)
    {
      os << ", ";
    }
    first_item = false;
    os << in_tag;
  }

  os << "]; out: [";
  first_item = true;
  for (const auto& out_tag : map.out_tags)
  {
    if (!first_item)
    {
      os << ", ";
    }
    first_item = false;
    os << out_tag;
  }

  os << "]; map: " << map.map;
  return os;
}
