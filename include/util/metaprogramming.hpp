#pragma once

template<typename T, typename... Types>
constexpr bool IsAnyOfV = false;

template<typename T, typename Type0, typename... Types>
constexpr bool IsAnyOfV<T, Type0, Types...> =
  std::is_same_v<T, Type0> || IsAnyOfV<T, Types...>;

template<typename T>
constexpr bool IsAnyOfV<T> = false;
