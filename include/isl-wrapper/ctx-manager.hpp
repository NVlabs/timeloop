#pragma once

#include <mutex>

#include "isl-wrapper/isl-wrapper.hpp"

IslCtx& GetIslCtx();
const std::unique_lock<std::mutex> GetBarvinokLock();