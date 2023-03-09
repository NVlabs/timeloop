#pragma once

#include <mutex>
#include <isl/cpp.h>

isl::ctx& GetIslCtx();
const std::unique_lock<std::mutex> GetBarvinokLock();