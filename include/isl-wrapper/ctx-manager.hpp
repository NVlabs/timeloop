#pragma once

#include <mutex>
#include <isl/cpp.h>

isl::ctx& GetIslCtx();

std::mutex& GetIslMutex();
