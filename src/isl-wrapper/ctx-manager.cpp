#include "isl-wrapper/ctx-manager.hpp"

#include <iostream>
#include <optional>
#include <thread>

/******************************************************************************
 * Local declarations
 *****************************************************************************/

thread_local std::optional<isl::ctx> gCtx;
std::mutex gIslMutex;

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

isl::ctx& GetIslCtx()
{
  if (!gCtx)
  {
    gCtx = isl::ctx(isl_ctx_alloc());
  }
  return *gCtx;
}

std::mutex& GetIslMutex()
{
  return gIslMutex;
}