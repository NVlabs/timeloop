#include "isl-wrapper/ctx-manager.hpp"

#include <optional>

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

const std::unique_lock<std::mutex> GetIslLock()
{
  return std::unique_lock(gIslMutex, std::try_to_lock);
}