#include "isl-wrapper/ctx-manager.hpp"

#include <optional>

/******************************************************************************
 * Local declarations
 *****************************************************************************/

thread_local std::optional<IslCtx> gCtx;
std::mutex gIslMutex;

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

IslCtx& GetIslCtx()
{
  if (!gCtx)
  {
    std::lock_guard<std::mutex> lock(gIslMutex);
    gCtx = IslCtx();
  }
  return *gCtx;
}

const std::unique_lock<std::mutex> GetBarvinokLock()
{
  return std::unique_lock(gIslMutex, std::try_to_lock);
}