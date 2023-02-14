#include "isl-wrapper/ctx-manager.hpp"

#include <optional>

/******************************************************************************
 * Local declarations
 *****************************************************************************/

thread_local IslCtx gCtx;
std::mutex gIslMutex;

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

IslCtx& GetIslCtx()
{
  return gCtx;
}

const std::unique_lock<std::mutex> GetIslLock()
{
  return std::unique_lock(gIslMutex, std::try_to_lock);
}