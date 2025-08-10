#pragma once
#include "enet_runtime_inline.h"
#include "enet_runtime_threaded.h"
#include "enet_types.h"

// Compile-time selection via -DENET_RUNTIME_THREADED
#if defined(ENET_RUNTIME_THREADED)
using EnetRuntime = EnetRuntimeThreaded;
#else
using EnetRuntime = EnetRuntimeInline;
#endif
