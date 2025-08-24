#pragma once
#include "enet_utils.h" // for AppContext
#include "host_ctx.h"
#include <string>
#include <utility>
#include <vector>
void HostClient_StartNetThread(AppContext &ctx); // starts dispatcher thread
void HostClient_StopNetThread();                 // joins thread on shutdown

void HostClient_SetStepInTick(bool v); // optional mode switch
void HostClient_Init(AppContext &ctx,
                     const std::vector<std::pair<std::string, int>> &endpoints);
void HostClient_Tick();
void HostClient_DrawImGui();
void HostClient_SetOpenCtx(
    HostOpenCtx *ctx); // if you use the open-phase context
