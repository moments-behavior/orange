#pragma once
#include <string>
#include <utility>
#include <vector>

// Forward-declare your ENet app context (from enet_utils.h)
struct AppContext;

/**
 * Initialize the host client module and connect to the given camera servers.
 * Call once at startup.
 */
void HostClient_Init(AppContext &ctx,
                     const std::vector<std::pair<std::string, int>> &endpoints);

/**
 * Pump ENet (optional, see HostClient_SetStepInTick), process replies, handle
 * timeouts/retries, and advance internal state. Call once per frame BEFORE
 * ImGui::NewFrame().
 */
void HostClient_Tick();

/**
 * Draw the Dear ImGui control panel (advance/resend/reset, logs, acks).
 * Call each frame between ImGui::NewFrame() and ImGui::Render().
 */
void HostClient_DrawImGui();

/**
 * If you already run ctx.net.step(...) in another thread, call this with false
 * BEFORE HostClient_Init() so HostClient_Tick() won't step ENet again.
 * Default: true (HostClient_Tick steps ENet itself).
 */
void HostClient_SetStepInTick(bool v);
