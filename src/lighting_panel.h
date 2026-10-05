#ifndef ORANGE_LIGHTING_PANEL
#define ORANGE_LIGHTING_PANEL

#include "lighting_control.h"
#include <thread>

// Renders the "Lighting" ImGui window (day/night cycle controls + a CCT
// curve preview) and drives start/stop of the lighting thread. Call once per
// frame between ImGui::NewFrame() and ImGui::Render().
//
// Shared verbatim between orange and the standalone Windows lighting
// prototype (see OrangeLightingPreview) so the panel only needs to be built
// and iterated on once, then dropped into orange unchanged.
void render_lighting_panel(LightingConfig &config, std::thread &thread_ref);

#endif
