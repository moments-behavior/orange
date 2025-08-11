#include "enet_fb_helpers.h"
#include "enet_runtime_select.h"
#include "enet_types.h"
#include "fetch_generated.h"
#include "utils.h"
#include <ImGuiFileDialog.h>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <vector>

inline const char *ManagerStateName(int v) {
    const char *const *names = FetchGame::EnumNamesManagerState();
    if (v < 0)
        return "Unknown";
    for (int i = 0; names[i] != nullptr; ++i)
        if (i == v)
            return names[i];
    return "Unknown";
}

struct PeerRow {
    uint32_t pid;
    std::string name, ip;
    uint16_t port;
    bool camsK;
    int cameras;
    bool stateK;
    int state;
    std::string state_label;
};

inline std::string enet_ip_string(const ENetAddress &a) {
    char buf[64] = {0};
    ENetAddress tmp = a; // API takes non-const*
    if (enet_address_get_host_ip(&tmp, buf, sizeof(buf)) != 0)
        return {};
    return std::string(buf);
}

static void BuildPeerRows(EnetRuntime &net, const PeerRegistry &reg,
                          std::vector<PeerRow> &out) {
    std::vector<PeerSnapshot> snaps;
    net.peers_snapshot(snaps);
    out.clear();
    out.reserve(snaps.size());
    for (const auto &ps : snaps) {
        PeerRow r;
        r.pid = ps.peer_id;
        r.name = reg.get_name(ps.peer_id);
        r.ip = enet_ip_string(ps.addr);
        r.port = ps.addr.port;
        r.camsK = reg.cameras_known(ps.peer_id, &r.cameras);
        r.stateK = reg.state_known(ps.peer_id, &r.state);
        r.state_label = r.stateK ? ManagerStateName(r.state) : "N/A";
        out.push_back(std::move(r));
    }
}

static bool is_connected_to(EnetRuntime &net, const char *ip, uint16_t port) {
    std::vector<PeerSnapshot> snap;
    net.peers_snapshot(snap);
    for (const auto &ps : snap) {
        if (enet_ip_string(ps.addr) == ip && ps.addr.port == port)
            return true;
    }
    return false;
}

static void connect_buttons(EnetRuntime &net, uint16_t port = 3333,
                            size_t channels = 2) {
    const char *ipA = "192.168.20.60";
    const char *ipB = "192.168.20.61";
    // Button A
    bool haveA = is_connected_to(net, ipA, port);
    ImGui::BeginDisabled(haveA);
    if (ImGui::Button("Connect 192.168.20.60")) {
        ConnectReq cr{ipA, port, channels, 0};
#ifdef ENET_RUNTIME_THREADED
        net.connect(std::move(cr)); // queued; result via Connect event
#else
        if (!net.connect(cr))
            std::fprintf(stderr, "connect(%s:%u) failed\n", ipA, port);
#endif
    }
    ImGui::EndDisabled();

    ImGui::SameLine();

    // Button B
    bool haveB = is_connected_to(net, ipB, port);
    ImGui::BeginDisabled(haveB);
    if (ImGui::Button("Connect 192.168.20.61")) {
        ConnectReq cr{ipB, port, channels, 0};
#ifdef ENET_RUNTIME_THREADED
        net.connect(std::move(cr));
#else
        if (!net.connect(cr))
            std::fprintf(stderr, "connect(%s:%u) failed\n", ipB, port);
#endif
    }
    ImGui::EndDisabled();
}

using OpenCamerasFn = std::function<void(const std::string &cfg_folder)>;

inline void draw_camera_open(const PeerRegistry &reg,
                             CameraControl *camera_control,
                             const std::vector<std::string> &server_names,
                             const std::string &selected_cfg_folder,
                             OpenCamerasFn on_open_cameras,
                             std::string &input_folder) {
    auto all_connected = [&]() {
        for (const auto &name : server_names)
            if (reg.get_pid_by_name(name) == 0)
                return false;
        return true;
    };
    auto all_idle = [&]() {
        for (const auto &name : server_names) {
            uint32_t pid = reg.get_pid_by_name(name);
            if (!pid)
                return false;
            int st = 0;
            if (!reg.state_known(pid, &st))
                return false;
            if (st != (int)FetchGame::ManagerState_IDLE)
                return false;
        }
        return true;
    };

    const bool can_open = (!camera_control->open) && all_connected() &&
                          all_idle() && !selected_cfg_folder.empty();

    // Open Cameras
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.0f, 0.5f, 0.0f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4{0.2f, 0.8f, 0.2f, 1.0f});
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4{0.1f, 0.6f, 0.1f, 1.0f});
    ImGui::BeginDisabled(!can_open);
    if (ImGui::Button("Open Cameras")) {
        on_open_cameras(selected_cfg_folder);
        camera_control->open = true;
    }
    ImGui::EndDisabled();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    // Save to…
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.0f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(0.7f, 0.2f, 0.9f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4(0.4f, 0.0f, 0.6f, 1.0f));
    if (ImGui::Button("Save to")) {
        IGFD::FileDialogConfig cfg;
        cfg.countSelectionMax = 1;
        cfg.path = input_folder;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseRecordingDir", "Choose a Directory", nullptr, cfg);
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    ImGui::SetWindowFontScale(1.5f);
    ImGui::Text("%s", input_folder.c_str());
    ImGui::SetWindowFontScale(1.0f);
}

inline void draw_peers_window(EnetRuntime &net, const PeerRegistry &peers) {
    std::vector<PeerRow> rows;
    BuildPeerRows(net, peers, rows);

    // Connect buttons row
    connect_buttons(net, /*port=*/3333, /*channels=*/2);
    ImGui::Separator();

    const ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
        ImGuiTableFlags_SizingStretchProp;

    if (ImGui::BeginTable("peers", 6, flags)) {
        ImGui::TableSetupColumn("Peer ID");
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("IP");
        ImGui::TableSetupColumn("Port");
        ImGui::TableSetupColumn("Cameras");
        ImGui::TableSetupColumn("Server State");
        ImGui::TableHeadersRow();

        for (const auto &r : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%u", r.pid);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(r.name.empty() ? "(unnamed)"
                                                  : r.name.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(r.ip.c_str());
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%u", r.port);
            ImGui::TableSetColumnIndex(4);
            r.camsK ? ImGui::Text("%d", r.cameras)
                    : ImGui::TextUnformatted("N/A");
            ImGui::TableSetColumnIndex(5);
            ImGui::TextUnformatted(r.state_label.c_str());
        }
        ImGui::EndTable();
    }
}
