#include "enet_fb_helpers.h"
#include "enet_runtime_select.h" // EnetRuntime + peers_snapshot(...)
#include "enet_types.h"          // PeerSnapshot, ENetAddress (+ enet.h)
#include <algorithm>
#include <cstdio>
#include <imgui.h>
#include <string>
#include <vector>

struct PeerRow {
    uint32_t pid;
    std::string name;
    std::string ip;
    uint16_t port;
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
        r.name = reg.get_name(ps.peer_id); // may be empty
        r.ip = enet_ip_string(
            ps.addr); // helper that calls enet_address_get_host_ip
        r.port = ps.addr.port;
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

static void ConnectButtons(EnetRuntime &net, uint16_t port = 3333,
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

void draw_peers_window(EnetRuntime &net, const PeerRegistry &reg) {
    std::vector<PeerRow> rows;
    BuildPeerRows(net, reg, rows);

    ImGui::Begin("Connected Peers");

    // Connect buttons row
    ConnectButtons(net, /*port=*/3333, /*channels=*/2);
    ImGui::Separator();

    const ImGuiTableFlags flags =
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
        ImGuiTableFlags_SizingStretchProp;

    if (ImGui::BeginTable("peers_table", 4, flags)) {
        ImGui::TableSetupColumn("Peer ID", ImGuiTableColumnFlags_DefaultSort);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("IP");
        ImGui::TableSetupColumn("Port");
        ImGui::TableHeadersRow();

        // Optional: sort rows by the active column
        if (ImGuiTableSortSpecs *sort = ImGui::TableGetSortSpecs();
            sort && sort->SpecsCount > 0) {
            const ImGuiTableColumnSortSpecs &s = sort->Specs[0];
            auto asc = (s.SortDirection == ImGuiSortDirection_Ascending);
            switch (s.ColumnIndex) {
            case 0:
                std::sort(rows.begin(), rows.end(),
                          [&](const PeerRow &a, const PeerRow &b) {
                              return asc ? a.pid < b.pid : a.pid > b.pid;
                          });
                break;
            case 1:
                std::sort(rows.begin(), rows.end(),
                          [&](const PeerRow &a, const PeerRow &b) {
                              return asc ? a.name < b.name : a.name > b.name;
                          });
                break;
            case 2:
                std::sort(rows.begin(), rows.end(),
                          [&](const PeerRow &a, const PeerRow &b) {
                              return asc ? a.ip < b.ip : a.ip > b.ip;
                          });
                break;
            case 3:
                std::sort(rows.begin(), rows.end(),
                          [&](const PeerRow &a, const PeerRow &b) {
                              return asc ? a.port < b.port : a.port > b.port;
                          });
                break;
            }
        }

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
        }

        ImGui::EndTable();
    }

    ImGui::End();
}
