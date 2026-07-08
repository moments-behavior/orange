#include "ptp_master.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t g_ptp4l_pid = -1;
static pid_t g_phc2sys_pid = -1;

static const char *PTP4L_BIN = "/usr/sbin/ptp4l";
static const char *PHC2SYS_BIN = "/usr/sbin/phc2sys";

static pid_t spawn(const std::vector<std::string> &args) {
    pid_t pid = fork();
    if (pid == 0) {
        // Die with orange (even if it crashes) so we never leave a stray
        // grandmaster fighting a future instance.
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto &a : args)
            argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);
        execv(argv[0], argv.data());
        _exit(127);
    }
    return pid;
}

void start_ptp_master(const std::vector<std::string> &interfaces) {
    if (interfaces.empty())
        return;
    if (g_ptp4l_pid > 0)
        return; // already serving (callers may re-scan cameras repeatedly)
    if (geteuid() != 0) {
        fprintf(stderr,
                "WARN: not running as root — cannot start a PTP grandmaster. "
                "PTP-synced starts will fall back to free-running.\n");
        return;
    }
    if (access(PTP4L_BIN, X_OK) != 0) {
        fprintf(stderr,
                "WARN: %s not found — install linuxptp to serve PTP time to "
                "the cameras. PTP-synced starts will fall back to "
                "free-running.\n",
                PTP4L_BIN);
        return;
    }
    if (system("pgrep -x ptp4l >/dev/null 2>&1") == 0) {
        printf("PTP: ptp4l already running — using the existing grandmaster "
               "as-is.\n");
        return;
    }

    // Dedupe: multiple cameras can share a NIC port.
    std::set<std::string> unique(interfaces.begin(), interfaces.end());

    std::vector<std::string> ptp4l_args = {PTP4L_BIN};
    std::string iface_list;
    for (const auto &ifname : unique) {
        ptp4l_args.push_back("-i");
        ptp4l_args.push_back(ifname);
        iface_list += (iface_list.empty() ? "" : ", ") + ifname;
    }
    // masterOnly: never become a slave to anything on the camera segments.
    // boundary_clock_jbod: the quad-port NIC has one PHC per port; let one
    // ptp4l instance serve all ports anyway.
    ptp4l_args.push_back("--masterOnly=1");
    ptp4l_args.push_back("--boundary_clock_jbod=1");

    g_ptp4l_pid = spawn(ptp4l_args);
    printf("PTP: started ptp4l grandmaster (pid %d) on %s\n", g_ptp4l_pid,
           iface_list.c_str());

    // phc2sys aligns every port PHC to CLOCK_REALTIME (with -rr the system
    // clock is the source while all ports are masters). Without this the four
    // port clocks drift apart and a cross-port gate time is meaningless. Give
    // ptp4l a moment to create its UDS before phc2sys connects.
    g_phc2sys_pid = spawn({"/bin/sh", "-c",
                           std::string("sleep 2; exec ") + PHC2SYS_BIN +
                               " -a -rr"});
    printf("PTP: started phc2sys (pid %d) to align port clocks\n",
           g_phc2sys_pid);
}

void stop_ptp_master() {
    for (pid_t *pid : {&g_phc2sys_pid, &g_ptp4l_pid}) {
        if (*pid > 0) {
            kill(*pid, SIGTERM);
            waitpid(*pid, nullptr, 0);
            *pid = -1;
        }
    }
}
