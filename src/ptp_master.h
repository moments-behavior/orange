#ifndef ORANGE_PTP_MASTER
#define ORANGE_PTP_MASTER

#include <string>
#include <vector>

// Emergent cameras are PTP slave-only (PtpStatus: Disabled/Listening/
// Calibrating/Slave) — something on the host must serve PTP time on every
// NIC port that has cameras, or the cameras never lock and gated PTP starts
// hang. start_ptp_master() spawns a linuxptp grandmaster for the given host
// interfaces (as reported by the Emergent SDK per camera):
//   - ptp4l on all interfaces, masterOnly, boundary_clock_jbod (the quad-port
//     NIC exposes a separate PHC per port)
//   - phc2sys -a -rr to discipline every port PHC from CLOCK_REALTIME, so all
//     ports (and therefore all cameras) share one time base and a cross-port
//     PtpAcquisitionGateTime is valid.
// No-op if not root, if linuxptp is missing, or if a ptp4l is already running
// (an externally managed grandmaster is left alone). Children are killed by
// stop_ptp_master() and also die with the parent (PR_SET_PDEATHSIG).
void start_ptp_master(const std::vector<std::string> &interfaces);
void stop_ptp_master();

#endif // ORANGE_PTP_MASTER
