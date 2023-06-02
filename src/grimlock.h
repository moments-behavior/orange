#ifndef GRIMLOCK
#define GRIMLOCK
#include "utility.h"
#include <ur_rtde/robotiq_gripper.h>
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <ur_rtde/robotiq_gripper.h>
#include <iostream>
#include <string>
#include "types.h"

using namespace ur_rtde;
using namespace std::chrono;

struct ControlState {
    bool activate = false;
    int ball_idx = 0;
    bool simple_task = true;
    bool stop = false;
    bool jogging = false;
};

void printStatus(int Status) {
    switch (Status)
    {
    case RobotiqGripper::MOVING:
        std::cout << "moving";
        break;
    case RobotiqGripper::STOPPED_OUTER_OBJECT:
        std::cout << "outer object detected";
        break;
    case RobotiqGripper::STOPPED_INNER_OBJECT:
        std::cout << "inner object detected";
        break;
    case RobotiqGripper::AT_DEST:
        std::cout << "at destination";
        break;
    }
    std::cout << std::endl;
}

enum ActionPrimitives {   
    GoHome, GoWorksurface, OpenGripper, CloseGripper, StopJ, StopL, GrabBall, PlaceBallRamp, DropBall, InitiateJogging, JogUp, StopJogging, GetPose, PoseTrans, PickBallArena, PickRampArena
};

static const char *ActionPrimitivesStr[] = {"GoHome", "GoWorksurface", "OpenGripper", "CloseGripper", "StopJ", "StopL", "GrabBall", "PlaceBallRamp", "DropBall", "InitiateJogging", "JogUp", "StopJogging", "GetPose", "PoseTrans", "PickBallArena", "PickRampArena"};

void moveJ_async(RTDEControlInterface* grimlock_control, ControlState* control_state, std::vector<double>& joint_psn, double speed, double acceleration)
{

    std::cout << "before start: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    grimlock_control->moveJ(joint_psn, speed, acceleration, true);  


    // Wait for start of asynchronous operation
    while (grimlock_control->getAsyncOperationProgress() < 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Async movej started.. " << std::endl;

    while (grimlock_control->getAsyncOperationProgress() >= 0)
    {
        if (!control_state->stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  
        }
        else {
            grimlock_control->stopJ(1.0);
            std::cout << "Async path interrupted" << std::endl;
            return;
        }
    }
    std::cout << "end: " << grimlock_control->getAsyncOperationProgress() << std::endl;
}

void moveL_async(RTDEControlInterface* grimlock_control, ControlState* control_state, std::vector<double>& tcp, double speed, double acceleration)
{

    std::cout << "before start: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    grimlock_control->moveL(tcp, speed, acceleration, true);  


    // Wait for start of asynchronous operation
    while (grimlock_control->getAsyncOperationProgress() < 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Async movej started.. " << std::endl;

    while (grimlock_control->getAsyncOperationProgress() >= 0)
    {
        if (!control_state->stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  
        }
        else {
            grimlock_control->stopL(1.0);
            std::cout << "Async path interrupted" << std::endl;
            return;
        }
    }
    std::cout << "end: " << grimlock_control->getAsyncOperationProgress() << std::endl;
}


void moveJ_path_async(RTDEControlInterface* grimlock_control, ControlState* control_state, std::vector<std::vector<double>>& path)
{

    std::cout << "before start: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    grimlock_control->moveJ(path, true);  


    // Wait for start of asynchronous operation
    while (grimlock_control->getAsyncOperationProgress() < 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Async movej started.. " << std::endl;

    while (grimlock_control->getAsyncOperationProgress() >= 0)
    {
        if (!control_state->stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  
        }
        else {
            grimlock_control->stopJ(1.0);
            std::cout << "Async path interrupted" << std::endl;
            return;
        }
    }
    std::cout << "end: " << grimlock_control->getAsyncOperationProgress() << std::endl;
}

void move_path_async(RTDEControlInterface* grimlock_control, ControlState* control_state, ur_rtde::Path& path)
{
    std::cout << "Move path asynchronously with progress feedback..." << std::endl;
    std::cout << "before start: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    grimlock_control->movePath(path, true);
    std::cout << "after start: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    // Wait for start of asynchronous operation
    while (grimlock_control->getAsyncOperationProgress() < 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Async path started.. " << std::endl;
    // Wait for end of asynchronous operation
    int waypoint = -1;
    std::cout << "at beginning: " << grimlock_control->getAsyncOperationProgress() << std::endl;
    while (grimlock_control->getAsyncOperationProgress() >= 0)
    {
        if (!control_state->stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            int new_waypoint = grimlock_control->getAsyncOperationProgress();
            if (new_waypoint != waypoint)
            {
                waypoint = new_waypoint;
                std::cout << "Moving to path waypoint " << waypoint << std::endl;
            }
        } else {
            grimlock_control->stopJ(1.0);
            std::cout << "Async path interrupted" << std::endl;
            return;
        }
    }
    std::cout << "Async path finished...\n\n" << std::endl;
}

void gripper_close_async(RobotiqGripper* gripper, ControlState* control_state)
{
    // gripper close
    gripper->close();
    while (gripper->objectDetectionStatus() == RobotiqGripper::MOVING)
    {
        if (!control_state->stop) {
            std::cout << "waiting..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            std::cout << "Gripper interrupted" << std::endl;
            return;
        }
    }
    printStatus(gripper->objectDetectionStatus());
}

void gripper_open_sync(RobotiqGripper* gripper, ControlState* control_state, int open_mm)
{
    gripper->move(open_mm);
    int status = gripper->waitForMotionComplete();
    printStatus(status);
}

void robot_process(const char* ip, ControlState* control_state, TSQueue<int>* queue, tuple_f* grimlock_xy, float* grimlock_rz) {
    
    RTDEControlInterface grimlock_control(ip);
    RTDEReceiveInterface grimlock_receive(ip);
    RobotiqGripper gripper(ip, 63352, true);
    gripper.connect();
    // Test emergency release functionality
    if (!gripper.isActive())
    {
    gripper.emergencyRelease(RobotiqGripper::OPEN);
    }
    std::cout << "Fault status: 0x" << std::hex << gripper.faultStatus() << std::dec << std::endl;
    std::cout << "activating gripper" << std::endl;
    gripper.activate();
    gripper.setUnit(RobotiqGripper::POSITION, RobotiqGripper::UNIT_MM);
    gripper.setUnit(RobotiqGripper::SPEED, RobotiqGripper::UNIT_NORMALIZED);
    gripper.setUnit(RobotiqGripper::FORCE, RobotiqGripper::UNIT_NORMALIZED);

    gripper.setPositionRange_mm(140);
    std::cout << "OpenPosition: " << gripper.getOpenPosition() << "  ClosedPosition: " << gripper.getClosedPosition()
            << std::endl;
    // We preset force and and speed so we don't need to pass it to the following move functions
    gripper.setForce(0.05);
    gripper.setSpeed(0.5);
    gripper.move(70);

    while (control_state->activate) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        while (!queue->empty()){
            int action = queue->pop();
            if (action == GoHome) {
                std::vector<double> home_j = {0.240322, -1.8224, 1.57128, -2.9112, -1.8073, -1.47553};
                grimlock_control.moveJ(home_j, 0.8, 0.6, true);  
            } else if (action == GoWorksurface) {
                std::vector<double> work_surface_j = {-1.0922616163836878, -1.812662740746969, 1.0695455710040491, -0.826763467197754, -1.5725553671466272, 0.48458147048950195};
                grimlock_control.moveJ(work_surface_j, 1.0, 1.5, true);
            } else if (action == OpenGripper) {
                gripper.move(70);
            } else if (action == CloseGripper) {
                gripper.close();
            } else if (action == InitiateJogging) {
                double speed_magnitude = 0.15;
                std::vector<double> speed_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                grimlock_control.jogStart(speed_vector, RTDEControlInterface::FEATURE_TOOL);
            } else if (action == StopJogging) {
                grimlock_control.jogStop();
            } else if (action == JogUp) {
                steady_clock::time_point t_start = grimlock_control.initPeriod();
                double speed_magnitude = 0.15;
                std::vector<double> speed_vector = {0.0, 0.0, -speed_magnitude, 0.0, 0.0, 0.0};
                grimlock_control.jogStart(speed_vector, RTDEControlInterface::FEATURE_TOOL);
                grimlock_control.waitPeriod(t_start);
            } else if (action == StopJ) {
                grimlock_control.stopJ(1.0);
            } else if (action == StopL) {
                grimlock_control.stopL(1.0);
            } else if (action == GetPose) {
                std::vector<double> actual_tcp_pose = grimlock_receive.getActualTCPPose();
                std::vector<double> actual_joint_q = grimlock_receive.getActualQ();
                std::cout << "actual_tcp_pose: " << std::endl;
                for(int i=0; i < actual_tcp_pose.size(); i++)
                    std::cout << actual_tcp_pose.at(i) << ' ';
                std::cout << std::endl;

                std::cout << "actual_joint_q: " << std::endl;
                for(int i=0; i < actual_joint_q.size(); i++)
                    std::cout << actual_joint_q.at(i) << ' ';
                std::cout << std::endl;
            } else if (action == PickBallArena) {
                std::vector<double> arena_tcp = {0.0, 0.0, 0.57148, -1.197, 2.905, 0.0};
                arena_tcp.at(0) = grimlock_xy->x / 1000.0f;
                arena_tcp.at(1) = grimlock_xy->y / 1000.0f;                


                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, arena_tcp, 0.5, 0.5);
                }

                arena_tcp.at(2) = 0.2913;
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, arena_tcp, 0.5, 0.5);
                }

                // grabbing 
                if (!control_state->stop) {
                    gripper_close_async(&gripper, control_state);
                }

                // move up 
                arena_tcp[2] = arena_tcp[2] + 0.1;
                
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, arena_tcp, 0.5, 0.5);
                }

            } else if (action == PickRampArena) {
                
                if (!control_state->stop) {
                    gripper_open_sync(&gripper, control_state, 140);
                }

                std::vector<double> prepare = {-0.57005, -0.45649, 0.57148, 0.0f, 3.1415, 0.0f};
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, prepare, 0.5, 0.5);
                }
                // go to above picking location 
                std::vector<double> arena_tcp = prepare;
                arena_tcp.at(0) = grimlock_xy->x / 1000.0f;
                arena_tcp.at(1) = grimlock_xy->y / 1000.0f;
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, arena_tcp, 0.5, 0.5);
                }
                // rotate gripper 
                std::vector<double> relative_p = {0.0, 0.0, 0.0, 0, 0, *grimlock_rz};
                std::vector<double> new_pose;
                new_pose = grimlock_control.poseTrans(arena_tcp, relative_p);
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, new_pose, 0.5, 0.5);
                }

                // move down 
                std::vector<double> actual_tcp_pose = grimlock_receive.getActualTCPPose();
                actual_tcp_pose[2] = 0.329047;
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, actual_tcp_pose, 0.5, 0.5);
                }

                // grabbing 
                if (!control_state->stop) {
                    gripper_close_async(&gripper, control_state);
                }

                // move up 
                actual_tcp_pose[2] = 0.372962;
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, actual_tcp_pose, 0.5, 0.5);
                }

            } else if (action == GrabBall) {
                std::vector<double> home_j = {0.240322, -1.8224, 1.57128, -2.9112, -1.8073, -1.47553};
                std::vector<double> clear_shelf = {-1.0802, -1.96873, 1.16649, -0.916983, -1.15136, -0.928407};
                std::vector<double> close_to_ball_dispensor = {-2.5463, -2.12161, 1.16651, -0.916947, -1.15138, -0.928419};
                std::vector<double> grap_tcp = {0.00209015, 0.298193, 1.0572, -2.62833, -0.0229904, -7.00752e-05};
                std::vector<double> close_to_ball_tcp = {0.00449163, 0.270184, 1.10409, -2.62838, -0.0229149, 4.93617e-05};
                
                // move to grab ball
                // TODO: check initial conditions before every move
                // if (!control_state->stop) {   
                //     moveJ_async(&grimlock_control, control_state, home_j, 2.0, 1.5);
                // }
                grimlock_control.moveJ(home_j, 2.0, 1.5, false);
                
                if (!control_state->stop) {
                    moveJ_async(&grimlock_control, control_state, clear_shelf, 2.0, 1.5);}
                if (!control_state->stop) {
                    moveJ_async(&grimlock_control, control_state, close_to_ball_dispensor, 1.5, 1.0);}
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, grap_tcp, 0.5, 0.5);}

                // grabbing 
                if (!control_state->stop) {
                    gripper_close_async(&gripper, control_state);
                }

                // move back home 
                if (!control_state->stop) {
                    moveL_async(&grimlock_control, control_state, close_to_ball_tcp, 0.5, 0.5);}
                if (!control_state->stop) {
                    moveJ_async(&grimlock_control, control_state, clear_shelf, 1.5, 1.0);}
                if (!control_state->stop) {
                    moveJ_async(&grimlock_control, control_state, home_j, 2.0, 1.5);}

                if (RobotiqGripper::STOPPED_INNER_OBJECT == gripper.objectDetectionStatus()) {
                    control_state->ball_idx = control_state->ball_idx + 1;
                }

            } else if (action == PlaceBallRamp) {
                std::vector<double> drop_ball_t; 
                // make the path, functionalize it 
                if (control_state->simple_task) {
                    drop_ball_t = {-0.5148749125401396, -0.5874186224841548, 0.3507017711584146, -1.201906495998176, 2.729668816002247, -0.4719344430225414};
                } else {
                    drop_ball_t = {-0.6039449461091941, -0.500736383332437, 0.3826673078527866, -1.2019741590191333, 2.729674424396362, -0.4719312874370168};
                }
                
                std::vector<double> above_ramp_t = drop_ball_t;
                above_ramp_t[2] = drop_ball_t[2] + 0.15;

                ur_rtde::Path path;
                double velocity = 1.3;
                double acceleration = 1.3;

                std::vector<double> drop_ball_t_path;
                std::vector<double> above_ramp_t_path;

                for (int i = 0; i < above_ramp_t.size(); i++)
                    above_ramp_t_path.push_back(above_ramp_t[i]);
                above_ramp_t_path.push_back(velocity);
                above_ramp_t_path.push_back(acceleration);
                above_ramp_t_path.push_back(0);

                for (int i = 0; i < drop_ball_t.size(); i++)
                    drop_ball_t_path.push_back(drop_ball_t[i]);
                drop_ball_t_path.push_back(0.5);
                drop_ball_t_path.push_back(0.5);
                drop_ball_t_path.push_back(0);
                
                path.addEntry({PathEntry::MoveJ, PathEntry::PositionTcpPose, above_ramp_t_path});
                path.addEntry({PathEntry::MoveL, PathEntry::PositionTcpPose, drop_ball_t_path});

                if (!control_state->stop) {
                    move_path_async(&grimlock_control, control_state, path);
                }
            }  else if (action == DropBall) {
                
                std::vector<double> above_ramp_t = grimlock_receive.getActualTCPPose();
                above_ramp_t[2] = above_ramp_t[2] + 0.15;

                ur_rtde::Path path;
                double velocity = 1.3;
                double acceleration = 1.3;

                std::vector<double> above_ramp_t_path;

                for (int i = 0; i < above_ramp_t.size(); i++)
                    above_ramp_t_path.push_back(above_ramp_t[i]);
                above_ramp_t_path.push_back(0.5);
                above_ramp_t_path.push_back(0.5);
                above_ramp_t_path.push_back(0);
                                
                path.addEntry({PathEntry::MoveL, PathEntry::PositionTcpPose, above_ramp_t_path});
                path.addEntry({PathEntry::MoveJ,
                               PathEntry::PositionJoints,
                               {0.24034643173217773, -1.8223873577513636, 1.5712783972369593, -2.9112163982787074, -1.8073275724994105, -1.475520435963766, velocity, acceleration, 0}});
                               
                if (!control_state->stop) {
                    gripper_open_sync(&gripper, control_state, 70);
                }

                if (!control_state->stop) {
                    move_path_async(&grimlock_control, control_state, path);
                }

            }
        }
    }
    
    grimlock_control.stopScript();
    printf("Robot thread exit\n");
}

#endif
