import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import random
import yaml
import os
import csv

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[qw] * q[qx] + q[qy] * q[qz])
    cosr_cosp = q[qw] * q[qw] - q[qx] * q[qx] - q[qy] * q[qy] + q[qz] * q[qz]
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[qw] * q[qy] - q[qz] * q[qx])
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2.0, sinp), np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[qw] * q[qz] + q[qx] * q[qy])
    cosy_cosp = q[qw] * q[qw] + q[qx] * q[qx] - q[qy] * q[qy] - q[qz] * q[qz]
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw))

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """ Calculates torques from position commands """
    return (target_q - q) * kp + (target_dq - dq) * kd
    
if __name__ == "__main__":
    # get config file name from command line
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    working_dir = os.path.dirname(os.path.abspath(__file__))
    data_root_dir = "data/g1_traj"
    
    vx_range = [-1.2, 1.2]
    vy_range = [-1.2, 1.2]
    yaw_range = [-1., 1.]
    force_ranges = [-50, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 50]
    
    x_force = random.choice(force_ranges)
    y_force = random.choice(force_ranges)
    vx = random.uniform(vx_range[0], vx_range[1])
    vy = random.uniform(vy_range[0], vy_range[1])
    yaw = random.uniform(yaw_range[0], yaw_range[1])
    
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{WORKING_DIR}", working_dir)
        policy_path = policy_path.replace("{POLICY}", config["policy"])
        xml_path = config["xml_path"].replace("{WORKING_DIR}", working_dir)
        
        log_on = config["log_on"]
        render = config["render"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        cmd = np.array([vx, vy, yaw])

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    
    counter = 0

    # One control one actual
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    if log_on:
        
        if not os.path.exists(data_root_dir):
            os.mkdir(data_root_dir)
            traj_num = 1
        else:
            trajs = os.listdir(data_root_dir)
            traj_num = len(list(filter(lambda file: file.endswith(".csv"), trajs))) + 1
            
        csvfile = open(os.path.join(data_root_dir, f"{traj_num}.csv"), "w", newline="")
        traj_logger = csv.writer(csvfile)
        
        joint_names = []
        dofs_q = []
        dofs_dq = []
        dofs_ddq = []
        
        for i in range(m.njnt):
            joint_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

        root_body = joint_names[0]
        dofs = joint_names[1:]
        
        root_body_q = [f"{root_body}_x",
                       f"{root_body}_y",
                       f"{root_body}_z",
                       f"{root_body}_qw",
                       f"{root_body}_qx",
                       f"{root_body}_qy",
                       f"{root_body}_qz"]

        root_body_dq = [f"{root_body}_vx",
                        f"{root_body}_vy",
                        f"{root_body}_vz",
                        f"{root_body}_wx",
                        f"{root_body}_wy",
                        f"{root_body}_wz"]
        
        root_body_ddq = [f"{root_body}_acc_x",
                         f"{root_body}_acc_y",
                         f"{root_body}_acc_z",
                         f"{root_body}_alpha_x",
                         f"{root_body}_alpha_y",
                         f"{root_body}_alpha_z"]
        
        for dof in dofs:
            dofs_q.append(f"{dof}_q")
            dofs_dq.append(f"{dof}_dq")
            dofs_ddq.append(f"{dof}_ddq")
        
        header = root_body_q + dofs_q + root_body_dq + dofs_dq + root_body_ddq + dofs_ddq + ["fallover"]
        traj_logger.writerow(header)  

    fallover_steps = 0

    # load policy
    policy = torch.jit.load(policy_path)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            d.qfrc_applied[0] = x_force
            d.qfrc_applied[1] = y_force
            mujoco.mj_step(m, d)
        
            counter += 1
            if counter % control_decimation == 0:
                
                # Log joint states as well as fallover indicator
                if log_on:
                    rpy = get_euler_xyz(d.qpos[3:7])
                    if d.qpos[2] < 0.58: # abs(rpy[1]) > 1.0 or abs(abs(rpy[0])-np.pi) > 0.8
                        fallover = 1
                        fallover_steps += 1
                    else:
                        fallover = 0
                    traj_logger.writerow(np.concatenate((d.qpos, d.qvel, d.qacc, [fallover])))
                    if fallover_steps == 25:
                        break
                
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    if log_on:
        csvfile.close()