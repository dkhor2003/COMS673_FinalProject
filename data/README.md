# G1 Robot State Trajectory CSV Configuration For Fallover Prediction

---
## Summary
- Columns 1 - 19 : qpos 
- Columns 20 - 37: qvel 
- Columns 38 - 55: qacc
- Column  56     : fallover indicator (0 for non-fall, 1 for fall)
---

## Column Details
**Column 1**: 
- Column Title: floating_base_joint_x
- Joint Name: floating_base_joint (root)
- About: CoM x-position wrt to starting point

**Column 2**: 
- Column Title: floating_base_joint_y
- Joint Name: floating_base_joint (root)
- About: CoM y-position wrt to starting point

**Column 3**: 
- Column Title: floating_base_joint_z
- Joint Name: floating_base_joint (root)
- About: CoM z-position wrt to starting point

**Column 4**: 
- Column Title: floating_base_joint_qw
- Joint Name: floating_base_joint (root)
- About: cos(angle/2) in quaternion

**Column 5**: 
- Column Title: floating_base_joint_qx
- Joint Name: floating_base_joint (root)
- About: rotation axis unit vector x-component * sin(angle/2)

**Column 6**: 
- Column Title: floating_base_joint_qy
- Joint Name: floating_base_joint (root)
- About: rotation axis unit vector y-component * sin(angle/2)

**Column 7**: 
- Column Title: floating_base_joint_qz
- Joint Name: floating_base_joint (root)
- About: rotation axis unit vector z-component * sin(angle/2)

**Column 8**: 
- Column Title: left_hip_pitch_joint_q
- Joint Name: left_hip_pitch_joint
- About: joint angle
- Upper Limit: 2.6093 rad
- Lower Limit: -2.2602 rad

**Column 9**: 
- Column Title: left_hip_roll_joint_q
- Joint Name: left_hip_roll_joint
- About: joint angle
- Upper Limit: 2.7926 rad
- Lower Limit: -0.3491 rad

**Column 10**: 
- Column Title: left_hip_yaw_joint_q
- Joint Name: left_hip_yaw_joint
- About: joint angle
- Upper Limit: 2.4818 rad
- Lower Limit: -2.4818 rad

**Column 11**: 
- Column Title: left_knee_joint_q
- Joint Name: left_knee_joint
- About: joint angle
- Upper Limit: 2.7314 rad
- Lower Limit: 0.0611 rad

**Column 12**: 
- Column Title: left_ankle_pitch_joint_q
- Joint Name: left_ankle_pitch_joint
- About: joint angle
- Upper Limit: 0.4538 rad
- Lower Limit: -0.8029 rad

**Column 13**: 
- Column Title: left_ankle_roll_joint_q
- Joint Name: left_ankle_roll_joint
- About: joint angle
- Upper Limit: 0.2356 rad
- Lower Limit: -0.2356 rad

**Column 14**: 
- Column Title: right_hip_pitch_joint_q
- Joint Name: right_hip_pitch_joint
- About: joint angle
- Upper Limit: 2.6093 rad
- Lower Limit: -2.2602 rad

**Column 15**: 
- Column Title: right_hip_roll_joint_q
- Joint Name: right_hip_roll_joint
- About: joint angle
- Upper Limit: 0.3491 rad
- Lower Limit: -2.7926 rad

**Column 16**: 
- Column Title: right_hip_yaw_joint_q
- Joint Name: right_hip_yaw_joint
- About: joint angle
- Upper Limit: 2.4818 rad
- Lower Limit: -2.4818 rad

**Column 17**: 
- Column Title: right_knee_joint_q
- Joint Name: right_knee_joint
- About: joint angle
- Upper Limit: 2.7314 rad
- Lower Limit: 0.0611 rad

**Column 18**: 
- Column Title: right_ankle_pitch_joint_q
- Joint Name: right_ankle_pitch_joint
- About: joint angle
- Upper Limit: 0.4538 rad
- Lower Limit: -0.8029 rad

**Column 19**: 
- Column Title: right_ankle_roll_joint_q
- Joint Name: right_ankle_roll_joint
- About: joint angle
- Upper Limit: 0.2356 rad
- Lower Limit: -0.2356 rad

**Column 20**: 
- Column Title: floating_base_joint_vx
- Joint Name: floating_base_joint (root)
- About: CoM linear x-velocity

**Column 21**: 
- Column Title: floating_base_joint_vy
- Joint Name: floating_base_joint (root)
- About: CoM linear y-velocity

**Column 22**: 
- Column Title: floating_base_joint_vz
- Joint Name: floating_base_joint (root)
- About: CoM linear z-velocity

**Column 23**: 
- Column Title: floating_base_joint_wx
- Joint Name: floating_base_joint (root)
- About: CoM angular x-velocity

**Column 24**: 
- Column Title: floating_base_joint_wy
- Joint Name: floating_base_joint (root)
- About: CoM angular y-velocity

**Column 25**: 
- Column Title: floating_base_joint_wz
- Joint Name: floating_base_joint (root)
- About: CoM angular z-velocity

**Column 26**: 
- Column Title: left_hip_pitch_joint_dq
- Joint Name: left_hip_pitch_joint
- About: joint angular velocity
- Upper Limit: 32 rad/s
- Lower Limit: -32 rad/s

**Column 27**: 
- Column Title: left_hip_roll_joint_dq
- Joint Name: left_hip_roll_joint
- About: joint angular velocity
- Upper Limit: 20 rad/s
- Lower Limit: -20 rad/s

**Column 28**: 
- Column Title: left_hip_yaw_joint_dq
- Joint Name: left_hip_yaw_joint
- About: joint angular velocity
- Upper Limit: 32 rad/s
- Lower Limit: -32 rad/s

**Column 29**: 
- Column Title: left_knee_joint_dq
- Joint Name: left_knee_joint
- About: joint angular velocity
- Upper Limit: 20 rad/s
- Lower Limit: -20 rad/s

**Column 30**: 
- Column Title: left_ankle_pitch_joint_dq
- Joint Name: left_ankle_pitch_joint
- About: joint angular velocity
- Upper Limit: 37 rad/s
- Lower Limit: -37 rad/s

**Column 31**: 
- Column Title: left_ankle_roll_joint_dq
- Joint Name: left_ankle_roll_joint
- About: joint angular velocity
- Upper Limit: 37 rad/s
- Lower Limit: -37 rad/s

**Column 32**: 
- Column Title: right_hip_pitch_joint_dq
- Joint Name: right_hip_pitch_joint
- About: joint angular velocity
- Upper Limit: 32 rad/s
- Lower Limit: -32 rad/s

**Column 33**: 
- Column Title: right_hip_roll_joint_dq
- Joint Name: right_hip_roll_joint
- About: joint angular velocity
- Upper Limit: 20 rad/s
- Lower Limit: -20 rad/s

**Column 34**: 
- Column Title: right_hip_yaw_joint_dq
- Joint Name: right_hip_yaw_joint
- About: joint angular velocity
- Upper Limit: 32 rad/s
- Lower Limit: -32 rad/s

**Column 35**: 
- Column Title: right_knee_joint_dq
- Joint Name: right_knee_joint
- About: joint angular velocity
- Upper Limit: 20 rad/s
- Lower Limit: -20 rad/s

**Column 36**: 
- Column Title: right_ankle_pitch_joint_dq
- Joint Name: right_ankle_pitch_joint
- About: joint angular velocity
- Upper Limit: 37 rad/s
- Lower Limit: -37 rad/s

**Column 37**: 
- Column Title: right_ankle_roll_joint_dq
- Joint Name: right_ankle_roll_joint
- About: joint angular velocity
- Upper Limit: 37 rad/s
- Lower Limit: -37 rad/s

**Column 38**: 
- Column Title: floating_base_joint_acc_x
- Joint Name: floating_base_joint (root)
- About: CoM linear x-acceleration

**Column 39**: 
- Column Title: floating_base_joint_acc_y
- Joint Name: floating_base_joint (root)
- About: CoM linear y-acceleration

**Column 40**: 
- Column Title: floating_base_joint_acc_z
- Joint Name: floating_base_joint (root)
- About: CoM linear z-acceleration

**Column 41**: 
- Column Title: floating_base_joint_alpha_x
- Joint Name: floating_base_joint (root)
- About: CoM angular x-acceleration

**Column 42**: 
- Column Title: floating_base_joint_alpha_y
- Joint Name: floating_base_joint (root)
- About: CoM angular y-acceleration

**Column 43**: 
- Column Title: floating_base_joint_alpha_z
- Joint Name: floating_base_joint (root)
- About: CoM angular z-acceleration

**Column 44**: 
- Column Title: left_hip_pitch_joint_ddq
- Joint Name: left_hip_pitch_joint
- About: joint angular acceleration

**Column 45**: 
- Column Title: left_hip_roll_joint_ddq
- Joint Name: left_hip_roll_joint
- About: joint angular acceleration

**Column 46**: 
- Column Title: left_hip_yaw_joint_ddq
- Joint Name: left_hip_yaw_joint
- About: joint angular acceleration

**Column 47**: 
- Column Title: left_knee_joint_ddq
- Joint Name: left_knee_joint
- About: joint angular acceleration

**Column 48**: 
- Column Title: left_ankle_pitch_joint_ddq
- Joint Name: left_ankle_pitch_joint
- About: joint angular acceleration

**Column 49**: 
- Column Title: left_ankle_roll_joint_ddq
- Joint Name: left_ankle_roll_joint
- About: joint angular acceleration

**Column 50**: 
- Column Title: right_hip_pitch_joint_ddq
- Joint Name: right_hip_pitch_joint
- About: joint angular acceleration

**Column 51**: 
- Column Title: right_hip_roll_joint_ddq
- Joint Name: right_hip_roll_joint
- About: joint angular acceleration

**Column 52**: 
- Column Title: right_hip_yaw_joint_ddq
- Joint Name: right_hip_yaw_joint
- About: joint angular acceleration

**Column 53**: 
- Column Title: right_knee_joint_ddq
- Joint Name: right_knee_joint
- About: joint angular acceleration

**Column 54**: 
- Column Title: right_ankle_pitch_joint_ddq
- Joint Name: right_ankle_pitch_joint
- About: joint angular acceleration

**Column 55**: 
- Column Title: right_ankle_roll_joint_ddq
- Joint Name: right_ankle_roll_joint
- About: joint angular acceleration

**Column 56**: 
- Column Title: fallover
- About: fallover indicator (0 for non-fall, 1 for fall)