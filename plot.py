import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

parser=ArgumentParser()
parser.add_argument('-i','--in_csv', required=True)
parser.add_argument('-o','--out_png', required=True)

args=parser.parse_args()
in_csv=args.in_csv
out_png=args.out_png

df=pd.read_csv(in_csv)

joint_names = []
dofs_q = []
dofs_dq = []
dofs_ddq = []

joint_names=['floating_base_joint', 
             'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
             'left_knee_joint', 
             'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
             'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
             'right_knee_joint', 
             'right_ankle_pitch_joint', 'right_ankle_roll_joint'
             ]

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

dofs_q+=root_body_q
dofs_dq+=root_body_dq
dofs_ddq+=root_body_ddq

fallover=df['fallover']
df.drop(columns=['fallover'],inplace=True)
fall_start=fallover[fallover.diff()==1].index

df_norm=(df-df.mean())/df.std()

fig, axs = plt.subplots(2,2,sharex=True,constrained_layout=True)
axs=axs.ravel()

df_norm[dofs_q].plot(ax=axs[0],legend=False)
df_norm[dofs_dq].plot(ax=axs[1],legend=False)
df_norm[dofs_ddq].plot(ax=axs[2],legend=False)
df[root_body_q].plot(ax=axs[3],legend=False)

if len(fall_start)>0:
    axs[0].axvspan(fall_start[0],len(df), color='red', alpha=0.5)
    axs[1].axvspan(fall_start[0],len(df), color='red', alpha=0.5)
    axs[2].axvspan(fall_start[0],len(df), color='red', alpha=0.5)
    axs[3].axvspan(fall_start[0],len(df), color='red', alpha=0.5)

axs[0].set_xlabel('step')
axs[0].set_ylabel('q (norm.)')
axs[1].set_xlabel('step')
axs[1].set_ylabel('dq (norm.)')
axs[2].set_xlabel('step')
axs[2].set_ylabel('ddq (norm.)')
axs[3].set_xlabel('step')
axs[3].set_ylabel('root q')

plt.savefig(out_png,dpi=300)