import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import create_dataset

from torchviz import make_dot

from nn import LSTM

parser=ArgumentParser()
parser.add_argument('-i','--in_csv', required=True)
parser.add_argument('-o','--out_png', required=True)
parser.add_argument('-m','--model_path', required=False, default=None)
parser.add_argument('--draw_graph', action='store_true', default=False)

args=parser.parse_args()
in_csv=args.in_csv
out_png=args.out_png
model_path=args.model_path
draw_graph=args.draw_graph

df=pd.read_csv(in_csv)

# copied from preprocessing.py
window_size=10
time_into_future=0.5
log_dt=0.02

times=np.arange(0, len(df)*log_dt, log_dt)

if args.model_path:
    # Define model
    model = LSTM(input_size=len(df.columns)-1)
    # Load the model
    model.load_state_dict(torch.load(model_path,weights_only=True))
    # Set the model to evaluation mode
    model.eval()

    traj = df.to_numpy()
    features = traj[:, :-1]
    targets = traj[:, -1]
    X, y = create_dataset(features=features, targets=targets)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model_out=[]
    for inputs, labels in loader:
        outputs = model(inputs)
        model_out=np.concatenate([model_out,outputs.detach().numpy()],axis=0)

    if draw_graph:
        make_dot(outputs.mean(), params=dict(model.named_parameters())).render()

    num_timesteps_into_future = int(time_into_future / log_dt)
    model_out=np.concatenate([np.zeros((window_size-1)),model_out],axis=0)

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
if args.model_path:
    for ax in axs:
        ax_twin = ax.twinx()
        # ax_twin.set_ylabel('model output')
        ax_twin.set_yticks([])
        ax_twin.set_ylim(0,1)
        ax_twin.plot(times[:-num_timesteps_into_future],model_out, ls='--', color='blue', alpha=0.5)

for col in dofs_q:
    axs[0].plot(times, df_norm[col])
for col in dofs_dq:
    axs[1].plot(times, df_norm[col])
for col in dofs_ddq:
    axs[2].plot(times, df_norm[col])
for col in root_body_q:
    axs[3].plot(times, df[col])

if len(fall_start)>0:
    axs[0].axvspan(fall_start[0]*log_dt,len(df)*log_dt, color='red', alpha=0.5)
    axs[1].axvspan(fall_start[0]*log_dt,len(df)*log_dt, color='red', alpha=0.5)
    axs[2].axvspan(fall_start[0]*log_dt,len(df)*log_dt, color='red', alpha=0.5)
    axs[3].axvspan(fall_start[0]*log_dt,len(df)*log_dt, color='red', alpha=0.5)

# axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('q (norm.)')
# axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('dq (norm.)')
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('ddq (norm.)')
axs[3].set_xlabel('time (s)')
axs[3].set_ylabel('root q')

plt.savefig(out_png,dpi=300)