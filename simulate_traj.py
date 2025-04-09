import mujoco
import mujoco.viewer
import pandas as pd
import numpy as np
import time
import argparse

def simulate(xml_path, traj_file, dt=0.002):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = dt
    df = pd.read_csv(traj_file)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for id, row in df.iterrows():
            qpos = row[:19].to_numpy()
            qvel = row[19:37].to_numpy()
            qacc = row[37:55].to_numpy()
            fallover = int(row[55]) 
            d.qpos = qpos
            d.qvel = qvel
            d.qacc = qacc
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(0.01)
       
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', type=str, default="model/scene.xml")
    parser.add_argument('--traj_file', type=str, default="data/g1_traj/1449.csv")

    args=parser.parse_args()
    xml_path = args.xml_path
    traj_file = args.traj_file

    simulate(xml_path, traj_file)