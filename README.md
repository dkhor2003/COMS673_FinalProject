<div align="center">
  <h1 align="center">G1 Real-Time Fallover Prediction</h1>
</div>

<p align="center">
  <strong>This is a repository for our COMS 673 Final Project. Our objective here is to ...</strong> 
</p>

---

## 📦 Installation and Configuration

Clone the repository:
```bash
git clone https://github.com/dkhor2003/COMS673_FinalProject.git
```

Create a virtual environment:
```bash
conda create -n COMS673 python=3.8
```

Then activate it:
```bash
conda activate COMS673
```

Install necessary dependencies:
```bash
conda install conda-forge::mujoco-python
conda install pytorch=2.3.1
conda install pyyaml
```

## Simulation in MuJoCo

Simulate G1 in the Mujoco simulator:

```bash
python deploy_mujoco.py g1.yaml
```

#### ➡️ Important YAML Configuration Variables

The available policies are located under the `policy` directory. 
- Update the `policy_path` in the YAML configuration file to switch to a different locomotion policy available under the `policy` directory.
- Update the `cmd_init` [vx, vy, yaw] in the YAML configuration file to change the desired starting motion. 

If `log_on` is `true`, running the above line will generate joint state trajectories as a csv file under the `data` folder.

