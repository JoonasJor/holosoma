# Holosoma Locomotion Quick Start

## Installing
```bash
git clone https://github.com/JoonasJor/holosoma
cd holosoma/
bash scripts/setup_inference.sh
bash scripts/setup_mujoco.sh
```

## Running
### Physical Deployment
1. Connect ethernet cable to Arska. Set ip address to 192.168.123.222 and netmask to 255.255.255.0
2. Set Arska to debug mode (hold L2 + R2)
2. Get network interface name (```ip addr``` / ```ifconfig```)
3. Run:
```bash
source scripts/source_inference_setup.sh
```
```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface <NETWORK INTERFACE NAME>
```
4. On Unitree controller, press "A" to activate the policy. Then press "Start" to enter walking mode
5. On Unitree controller, control Arska with left and right joysticks

### MuJoCo Simulation
1. In terminal 1, run:
```bash
source scripts/source_mujoco_setup.sh
```
```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```
2. In terminal 2, run:
```bash
source scripts/source_inference_setup.sh
```
```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```
3. In MuJoCo window, press 8 to lower the gantry until robot touches ground
4. In terminal 2, press "]" to activate the policy. Then press "=" to enter walking mode
5. In MuJoCo window, press 9 to disable elastic band
6. In terminal 2, control robot with w/a/s/d and q/e

## Training Locomotion
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac-low-speed \
    simulator:mjwarp \
    logger:wandb \
    --logger.video.enabled False
```
