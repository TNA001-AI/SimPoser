import os
from copy import copy
import mujoco
import logging
import time
from dataclasses import asdict
from pprint import pformat
from contextlib import nullcontext
import torch
import numpy as np
import mujoco.viewer
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    sanity_check_dataset_name,
)
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot.common.utils.utils import has_method, init_logging, log_say

RECORD = 10
RANDOM = True
policy_path="/home/tao/lerobot/outputs/train/act_so100_test_sim3/checkpoints/last/pretrained_model"


pre_cfg = PreTrainedConfig.from_pretrained(policy_path)
# print(pre_cfg)


cfg = ControlPipelineConfig(    
    robot=So100RobotConfig(),
    control=RecordControlConfig(
        fps=30,   
        single_task="Sim_Demo",   
        repo_id='Tna001/eval_so100_simulation3', 
        tags=["so100","simulation"],
        warmup_time_s=0, episode_time_s=90, reset_time_s=0,
        num_episodes=2, # Number of episodes
        push_to_hub=False,
        local_files_only=True,
        play_sounds=False, 
        device=torch.device("cuda"),
    )
)
cfg.control.policy = pre_cfg
cfg.control.policy.pretrained_path = policy_path

dataset0 = LeRobotDataset(
    repo_id='Tna001/so100_simulation2',
    root=cfg.control.root,
    local_files_only=cfg.control.local_files_only,
)

init_logging()
logging.info(pformat(asdict(cfg)))

robot = make_robot_from_config(cfg.robot)

ccfg = cfg.control # RecordControlConfig


dataset = LeRobotDataset.create(
    ccfg.repo_id,
    ccfg.fps,
    root=ccfg.root,
    robot=robot,
    use_videos=ccfg.video,
    image_writer_processes=ccfg.num_image_writer_processes,
    image_writer_threads=ccfg.num_image_writer_threads_per_camera * len(robot.cameras)
)



EPOCH = ccfg.num_episodes
_XML = "so_arm100/scene.xml"
max_vel = 4 * np.pi  
max_acc = 8.0
open_gripper = 0.3
close_gripper = -0.0
RENDER_WIDTH, RENDER_HEIGHT = 480, 640
CAMERA_NAMES = ["top_view", "front_view"]
FPS = ccfg.fps
rate = RateLimiter(frequency=FPS, warn=True)
dt = rate.period
VIDEO_DIR = "recordings"
os.makedirs(VIDEO_DIR, exist_ok=True)


model = mujoco.MjModel.from_xml_path(_XML)
renderer = mujoco.Renderer(model, RENDER_WIDTH, RENDER_HEIGHT)
cameras = {}
for name in CAMERA_NAMES:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    cameras[name] = cam


data = mujoco.MjData(model)

joint_names = [
    "Rotation", "Pitch", "Elbow",
    "Wrist_Pitch", "Wrist_Roll", "Jaw"
]
actuator_ids = np.array([model.actuator(name).id for name in joint_names])
# print("actuator_ids:", actuator_ids)


jaw_act_id = model.actuator("Jaw").id
jaw_joint_id = model.joint("Jaw").id
object_qpos_id = model.jnt_qposadr[model.joint("object_free").id]

place_site_id = model.site("place_site").id
object_geom_id = model.geom("object_geom").id
fixed_jaw_id = model.geom("fixed_jaw_pad_1").id
moving_jaw_id = model.geom("moving_jaw_pad_1").id


def main(dataset: LeRobotDataset = dataset,cfg: RecordControlConfig = ccfg):
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False
    ) as viewer:
        # print("ds_meta:", dataset.meta)
        # print("ds_meta0:", dataset0.meta)
        # print("policy:", cfg.policy)
        policy = make_policy(cfg.policy, cfg.device, ds_meta=dataset.meta)
        time.sleep(RECORD)
        for epoch in range(EPOCH):  
            log_say(f"\n🚀 Recording episode {dataset.num_episodes}", cfg.play_sounds)
            object_pos = reset_scene()

            run_epoch(viewer, dataset, cfg,time.time(), policy)

            dataset.save_episode(cfg.single_task)

    renderer.close()
    dataset.consolidate(cfg.run_compute_stats)
    log_say("Exiting", cfg.play_sounds)
    return dataset

def reset_scene():
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    # domain randomization
    if RANDOM:
        randomize_domain()
    # Set the initial position of the object
    # object_pos = np.random.uniform(low=[0.2, -0.1, 0.01], high=[0.3, 0.1, 0.01])
    object_pos = np.random.uniform(low=[0.18, -0.12, 0.01], high=[0.20, -0.10, 0.01])
    data.qpos[object_qpos_id: object_qpos_id + 7] = np.concatenate([object_pos, [1, 0, 0, 0]])
    mujoco.mj_forward(model, data)

    return object_pos




def run_epoch(viewer,dataset,cfg,start_time,policy):
    while viewer.is_running():
        mujoco.mj_forward(model, data)

        # TODO
        state_ = torch.from_numpy(data.qpos[0:6].astype(np.float32))
        state   = torch.rad2deg(state_)
        obs_dict = {}
        obs_dict["observation.state"] = state

        for name, cam in cameras.items():
            renderer.update_scene(data, cam)
            rendered_img = renderer.render()
            if rendered_img.dtype != np.uint8:
                rendered_img = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
            rendered_tensor = torch.from_numpy(rendered_img.copy()) 
            # print(f"rendered_tensor shape: {rendered_tensor.shape}")
            obs_dict[f"observation.images.{name}"] = rendered_tensor    
        observation = obs_dict
        
        pred_action = predict_action(observation, policy, cfg.device, use_amp=False)
        # TODO: add some clipping
        # action = np.clip(pred_action, ..., ...)

        action = {"action": pred_action}

        data.ctrl[actuator_ids] =  torch.deg2rad(pred_action).numpy().astype(np.float32)
        mujoco.mj_step(model, data)

        viewer.sync()
        rate.sleep()


        frame = {**obs_dict, **action}
        dataset.add_frame(frame)
        if time.time() - start_time >= cfg.episode_time_s:
            log_say("⏱️ Time reached. Finishing epoch.", cfg.play_sounds)
            return

        
def predict_action(observation, policy, device, use_amp=True):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action
def randomize_domain():
    """Domain randomization: randomize robot, object, table, sky color, and light properties."""

    # Randomize robot visual parts (group=2 geoms only)
    for i in range(model.ngeom):
        geom = model.geom(i)
        if geom.group == 2:
            rgba = np.random.uniform(0.2, 1.0, size=3).tolist() + [1.0]
            model.geom_rgba[i] = rgba

    # Randomize the color of the target object
    try:
        object_geom_id = model.geom("object_geom").id
        object_color = np.random.uniform(0.2, 1.0, size=3).tolist() + [1.0]
        model.geom_rgba[object_geom_id] = object_color
    except Exception as e:
        print(f"Failed to randomize object color: {e}")

    # Randomize the color of the table
    try:
        table_mat_id = model.mat("table").id
        random_color = np.random.uniform(0.4, 1.0, size=3)
        model.mat_rgba[table_mat_id, :3] = random_color
        model.mat_rgba[table_mat_id, 3] = 1.0
    except Exception as e:
        print(f"Failed to randomize table color: {e}")

    # Randomize light properties
    try:
        for i in range(model.nlight):
            # Randomize diffuse color
            model.light_diffuse[i] = np.random.uniform(0.5, 1.0, size=3)
            # Randomize ambient color
            model.light_ambient[i] = np.random.uniform(0.2, 0.6, size=3)
            # Randomize specular color (optional, usually low)
            model.light_specular[i] = np.random.uniform(0.0, 0.2, size=3)
            # Randomize light position slightly
            model.light_pos[i][:3] += np.random.uniform(-1.2, 1.2, size=3)
    except Exception as e:
        print(f"Failed to randomize light properties: {e}")

    # Randomize background board material
    try:
        background_geom_id = model.geom("background_board").id

        # Material names corresponding to the 60 backgrounds
        background_materials = ["bg_mat1", "bg_mat2", "bg_mat3", "bg_mat4", "bg_mat5"]

        # Randomly pick one
        chosen_material = np.random.choice(background_materials)

        # Get material id
        material_id = model.mat(chosen_material).id

        # Apply material to the background board
        model.geom_matid[background_geom_id] = material_id

    except Exception as e:
        print(f"Failed to randomize background material: {e}")

    # Randomize camera positions (slight perturbations)
    try:
        for name in CAMERA_NAMES:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            cam_pos = model.cam_pos[cam_id]
            perturb = np.random.uniform(-0.02, 0.02, size=3)  # slight +-2cm perturb
            model.cam_pos[cam_id] = cam_pos + perturb
    except Exception as e:
        print(f"Failed to randomize camera position: {e}")
        
    mujoco.mj_forward(model, data)

if __name__ == "__main__":
    main()
