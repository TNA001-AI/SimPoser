import os
import mujoco
import logging
import time
import inspect
import re
import json
import openai
from openai import OpenAI
from dataclasses import asdict
from pprint import pformat

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

from lerobot.common.utils.utils import has_method, init_logging, log_say


RANDOM = False

cfg = ControlPipelineConfig(    
    robot=So100RobotConfig(),
    control=RecordControlConfig(
        fps=30,   
        single_task="Sim_Demo",   
        repo_id='Tna001/so100_simulation3', 
        tags=["so100","simulation"],
        warmup_time_s=0, episode_time_s=55, reset_time_s=0,
        num_episodes=80, # Number of episodes to record
        push_to_hub=False,
        local_files_only=True,
        policy=None,
        play_sounds=False,  
    )
)

init_logging()
logging.info(pformat(asdict(cfg)))

robot = make_robot_from_config(cfg.robot)

ccfg = cfg.control # RecordControlConfig
sanity_check_dataset_name(ccfg.repo_id, ccfg.policy)
dataset = LeRobotDataset.create(
    ccfg.repo_id,
    ccfg.fps,
    root=ccfg.root,
    robot=robot,
    use_videos=ccfg.video,
    image_writer_processes=ccfg.num_image_writer_processes,
    image_writer_threads=ccfg.num_image_writer_threads_per_camera * len(robot.cameras),
)


import torch
import numpy as np
import mujoco.viewer
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter

import mink
import warnings
warnings.filterwarnings("ignore", message="Converted P to scipy.sparse.csc.csc_matrix")
warnings.filterwarnings("ignore", message="Converted G to scipy.sparse.csc.csc_matrix")
from dm_control.viewer import user_input

EPOCH = ccfg.num_episodes
_XML = "so_arm100/scene.xml"
max_vel = 3 * np.pi  
max_acc = 1*np.pi
open_gripper = 0.3
close_gripper = -0.0
RENDER_WIDTH, RENDER_HEIGHT = 480, 640
CAMERA_NAMES = ["top_view", "front_view"]
FPS = ccfg.fps
rate = RateLimiter(frequency=FPS, warn=True)
dt = rate.period
VIDEO_DIR = "recordings"
os.makedirs(VIDEO_DIR, exist_ok=True)

@dataclass
class KeyCallback:
    pause: bool = False
    exit: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause
        elif key == user_input.KEY_Q:
            self.exit = True




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
    "Wrist_Pitch", "Wrist_Roll"
]
dof_ids = np.array([model.joint(name).id for name in joint_names])
actuator_ids = np.array([model.actuator(name).id for name in joint_names])

configuration = mink.Configuration(model)

end_effector_task = mink.FrameTask(
    frame_name="gripper_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,
    lm_damping=1e-2,
)

posture_cost = np.zeros((model.nv,))
posture_task = mink.PostureTask(model, posture_cost)

damping_cost = np.zeros((model.nv,))
damping_cost[:] = 1e-2
damping_task = mink.DampingTask(model, damping_cost)

tasks = [end_effector_task, posture_task]
limits = [mink.ConfigurationLimit(model)]
key_callback = KeyCallback()

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
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:

        for epoch in range(EPOCH):  
            log_say(f"\n🚀 Recording episode {dataset.num_episodes}", cfg.play_sounds)
            object_pos = reset_scene()

            run_epoch(object_pos, viewer, dataset, cfg, 
                      vel_prev = np.zeros(model.nv, dtype=np.float32),start_time=time.time())

            dataset.save_episode(cfg.single_task)
            if key_callback.exit:
                break

    renderer.close()
    dataset.consolidate(cfg.run_compute_stats)
    log_say("Exiting", cfg.play_sounds)
    return dataset

def reset_scene():
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)

    end_effector_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)

        # domain randomization
    if RANDOM:
        randomize_domain()


    object_pos = np.random.uniform(low=[0.18, -0.12, 0.01], high=[0.20, -0.10, 0.01])
    data.qpos[object_qpos_id: object_qpos_id + 7] = np.concatenate([object_pos, [1, 0, 0, 0]])
    mujoco.mj_forward(model, data)

    return object_pos

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


# --------------------------  Stage functions  -------------------------------
def approach(context, params):
    """Approach the object before grasping.
    Params may be empty;
    """
    target = context["object_pos"] + np.array([0.0, 0.0, 0.10])
    if np.linalg.norm(context["ee_pos"] - target) < 0.01:
        print("Approaching the object...")
        data.ctrl[jaw_act_id] = open_gripper
        return True, target
    return False, target

def wait_open(context, params):
    """Wait for the gripper to open before grasping the object.
    Params may be empty;
    """
    target = None
    if context["jaw_angle"] >= open_gripper - 0.01:
        return True, target
    return False, target

def grasp(context, params):
    """Grasp the object.
    Required params: grasp_height (float)
    """
    # grasp_h = params.get("grasp_height", 0.018)
    grasp_h = params.get("grasp_height")
    target = context["object_pos"] + np.array([0.0, 0.0, grasp_h])
    if np.linalg.norm(context["ee_pos"] - target) < 0.01:
        data.ctrl[jaw_act_id] = close_gripper
        return True, target
    return False, target

def wait_close(context, params):
    """Wait for the gripper to close on the object.
    Params may be empty;
    """
    target = None
    contact_fixed = contact_moving = False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if object_geom_id in (g1, g2):
            other = g2 if g1 == object_geom_id else g1
            if other == fixed_jaw_id:
                contact_fixed = True
            elif other == moving_jaw_id:
                contact_moving = True
        if contact_fixed and contact_moving:
            break
    if contact_fixed and contact_moving:
        return True, target
    return False, target

def lift(context, params):
    """Lift the object to a specified height.
    Required params: lift_height (float)
    """
    # lift_h = params.get("lift_height", 0.15)
    lift_h = params.get("lift_height")
    target = context["object_pos"] + np.array([0.0, 0.0, lift_h])
    if np.linalg.norm(context["ee_pos"] - target) < 0.01:
        return True, target
    return False, target

def move(context, params):
    """Move the object above the place position.
    Required params: lift_height (float)
    """
    # lift_h = params.get("lift_height", 0.15)
    lift_h = params.get("lift_height")
    target = context["place_pos"] + np.array([0.0, 0.0, lift_h])
    if np.linalg.norm(context["ee_pos"] - target) < 0.02:
        return True, target
    return False, target

def place(context, params):
    """Place the object at the target position.
    Required params: grasp_height (float)
    """
    end_effector_task.orientation_cost = 0.1
    # grasp_h = params.get("grasp_height", 0.018)
    grasp_h = params.get("grasp_height")
    target = context["place_pos"] + np.array([0.0, 0.0, grasp_h])
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    if np.linalg.norm(context["ee_pos"] - target) < 0.01:
        end_effector_task.orientation_cost = 0.0
        data.ctrl[jaw_act_id] = open_gripper
        return True, target
    return False, target

def wait_open_place(context, params):
    """Wait for the gripper to open after placing.
    Params may be empty;
    """
    target = None
    if context["jaw_angle"] >= open_gripper - 0.01:
        return True, target
    return False, target

def retreat(context, params):
    """Retreat to the home position and finish the epoch.
    Params may be empty;
    """
    target = np.array([0.2, 0.0, 0.15])
    data.ctrl[jaw_act_id] = close_gripper
    if np.linalg.norm(context["ee_pos"] - target) < 0.01:
        print("✅ One epoch completed.")
        return True, target  # end of episode
    return False, target
# ---------------------  Stage registry and sequence  ------------------------

stage_functions = {
    "approach": approach,
    "wait_open": wait_open,
    "grasp": grasp,
    "wait_close": wait_close,
    "lift": lift,
    "move": move,
    "place": place,
    "wait_open_place": wait_open_place,
    "retreat": retreat,
}

stages_sequence = list(stage_functions)  # use each once

# --------------------------  Helper utilities  ------------------------------

def collect_context(object_pos, place_pos):
    """Return state dict consumed by stages & LLM."""
    return {
        "object_pos": object_pos,
        "place_pos":  place_pos,
        "ee_pos":     configuration.get_transform_frame_to_world("gripper_site", "site").translation().copy(),
        "jaw_angle":  float(data.qpos[jaw_joint_id]),
    }

def pose_controller(stage, context):
    """Query ChatGPT for next_stage and params (returns dict)."""
    stage_code = inspect.getsource(stage_functions[stage])
    sys_prompt = (
        "You are a helpful assistant for robotic control. Given current stage function and environment snapshot, your task is to determine the parameters needed to complete the task. Here is the stage function:\n"
        f"{stage_code}\n"
        "For each stage, the parameters you need to predict are listed in the function's docstring under 'Required params'. "
        "For each call, respond with JSON: {'params': dict} only."
    )
    user_prompt = f"Current stage: {stage}\nCurrent Environment snapshot:\n{context}"
    rsp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    response = rsp.choices[0].message.content
    print("\nresponse:", response)
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
    print("\ncleaned:", cleaned)
    payload = json.loads(cleaned)

    print(f"LLM prompt: system: {sys_prompt} \n user: {user_prompt}]")
    print(f"LLM response: {response}")
    print(f"LLM response: {cleaned}")

    return payload.get("params", {})

def stage_controller(stage, context, remaining):
    """Query ChatGPT for next_stage (returns str)."""
    stage_code = "\n\n".join(inspect.getsource(fn) for fn in stage_functions.values())
    sys_prompt = (
        f"You are a helpful assistant for robotic control. Give current stage, your task is to determine the next stage from {remaining}. Here are the available stage functions:\n"
        f"{stage_code}\n"
        "For each call, respond with JSON: {'next_stage': str} only."
    )
    user_prompt = f"Current stage: {stage}\nCurrent Environment snapshot:\n{context}"
    rsp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
    )
    print(f"LLM prompt: system: {sys_prompt} \n user: {user_prompt}]")
    response = rsp.choices[0].message.content
    cleaned = re.sub(r"^```json\s*|\s*```$", "", response.strip())
    print(f"LLM response: {response}")
    print(f"LLM response: {cleaned}")
    payload = json.loads(cleaned)

    return payload["next_stage"]

# --------------------------  Main control loop  -----------------------------
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_epoch(object_pos, writers,viewer):
    place_pos = np.array([0.3, 0.0, 0.03])
    data.site_xpos[place_site_id] = place_pos
    stage = "approach"
    remaining = stages_sequence.copy()
    params = {}

    while viewer.is_running() and not key_callback.exit and stage is not None:
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        initial_T = end_effector_task.transform_target_to_world
        rotation_matrix = initial_T.as_matrix()[:3, :3]

        context = collect_context(object_pos, place_pos)
        finished, target_return = stage_functions[stage](context, params)

        if stage == "retreat" and finished:
            return
        
        if target_return is not None:
            target = target_return

        if finished is True:
            remaining.remove(stage)
            stage = stage_controller(stage, context, remaining)
            params = pose_controller(stage, data)
            print(f"Remaining stages after remove {stage}:", remaining)

        print(f"Current target: {target}")
        # IK 控制流程
        T_goal = mink.SE3.from_matrix(np.vstack([
            np.hstack([rotation_matrix, target.reshape(3, 1)]),
            np.array([0, 0, 0, 1])
        ]))
        end_effector_task.set_target(T_goal)

        for _ in range(1):
            
            vel = mink.solve_ik(configuration, [*tasks, damping_task], dt, "osqp", 1e-5)
            # print("IK velocity:", vel)
            configuration.integrate_inplace(vel, dt)
            err = end_effector_task.compute_error(configuration)
            # print("Position error:", np.linalg.norm(err[:3]), "Orientation error:", np.linalg.norm(err[3:]))
            if np.linalg.norm(err[:3]) < 1e-5:
                break

        # 控制与视频录制
        if not key_callback.pause:
            data.ctrl[actuator_ids] = configuration.q[dof_ids]
            mujoco.mj_step(model, data)
            for name, cam in cameras.items():
                renderer.update_scene(data, cam)
                frame = renderer.render()
                writers[name].append_data(frame)
        else:
            mujoco.mj_forward(model, data)

        viewer.sync()
        rate.sleep()
        if key_callback.exit:
                break


        T_goal = mink.SE3.from_matrix(np.vstack([
            np.hstack([rotation_matrix, target.reshape(3, 1)]),
            np.array([0, 0, 0, 1])
        ]))
        end_effector_task.set_target(T_goal)
        # for _ in range(8):
        #     vel = mink.solve_ik(configuration, [*tasks, damping_task], dt, "osqp", 1e-5)
        #     configuration.integrate_inplace(vel, dt)
        #     err = end_effector_task.compute_error(configuration)
        #     if np.linalg.norm(err[:3]) < 1e-5:
        #         break

        vel = mink.solve_ik(configuration, [*tasks, damping_task], dt, "osqp", 1e-5)

        vel = np.clip(vel, -max_vel, max_vel)


        acc = (vel - vel_prev) / dt
        acc = np.clip(acc, -max_acc, max_acc)
        vel = vel_prev + acc * dt
        vel = np.clip(vel, -max_vel, max_vel)

        configuration.integrate_inplace(vel, dt)
        vel_prev = vel.copy()


        data.ctrl[actuator_ids] = configuration.q[dof_ids]
        mujoco.mj_step(model, data)


        # Dataset recording

        state_arm = torch.as_tensor(state_, dtype=torch.float32)                           # [5]
        jaw_state = torch.tensor([data.qpos[jaw_joint_id]], dtype=torch.float32)           # [1]
        state = torch.cat([state_arm, jaw_state])      
        state = torch.rad2deg(state)                                                       # [6]

        action_arm = torch.as_tensor(data.ctrl[actuator_ids].copy(), dtype=torch.float32)  # [5]
        jaw_action = torch.tensor([data.ctrl[jaw_act_id]], dtype=torch.float32)            # [1]
        action = torch.cat([action_arm, jaw_action])   
        action = torch.rad2deg(action)                                                     # [6]
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"]  = state
        action_dict["action"] = action

        for name, cam in cameras.items():
            renderer.update_scene(data, cam)
            rendered_img = renderer.render()
            if rendered_img.dtype != np.uint8:
                rendered_img = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
            rendered_tensor = torch.from_numpy(rendered_img.copy()) 
            obs_dict[f"observation.images.{name}"] = rendered_tensor
        frame = {**obs_dict, **action_dict}
        dataset.add_frame(frame)

        viewer.sync()
        rate.sleep()
        if key_callback.exit:
                break

if __name__ == "__main__":
    main()
