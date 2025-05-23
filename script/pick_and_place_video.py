import warnings

warnings.filterwarnings("ignore", message="Converted P to scipy.sparse.csc.csc_matrix")
warnings.filterwarnings("ignore", message="Converted G to scipy.sparse.csc.csc_matrix")

import os
import numpy as np
import mujoco
import mujoco.viewer
import imageio
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter
import mink
from dm_control.viewer import user_input

RANDOM = True
EPOCH = 10
_XML = "so_arm100/scene.xml"
open_gripper = 0.3
close_gripper = -0.0
RENDER_WIDTH, RENDER_HEIGHT = 480, 640
CAMERA_NAMES = ["top_view", "front_view", "main_cam"]
FPS = 30
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
# print(renderer.context.device) 
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
damping_cost[:] = 1e-3
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


def main():
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:

        for epoch in range(EPOCH):  
            print(f"\n🚀 Starting epoch {epoch + 1}")
            object_pos = reset_scene()

            writers = {}
            for name in CAMERA_NAMES:
                video_path = f"{VIDEO_DIR}/{name}_epoch{epoch + 1}.mp4"
                writers[name] = imageio.get_writer(video_path, fps=FPS)

            run_epoch(object_pos, writers, viewer)

            for writer in writers.values():
                writer.close()

            if key_callback.exit:
                break

    renderer.close()
    print(f"✅ Saved videos for {EPOCH} epochs to {VIDEO_DIR}")

def reset_scene():
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)

    end_effector_task.set_target_from_configuration(configuration)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)

    # domain randomization
    if RANDOM:
        randomize_domain()

    object_pos = np.random.uniform(low=[0.2, -0.1, 0.01], high=[0.3, 0.1, 0.01])
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



def run_epoch(object_pos, writers,viewer):
    place_pos = np.array([0.3, 0.0, 0.03])
    data.site_xpos[place_site_id] = place_pos
    grasp_height = 0.018
    lift_height = 0.15
    stage = "approach"

    while viewer.is_running() and not key_callback.exit:
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        ee_T = configuration.get_transform_frame_to_world("gripper_site", "site")
        ee_pos = ee_T.translation()
        initial_T = end_effector_task.transform_target_to_world
        rotation_matrix = initial_T.as_matrix()[:3, :3]
        jaw_angle = data.qpos[jaw_joint_id]
        # print("Stage:", stage)
        if stage == "approach":
            target = object_pos + np.array([0.0, 0.0, 0.10])
            # print(np.linalg.norm(ee_pos - target))
            if np.linalg.norm(ee_pos - target) < 0.01:
                data.ctrl[jaw_act_id] = open_gripper  
                stage = "wait_open"

        elif stage == "wait_open":
            if jaw_angle >= open_gripper - 0.01:
                stage = "grasp"

        elif stage == "grasp":
            target = object_pos + np.array([0.0, 0.0, grasp_height])
            if np.linalg.norm(ee_pos - target) < 0.01:
                data.ctrl[jaw_act_id] = close_gripper 
                stage = "wait_close"

        elif stage == "wait_close":
            contact_fixed = False
            contact_moving = False
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
                stage = "lift"

        elif stage == "lift":
            target = object_pos + np.array([0.0, 0.0, lift_height])
            if np.linalg.norm(ee_pos - target) < 0.01:
                stage = "move"

        elif stage == "move":
            target = place_pos + np.array([0.0, 0.0, lift_height])
            # print(np.linalg.norm(ee_pos - target))
            if np.linalg.norm(ee_pos - target) < 0.02:
                stage = "place"

        elif stage == "place":
            end_effector_task.orientation_cost = 0.1
            target = place_pos + np.array([0.0, 0.0, grasp_height])
            rotation_matrix = np.array([
                [0,  0,  1],
                [1,  0,  0],
                [0,  1 , 0]
            ])
            if np.linalg.norm(ee_pos - target) < 0.01:
                end_effector_task.orientation_cost = 0.0 
                data.ctrl[jaw_act_id] = open_gripper  
                stage = "wait_open_place"

        elif stage == "wait_open_place":
            if jaw_angle >= open_gripper - 0.01:
                stage = "retreat"

        elif stage == "retreat":
            target = np.array([0.2, 0.0, 0.15])
            data.ctrl[jaw_act_id] = close_gripper
            if np.linalg.norm(ee_pos - target) < 0.01:
                print("✅ One epoch completed.")
                return  # 结束当前 epoch

        else:
            target = ee_pos
            print("Invalid stage:", stage)

        # IK 控制流程
        T_goal = mink.SE3.from_matrix(np.vstack([
            np.hstack([rotation_matrix, target.reshape(3, 1)]),
            np.array([0, 0, 0, 1])
        ]))
        end_effector_task.set_target(T_goal)

        for _ in range(8):
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

if __name__ == "__main__":
    main()
