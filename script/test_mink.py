from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter

import mink

_XML = "so_arm100/scene.xml"

@dataclass
class KeyCallback:
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_SPACE:
            self.pause = not self.pause

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML)
    data = mujoco.MjData(model)

    joint_names = [
        "Rotation", "Pitch", "Elbow",
        "Wrist_Pitch", "Wrist_Roll", "Jaw"
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    configuration = mink.Configuration(model)

    # Control task for the robot arm end-effector (e.g., gripper center)
    end_effector_task = mink.FrameTask(
        frame_name="gripper_site", 
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,  # Set to 0.0 to ignore orientation due to 5-DOF
        lm_damping=1e-2,
    )

    posture_cost = np.zeros((model.nv,))
    posture_task = mink.PostureTask(model, posture_cost)

    damping_cost = np.zeros((model.nv,))
    damping_cost[:] = 1e-3
    damping_task = mink.DampingTask(model, damping_cost)

    tasks = [
        end_effector_task,
        posture_task,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)

        end_effector_task.set_target_from_configuration(configuration)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        goal = np.array([0.3, 0.1, 0.3])
        # site_id = model.site("goal_site").id 
        initial_T = end_effector_task.transform_target_to_world
        rotation_matrix = initial_T.as_matrix()[:3, :3]

        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.period

        while viewer.is_running():
            # Visualize the target site
            # data.site_xpos[site_id] = goal
            # mujoco.mj_forward(model, data) 

            # Construct target SE3 pose (target position + fixed orientation)
            T_goal = mink.SE3.from_matrix(np.vstack([
                np.hstack([rotation_matrix, goal.reshape(3, 1)]),
                np.array([0, 0, 0, 1])
            ]))
            end_effector_task.set_target(T_goal)

            for _ in range(50):
                vel = mink.solve_ik(configuration, [*tasks, damping_task], dt, "osqp", 1e-5)
                configuration.integrate_inplace(vel, dt)

                err = end_effector_task.compute_error(configuration)
                # print("Position error:", np.linalg.norm(err[:3]), "Orientation error:", np.linalg.norm(err[3:]))
                if np.linalg.norm(err[:3]) < 1e-5:
                    break

            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            viewer.sync()
            rate.sleep()

            # # Print current end-effector position
            # # Use data.qpos to sync with the viewer
            # configuration.update(data.qpos)
            # current_T = configuration.get_transform_frame_to_world("gripper_site", "site")
            # current_pos = current_T.translation()
            # rotation_matrix = current_T.rotation().as_matrix()
            # print("Current orientation (rotation matrix):\n", rotation_matrix)
            # # print(f"Current end-effector position: x={current_pos[0]:.4f}, y={current_pos[1]:.4f}, z={current_pos[2]:.4f}")
