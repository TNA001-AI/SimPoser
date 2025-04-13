import numpy as np
import mujoco
import mujoco.viewer as viewer
import imageio

# Simulation settings
DURATION = 4  # seconds
FRAMERATE = 60  # Hz
step_size = 0.5
tol = 0.01
damping = 0.15
VIDEO_PATH = "ik_result_hd.mp4"

# Load model and data
model = mujoco.MjModel.from_xml_path("so_arm100/scene.xml")
data = mujoco.MjData(model)

# Reset to keyframe named "home"
home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)

# Initialize Jacobians
jacp = np.zeros((3, model.nv))  # translation
jacr = np.zeros((3, model.nv))  # rotation

# Set target (goal) and end-effector
end_effector_id = model.body('Fixed_Jaw').id
goal = np.array([0.30, 0.0, 0.35])
error = goal - data.body(end_effector_id).xpos.copy()

def check_joint_limits(q):
    """Ensure joint angles stay within their limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

# Setup high-res renderer
renderer = mujoco.Renderer(model, height=1080, width=1920)
# Configure a custom camera
# Use named camera from XML
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_cam")


# Initialize video writer
writer = imageio.get_writer(VIDEO_PATH, fps=FRAMERATE)

# Launch viewer
with viewer.launch_passive(model, data) as v:
    render_interval = 1.0 / FRAMERATE
    next_frame_time = 0.0

    while data.time < DURATION:
        # 控制器逻辑
        error = goal - data.body(end_effector_id).xpos
        if np.linalg.norm(error) >= tol:
            mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
            n = jacp.shape[1]
            I = np.identity(n)
            product = jacp.T @ jacp + damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp.T
            else:
                j_inv = np.linalg.inv(product) @ jacp.T

            delta_q = j_inv @ error
            q = data.qpos.copy()
            q += step_size * delta_q
            check_joint_limits(q)
            data.ctrl[:] = q

        # Step simulation
        mujoco.mj_step(model, data)

        # 渲染帧控制
        if data.time >= next_frame_time:
            renderer.update_scene(data, camera)
            frame = renderer.render()
            writer.append_data(frame)
            next_frame_time += render_interval

        v.sync()


# Cleanup
writer.close()
renderer.close()
print(f"Video saved to {VIDEO_PATH}")
