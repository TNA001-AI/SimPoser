# import numpy as np
# import mujoco
# import mujoco.viewer as viewer
# import imageio


# #Video Setup
# DURATION = 4 #(seconds)
# FRAMERATE = 60 #(Hz)
# frames = []

# model = mujoco.MjModel.from_xml_path("so_arm100/scene.xml")
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model)
# #Reset state and time.
# mujoco.mj_resetData(model, data)

# #Init position.
# # pi = np.pi
# # data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0] #ENABLE if you want test circle

# #Init parameters
# jacp = np.zeros((3, model.nv)) #translation jacobian
# jacr = np.zeros((3, model.nv)) #rotational jacobian
# step_size = 0.5
# tol = 0.01
# alpha = 0.5
# damping = 0.15

# home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
# mujoco.mj_resetDataKeyframe(model, data, home_id)

# #Get error.
# end_effector_id = model.body('Fixed_Jaw').id #"End-effector we wish to control.
# current_pose = data.body(end_effector_id).xpos #Current pose

# goal = [0.1385675,0.06670183,0.35506766]

# error = np.subtract(goal, current_pose) #Init Error
# print("current_pose:", current_pose)

# def check_joint_limits(q):
#     """Check if the joints is under or above its limits"""
#     for i in range(len(q)):
#         q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

# def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
#     """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
#     as a function of time t and frequency f."""
#     x = r * np.cos(2 * np.pi * f * t) + h
#     y = r * np.sin(2 * np.pi * f * t) + k
#     z = 0.5
#     return np.array([x, y, z])

# #Simulate
# while data.time < DURATION:
    
#     # goal = circle(data.time, 0.1, 0.5, 0.0, 0.5) #ENABLE to test circle.
    
#     if (np.linalg.norm(error) >= tol):
#         #Calculate jacobian
#         mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
#         #Calculate delta of joint q
#         n = jacp.shape[1]
#         I = np.identity(n)
#         product = jacp.T @ jacp + damping * I

#         if np.isclose(np.linalg.det(product), 0):
#             j_inv = np.linalg.pinv(product) @ jacp.T
#         else:
#             j_inv = np.linalg.inv(product) @ jacp.T

#         delta_q = j_inv @ error

#         #Compute next step
#         q = data.qpos.copy()
#         q += step_size * delta_q
        
#         #Check limits
#         check_joint_limits(data.qpos)
        
#         #Set control signal
#         data.ctrl = q 
#         #Step the simulation.
#         mujoco.mj_step(model, data)

#         error = np.subtract(goal, data.body(end_effector_id).xpos)
#         # print("current_pose:",data.body(end_effector_id).xpos)
#         # print("Error =>", error)

#     #Render and save frames.
#     if len(frames) < data.time * FRAMERATE:
#         renderer.update_scene(data)
#         pixels = renderer.render()
#         frames.append(pixels)
        
# #Display video.
# imageio.mimsave("ik_result.gif", frames, fps=FRAMERATE)
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
goal = np.array([0.1385675, 0.06670183, 0.35506766])
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
    while data.time < DURATION:
        # Compute error
        error = goal - data.body(end_effector_id).xpos

        if np.linalg.norm(error) >= tol:
            # Compute Jacobian
            mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
            n = jacp.shape[1]
            I = np.identity(n)
            product = jacp.T @ jacp + damping * I

            # Compute pseudo-inverse
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

        # Render high-res frame and save to video
        renderer.update_scene(data, camera)
        frame = renderer.render()
        writer.append_data(frame)

        # Sync viewer
        v.sync()

# Cleanup
writer.close()
renderer.close()
print(f"Video saved to {VIDEO_PATH}")
