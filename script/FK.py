# import mujoco
# import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("so_arm100/scene.xml")
# data = mujoco.MjData(model)


# model.opt.timestep  = 0.01 # s


# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("MuJoCo GUI launched. Close the window to exit.")
#     while viewer.is_running():
#         mujoco.mj_step(model, data)
#         viewer.sync()


import mujoco
import mujoco.viewer
import numpy as np
import time

import numpy as np
import time

class FifthOrderInterpolator:
    def __init__(self, q_start, q_end, duration):
        """
        Initialize the 5th-order polynomial interpolator.

        Parameters:
        - q_start: np.array, initial joint positions
        - q_end: np.array, target joint positions (same shape as q_start)
        - duration: float, interpolation duration in seconds
        """
        assert q_start.shape == q_end.shape, "q_start and q_end must have the same shape"
        self.q_start = q_start.copy()
        self.q_end = q_end.copy()
        self.duration = duration
        self.start_time = None

    def reset(self):
        """Reset the interpolation timer to start from current time."""
        self.start_time = time.time()

    def step(self):
        """
        Compute the interpolated joint positions at the current time.

        Returns:
        - q: np.array, interpolated joint positions
        - done: bool, True if interpolation is complete
        """
        if self.start_time is None:
            self.reset()

        t = time.time() - self.start_time
        tau = np.clip(t / self.duration, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        q = (1 - s) * self.q_start + s * self.q_end
        done = t >= self.duration
        return q, done


model = mujoco.MjModel.from_xml_path("so_arm100/scene.xml")
data = mujoco.MjData(model)

model.opt.timestep = 0.01

home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, home_id)

# Interpolator parameters
q_start = data.qpos.copy()
q_end_partial = np.array([0.8, -0.7, -0.7, -0.7, 0.5])
q_end = q_start.copy()
q_end[:5] = q_end_partial

interpolator = FifthOrderInterpolator(q_start, q_end, duration=10.0)
interpolator.reset()

end_effector_id = model.body('Fixed_Jaw').id #"End-effector we wish to control.


with mujoco.viewer.launch_passive(model, data) as viewer:
    print("MuJoCo GUI launched. Close the window to exit.")

    while viewer.is_running():
        q_interp, done = interpolator.step()
        data.qpos[:] = q_interp
        mujoco.mj_forward(model, data)

        current_pose = data.body(end_effector_id).xpos #Current pose       
        print("current_pose:", current_pose)
        
        viewer.sync()
        time.sleep(model.opt.timestep)