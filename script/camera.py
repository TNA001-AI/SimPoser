import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt


model = mujoco.MjModel.from_xml_path("so_arm100/scene.xml")
data = mujoco.MjData(model)

# 初始化 renderer
renderer = mujoco.Renderer(model)

# 设置你要渲染的摄像头
camera_id = 1 # 或者用 index，例如 0

# 渲染一帧
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera=camera_id)
img = renderer.render()

# 显示图像
plt.imshow(img)
plt.axis("off")
plt.title("top_view camera image")
plt.show()
