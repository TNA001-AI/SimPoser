# SimPoser

To run the project, please follow the below steps:

Ubuntu:

```
conda create -n simposer python=3.13 numpy matplotlib
conda activate simposer
pip install mujoco imageio[ffmpeg]

conda activate simposer
cd simposer
pyhton ./script/IK.py
```

macOS:
```
conda create -n simposer python=3.13 numpy matplotlib
conda activate simposer
pip install "imageio[ffmpeg]" mujoco

cd simposer
mjpython ./script/IK.py 
```
