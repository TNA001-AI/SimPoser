# SimPoser

To run the project, please follow the below steps:

Ubuntu:

```
conda create -n simposer python=3.13 numpy matplotlib
conda activate simposer
pip install mujoco imageio[ffmpeg]
# New
conda install conda-forge::dm_control
pip install "mink[examples]"


conda activate simposer
cd simposer
python ./script/test_mink.py
```

macOS:
```
conda create -n simposer python=3.13 numpy matplotlib
conda activate simposer
pip install "imageio[ffmpeg]" mujoco

cd simposer
mjpython ./script/IK.py 
```
