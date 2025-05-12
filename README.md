# Simulated Dataset Generation for Imitation Learning with Large Language Models

To try out the LLM generation demo, please follow the below steps:

Ubuntu:

```
conda create -n simposer python=3.13 numpy matplotlib
conda activate simposer
pip install mujoco imageio[ffmpeg]

conda install conda-forge::dm_control
pip install "mink[examples]"

conda activate simposer
cd simposer
python script/pick_and_place_video_llm.py
```

macOS:
```
conda create -n simposer python=3.13
conda activate simposer
pip install requirements.txt

cd simposer
`mjpython script/pick_and_place_video_llm.py`
```

