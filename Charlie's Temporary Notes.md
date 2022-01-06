* I got an error installing mujoco (using `pip install -r requirements.txt`)
    * I fixed this with:
    * ```sudo apt install gcc & pip install mujoco_py==2.0.2.8```

* Troubleshooting: try to install these: `sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
`
* openai safety gym wasnt working so I've added as a submodule
* mujoco200 seems to be required for mujoco-py==2.0.7 which is required for building from egg
* Instead, install mujoco manually (if you have to) here: https://github.com/openai/mujoco-py
* 