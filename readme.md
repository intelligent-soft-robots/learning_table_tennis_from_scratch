# Instructions

- install the required packages. See these [instructions](https://github.com/intelligent-soft-robots/intelligent-soft-robots.github.io/wiki/01_installation-from-debian)

Assuming dev/setup.bash is being sourced in all new terminal, to run things:

- if running in real time :
    - from a terminal execute : start_robots
    - in another terminal, python run ```python hysr_one_ball_random.py``` or ```python hysr_one_ball_swing.py```

- if running in accelerated time:
    - from a terminal execute the executable start_robots_accelerated
    - in another terminal, python run ```python hysr_one_ball_random.py --accelerated``` or ```python hysr_one_ball_swing.py --accelerated```
