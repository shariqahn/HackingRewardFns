from gym.envs.registration import register

register(
    id='slider-v0',
    entry_point='gym_slider.envs:SliderEnv',
)