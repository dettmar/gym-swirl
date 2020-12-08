from gym.envs.registration import register

register(
    id='swirl-v1',
    entry_point='gym_swirl.envs:Swirl',
)
