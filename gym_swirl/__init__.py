from gym.envs.registration import register

register(
    id='swirl',
    entry_point='gym_swirl.envs:ActiveParticles',
)
