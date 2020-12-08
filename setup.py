from setuptools import setup

# Based on boiler plate https://github.com/openai/gym/blob/master/docs/creating-environments.md

setup(name='gym-swirl',
	version='0.1',
	description="Upscale.app's backend upscaling worker",
	author="Johan Dettmar",
	url="https://github.com/dettmar/swirl",
	install_requires=['gym', 'torch', 'torchdiffeq']
)
