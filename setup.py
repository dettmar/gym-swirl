from setuptools import setup

# Based on boiler plate https://github.com/openai/gym/blob/master/docs/creating-environments.md

setup(name="gym-swirl",
	version="0.1",
	description="Simulation of Active Colloids with Social Interaction",
	author="Johan Dettmar",
	url="https://github.com/dettmar/gym-swirl",
	license="MIT",
	install_requires=["gym", "torch", "matplotlib", "jupyter"]
)
