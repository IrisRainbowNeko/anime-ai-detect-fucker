from setuptools import setup, find_packages

setup(
    name='anime-ai-detect-attacker',
    version='1.0',
    description='anime-ai-detect-attacker',
    packages=find_packages(),  # same as name
    install_requires=['numpy'],  # external packages as dependencies
)