# Description: This file contains the setup.py of the AI Pipeline.
# Maintainers: Diego Colombo and Tara E. Walker

from setuptools import setup, find_packages
VERSION = '1.0'
DESCRIPTION = 'AI Pipeline'

setup(
    name='aipipeline', 
    version=VERSION, 
    packages=find_packages(include=['aipipeline', 'aipipeline.*']),
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'transformers',
        'pydantic',
        'fastapi',
        'uvicorn',
        'llama-index',
        ]
)