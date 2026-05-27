import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="flexllm",
    version="0.1.0",
    author="Lxn",
    author_email="liangxuning@126.comm",
    description="Flexible LLM inference scheduler with hybrid CPU-GPU KV Cache management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lxnlxnlxnlxnlxn/flexllm",
    license="MIT",
    packages=find_packages(exclude=["tests"]), 
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    # entry_points={
    #     "console_scripts": [
    #         "flexllm-sim=flexllm.simulator:main",
    #     ],
    # },
)