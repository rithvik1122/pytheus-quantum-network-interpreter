#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pytheus-quantum-network-interpreter",
    version="1.0.0",
    author="PyTheus Quantum Network Interpreter Team",
    author_email="research.team@quantum-lab.org",
    description="A comprehensive interpreter for analyzing and visualizing PyTheus-optimized quantum networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pytheus-quantum-network-interpreter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="quantum networks, PyTheus, automated design, network interpretation, quantum key distribution",
    project_urls={
        "Bug Reports": "https://github.com/your-username/pytheus-quantum-network-interpreter/issues",
        "Source": "https://github.com/your-username/pytheus-quantum-network-interpreter",
        "Documentation": "https://github.com/your-username/pytheus-quantum-network-interpreter/tree/main/docs",
    },
)
