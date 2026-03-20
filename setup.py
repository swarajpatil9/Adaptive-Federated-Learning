from setuptools import setup, find_packages

setup(
    name="aflf",
    version="0.1.0",
    description="Adaptive Federated Learning Framework",
    author="Swaraj Patil",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.3.0",
        "opacus>=1.4.0",
        "pyyaml>=6.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
