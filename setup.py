"""
Setup script for DLSA package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dlsa",
    version="1.0.0",
    author="Reimplementation",
    description="Deep Learning Statistical Arbitrage - Clean Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/dlsa-clean",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "data": [
            "pandas-datareader>=0.10.0",
        ],
    },
    # Scripts are in root scripts/ folder, run directly:
    # python scripts/run_simulation.py
    # python scripts/run_factor_model.py
    # python scripts/run_stats.py
)
