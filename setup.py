from setuptools import setup, find_packages

setup(
    name="dpasa",
    version="1.0.0",
    description="Dual-Population Adversarial Strategy Adaptation (DPASA)",
    author="Haythem Ghazouani",
    author_email="haythem.ghazouani@enicar.u-carthage.tn",
    url="https://github.com/haythemghz/DPASA-algorithm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
