from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robust-layernorm",
    version="0.1.0",
    author="Your Name",
    author_email="mohsen.mostafa.ai@outlook.com",
    description="Robust Layer Normalization with Adaptive Noise Suppression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/R-LayerNorm/R-LayerNorm/tree/main",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)
