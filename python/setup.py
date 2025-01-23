from setuptools import setup, find_namespace_packages

setup(
    name="neural_networks",
    version="0.1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pytest>=6.0.0",
        "safetensors>=0.4.0",
        "PyQt6>=6.0.0",
        "Pillow>=8.0.0",
    ],
) 